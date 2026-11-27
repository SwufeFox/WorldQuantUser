"""Resident single-writer daemon, HTTP loopback IPC, and non-blocking CLI handlers."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cache_store
import machine_lib as ml
import workspace_db

from . import buffer_cmd, sm_workers, state_machine
from .runtime import CliError

_PRIORITY_MAP = {
    "AUTO_CANCELLING": 0,
    "AUTO_SUBMITTING": 1,
    "AUTO_BUFFER": 2,
    "AUTO_HYDRATE_PNL": 6,
    "AUTO_PREFLIGHT": 7,
}


def _run_priority(doc):
    st = doc.get("state")
    if st in _PRIORITY_MAP:
        return _PRIORITY_MAP[st]
    if st == "AUTO_FULL_VALIDATE":
        return 3 if doc.get("full_records") else 5
    if st == "AUTO_QUICK_SIM":
        accepted = ((doc.get("execution") or {}).get("accepted") or 0)
        return 4 if accepted > 0 else 8
    return 99


def _request_integer(value, name, *, minimum=0):
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise CliError("INVALID_INPUT", f"{name} must be an integer", f"send an integer for {name}") from exc
    if not minimum <= integer <= sys.maxsize:
        raise CliError("INVALID_INPUT", f"{name} must be between {minimum} and {sys.maxsize}",
                       f"send an integer in the supported range for {name}")
    return integer


def _wait_seconds(value):
    try:
        seconds = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError, OverflowError):
        seconds = math.nan
    if not math.isfinite(seconds) or seconds < 0:
        raise CliError("INVALID_INPUT", "max_wait_seconds must be a finite non-negative number",
                       "send a numeric wait duration in seconds")
    return min(seconds, 60.0)


def _gate_ready(engine, *, run_id=None, gate=None):
    if run_id is not None:
        try:
            doc = engine.load_run_raw(run_id)
        except CliError as exc:
            if exc.code != "NOT_FOUND":
                raise
            return {"ready": True, "reason": "not_found", "run": None}
        st = doc.get("state")
        if st in (state_machine.GATE_STATES | state_machine.TERMINAL_STATES) and (gate is None or st == gate):
            return {"ready": True, "reason": "target_reached", "run": engine.read_run_view(doc, tool="wait")}
        return None
    inbox = engine.inbox(gate=gate)
    if inbox["actionable"]:
        return {"ready": True, "reason": "gate_ready", "inbox": inbox}
    return None


class DaemonServer:
    """Single-writer resident daemon driving the persistent state machine over loopback HTTP."""

    def __init__(self, engine, *, host="127.0.0.1", port=0, session_factory=None):
        """Bind *engine* to a daemon instance ready to serve on *host*:*port*."""
        self.engine = engine
        self.host = host
        self.port = port
        self.session_factory = session_factory or ml.load_session
        self.token = secrets.token_hex(32)
        self.lock_path = engine.daemon_dir / "daemon.lock"
        self.endpoint_path = engine.daemon_dir / "endpoint.json"
        self._stop_event = threading.Event()
        self._wake_cond = threading.Condition()
        self._gate_cond = threading.Condition()
        self._lock_ctx = None
        self._store_ctx = None
        self._httpd = None
        self._http_thread = None
        self._worker_thread = None
        self._last_reconcile = 0.0
        self._last_error = None
        self._started_at = None

    def start(self):
        """Acquire daemon lock, start HTTP + scheduler threads, and publish endpoint.json."""
        if self._httpd is not None:
            return self.endpoint_info()
        self._lock_ctx = cache_store.file_lock(self.lock_path, timeout=0)
        try:
            self._lock_ctx.__enter__()
        except cache_store.LockTimeout as exc:
            self._lock_ctx = None
            raise CliError("SESSION_BUSY", "another daemon already owns this research directory",
                           "run `research_cli.py daemon status` or `daemon stop` before starting another") from exc

        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):  # noqa: A002 - silence stderr HTTP access logs
                return

            def _send(self, status, payload):
                raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                try:
                    self.end_headers()
                    self.wfile.write(raw)
                except ConnectionError:
                    self.close_connection = True

            def _authorized(self):
                header = self.headers.get("Authorization") or ""
                expected = f"Bearer {server_ref.token}"
                return hmac.compare_digest(header.encode("utf-8"), expected.encode("utf-8"))

            def _read_json(self):
                length = _request_integer(self.headers.get("Content-Length", "0"), "Content-Length")
                if length == 0:
                    return {}
                raw = self.rfile.read(length)
                try:
                    data = json.loads(raw.decode("utf-8"))
                except ValueError as exc:
                    raise CliError("INVALID_INPUT", f"invalid request JSON: {exc}",
                                   "send a valid JSON object body") from exc
                if not isinstance(data, dict):
                    raise CliError("INVALID_INPUT", "request body must be a JSON object",
                                   "send a JSON object body")
                return data

            def do_GET(self):  # noqa: N802
                if not self._authorized():
                    self._send(403, {"ok": False, "error": {
                        "code": "ACCESS_DENIED", "message": "invalid or missing bearer token",
                        "hint": "use endpoint credentials from daemon/endpoint.json"}})
                    return
                parsed = urllib.parse.urlsplit(self.path)
                qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                try:
                    if parsed.path == "/v1/status":
                        self._send(200, {"ok": True, "data": server_ref.status_snapshot()})
                    elif parsed.path == "/v1/inbox":
                        gate = qs.get("gate", [None])[0] or None
                        limit = _request_integer(qs.get("limit", ["20"])[0], "limit", minimum=1)
                        self._send(200, {"ok": True, "data": server_ref.engine.inbox(gate=gate, limit=limit)})
                    elif parsed.path == "/v1/runs":
                        st = qs.get("state", [None])[0] or None
                        limit = _request_integer(qs.get("limit", ["50"])[0], "limit", minimum=1)
                        self._send(200, {"ok": True, "data": server_ref.engine.list_runs(state=st, limit=limit)})
                    elif parsed.path == "/v1/inspect":
                        run_id = qs.get("run_id", [None])[0]
                        arm = qs.get("arm", [None])[0] or None
                        self._send(200, {"ok": True, "data": server_ref.engine.inspect_run(run_id, arm_label=arm)})
                    else:
                        self._send(404, {"ok": False, "error": {
                            "code": "NOT_FOUND", "message": f"unknown route {parsed.path}",
                            "hint": "inspect daemon API routes in research_cli.py help"}})
                except CliError as exc:
                    status = 404 if exc.code == "NOT_FOUND" else 400
                    self._send(status, {"ok": False, "error": {
                        "code": exc.code, "message": exc.message, "hint": exc.hint, "details": exc.details}})
                except Exception as exc:  # noqa: BLE001
                    self._send(500, {"ok": False, "error": {
                        "code": "SERVICE_ERROR", "message": f"{type(exc).__name__}: {exc}",
                        "hint": "inspect daemon status or restart the daemon"}})

            def do_POST(self):  # noqa: N802
                if not self._authorized():
                    self._send(403, {"ok": False, "error": {
                        "code": "ACCESS_DENIED", "message": "invalid or missing bearer token",
                        "hint": "use endpoint credentials from daemon/endpoint.json"}})
                    return
                parsed = urllib.parse.urlsplit(self.path)
                try:
                    body = self._read_json()
                    if parsed.path == "/v1/propose":
                        view = server_ref.engine.propose(body.get("spec"), source_path=body.get("source_path"))
                        server_ref.notify_work()
                        self._send(200, {"ok": True, "data": view})
                    elif parsed.path == "/v1/act":
                        view = server_ref.engine.act(body.get("run_id"), body.get("action"), body.get("params"))
                        server_ref.notify_work()
                        if view.get("state") in (state_machine.GATE_STATES | state_machine.TERMINAL_STATES):
                            server_ref.notify_gate()
                        self._send(200, {"ok": True, "data": view})
                    elif parsed.path == "/v1/cancel":
                        view = server_ref.engine.cancel(body.get("run_id"), reason=body.get("reason"))
                        server_ref.notify_work()
                        server_ref.notify_gate()
                        self._send(200, {"ok": True, "data": view})
                    elif parsed.path == "/v1/config":
                        cfg = server_ref.engine.update_config(body)
                        server_ref.notify_work()
                        self._send(200, {"ok": True, "data": cfg})
                    elif parsed.path == "/v1/wait":
                        data = server_ref.wait_for_gate(
                            run_id=body.get("run_id"),
                            gate=body.get("gate"),
                            max_wait_seconds=body.get("max_wait_seconds", 30.0),
                        )
                        self._send(200, {"ok": True, "data": data})
                    elif parsed.path == "/v1/stop":
                        server_ref._stop_event.set()
                        server_ref.notify_work()
                        server_ref.notify_gate()
                        self._send(200, {"ok": True, "data": {"stopping": True, "pid": os.getpid()}})
                    else:
                        self._send(404, {"ok": False, "error": {
                            "code": "NOT_FOUND", "message": f"unknown route {parsed.path}",
                            "hint": "inspect daemon API routes in research_cli.py help"}})
                except CliError as exc:
                    status = 404 if exc.code == "NOT_FOUND" else 400
                    self._send(status, {"ok": False, "error": {
                        "code": exc.code, "message": exc.message, "hint": exc.hint, "details": exc.details}})
                except Exception as exc:  # noqa: BLE001
                    self._send(500, {"ok": False, "error": {
                        "code": "SERVICE_ERROR", "message": f"{type(exc).__name__}: {exc}",
                        "hint": "inspect daemon status or restart the daemon"}})

        try:
            store = workspace_db.keepalive(self.engine.root)
            store.__enter__()
            self._store_ctx = store
            self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
            self._httpd.daemon_threads = True
            self.port = self._httpd.server_address[1]
            self._started_at = time.time()
            cache_store.write_json_atomic(self.endpoint_path, self.endpoint_info(), mode=0o600)
            self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="wqu-daemon-http")
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="wqu-daemon-worker")
            self._http_thread.start()
            self._worker_thread.start()
            return self.endpoint_info()
        except Exception:
            self.stop()
            raise

    def endpoint_info(self):
        """Return endpoint descriptor dictionary written to ``endpoint.json``."""
        return {
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
            "token": self.token,
            "research_dir": str(self.engine.root),
            "started_at": self._started_at,
        }

    def status_snapshot(self):
        """Return live daemon status, config, and inbox counts."""
        inbox = self.engine.inbox()
        return {
            "running": not self._stop_event.is_set(),
            "pid": os.getpid(),
            "host": self.host,
            "port": self.port,
            "research_dir": str(self.engine.root),
            "uptime_seconds": round(time.time() - self._started_at, 2) if self._started_at else 0.0,
            "config": self.engine.load_config(),
            "counts_by_state": inbox["counts_by_state"],
            "actionable_total": inbox["actionable_total"],
            "active_auto_total": inbox["active_auto_total"],
            "last_error": self._last_error,
        }

    def notify_work(self):
        """Wake the background scheduler immediately."""
        with self._wake_cond:
            self._wake_cond.notify_all()

    def notify_gate(self):
        """Wake any caller blocked in ``/v1/wait``."""
        with self._gate_cond:
            self._gate_cond.notify_all()

    def wait_for_gate(self, *, run_id=None, gate=None, max_wait_seconds=30.0):
        """Block up to *max_wait_seconds* (capped at 60s) until a matching gate/terminal event arrives."""
        deadline = time.monotonic() + _wait_seconds(max_wait_seconds)
        with self._gate_cond:
            while True:
                hit = _gate_ready(self.engine, run_id=run_id, gate=gate)
                if hit is not None:
                    return {"woken": True, **hit}
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop_event.is_set():
                    return {"woken": False, "reason": "timeout", "inbox": self.engine.inbox(gate=gate)}
                self._gate_cond.wait(timeout=min(remaining, 1.0))

    def _worker_loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            try:
                cfg = self.engine.load_config()
                interval = int(cfg.get("reconcile_interval_seconds", 300))
                if interval > 0 and now - self._last_reconcile >= interval:
                    self._last_reconcile = now
                    try:
                        sm_workers.reconcile_orphan_slots(self.engine, self.session_factory)
                    except ml.CaptchaRequired as exc:
                        self._last_error = f"{exc.code}: {exc}"
                        return
                    except Exception as exc:  # noqa: BLE001
                        self._last_error = f"reconcile: {type(exc).__name__}: {exc}"

                active = self.engine.iter_runs(states=state_machine.AUTO_STATES)
                due = [doc for doc in active if doc.get("next_wake_at") is None or doc["next_wake_at"] <= now]
                due.sort(key=lambda d: (_run_priority(d), d.get("created_at") or "", d["run_id"]))
                if due:
                    target = due[0]
                    before_state = target.get("state")
                    after = sm_workers.advance_run_once(self.engine, target["run_id"], self.session_factory)
                    last_error = after.get("last_error") or {}
                    if last_error.get("code") == "AUTH_CAPTCHA":
                        self._last_error = f"AUTH_CAPTCHA: {last_error['message']}"
                        return
                    if after.get("state") in (state_machine.GATE_STATES | state_machine.TERMINAL_STATES) or after.get("state") != before_state:
                        self.notify_gate()
                    continue

                waiting_times = [
                    doc["next_wake_at"] - now
                    for doc in active
                    if isinstance(doc.get("next_wake_at"), (int, float))
                ]
                sleep_sec = max(0.05, min([1.0, *waiting_times])) if waiting_times else 1.0
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"worker: {type(exc).__name__}: {exc}"
                sleep_sec = 1.0

            with self._wake_cond:
                if self._stop_event.is_set():
                    break
                self._wake_cond.wait(timeout=sleep_sec)

    def stop(self):
        """Stop HTTP and the worker, then unpublish the endpoint and release daemon ownership."""
        if self._lock_ctx is None:
            return
        self._stop_event.set()
        self.notify_work()
        self.notify_gate()
        if self._httpd is not None:
            if self._http_thread is not None and self._http_thread.is_alive():
                self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._worker_thread is not None and self._worker_thread.ident is not None \
                and self._worker_thread is not threading.current_thread():
            # Retain ownership until the current step has saved its final transition.
            self._worker_thread.join()
        try:
            try:
                if self._store_ctx is not None:
                    self._store_ctx.__exit__(None, None, None)
                    self._store_ctx = None
            finally:
                cache_store.remove(self.endpoint_path)
        finally:
            if self._lock_ctx is not None:
                try:
                    self._lock_ctx.__exit__(None, None, None)
                finally:
                    self._lock_ctx = None


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _read_endpoint(engine):
    path = engine.daemon_dir / "endpoint.json"
    info, err = cache_store.load_json(path)
    if err or not isinstance(info, dict):
        return None
    if not info.get("host") or not info.get("port") or not info.get("token"):
        return None
    return info


def _http_call(endpoint, method, path, body=None, timeout=10.0):
    url = f"http://{endpoint['host']}:{endpoint['port']}{path}"
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {endpoint['token']}")
    if data is not None:
        req.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except ValueError as jexc:
            raise CliError("SERVICE_ERROR", f"daemon returned HTTP {exc.code}: {raw[:120]}",
                           "run `research_cli.py daemon status` to inspect daemon health") from jexc
        err = (payload.get("error") or {}) if isinstance(payload, dict) else {}
        raise CliError(
            err.get("code") or "SERVICE_ERROR",
            err.get("message") or f"daemon HTTP {exc.code}",
            err.get("hint") or "inspect daemon error details or run `research_cli.py daemon status`",
            details=err.get("details"),
        ) from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        err = (payload.get("error") or {}) if isinstance(payload, dict) else {}
        raise CliError(
            err.get("code") or "SERVICE_ERROR",
            err.get("message") or "daemon rejected request",
            err.get("hint") or "inspect daemon logs and `research_cli.py daemon status`",
            details=err.get("details"),
        )
    return payload["data"]


def _spawn_detached_daemon(engine):
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "research_cli.py"), "--research-dir", str(engine.root), "daemon", "_serve"]
    kwargs = {
        "cwd": str(repo_root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def _probe_endpoint(engine):
    endpoint = _read_endpoint(engine)
    if endpoint is not None:
        try:
            status = _http_call(endpoint, "GET", "/v1/status", timeout=1.0)
        except (OSError, CliError):
            return None
        if status.get("running"):
            return endpoint
    return None


def _spawn_and_wait(engine, timeout=15.0):
    deadline = time.monotonic() + timeout
    try:
        with cache_store.file_lock(engine.daemon_dir / "startup.lock", timeout=timeout):
            endpoint = _probe_endpoint(engine)
            if endpoint is not None:
                return endpoint
            proc = _spawn_detached_daemon(engine)
            while time.monotonic() < deadline:
                endpoint = _probe_endpoint(engine)
                if endpoint is not None:
                    return endpoint
                if proc.poll() is not None:
                    # A daemon launched elsewhere may own the lock and still be starting.
                    try:
                        with cache_store.file_lock(engine.daemon_dir / "daemon.lock", timeout=0):
                            return _probe_endpoint(engine)
                    except cache_store.LockTimeout:
                        pass
                time.sleep(0.05)
    except cache_store.LockTimeout:
        return _probe_endpoint(engine)
    return None


def _live_endpoint_or_none(engine, *, autostart=True):
    endpoint = _probe_endpoint(engine)
    if endpoint is not None or not autostart or os.environ.get("WQU_OFFLINE_TEST_WORKSPACE"):
        return endpoint
    return _spawn_and_wait(engine)


def _engine_for_args(args):
    return state_machine.StateMachineEngine(getattr(args, "research_dir", None))


def cmd_daemon_start(args, _sink):
    """Ensure the resident single-writer daemon is running for this research directory."""
    engine = _engine_for_args(args)
    ep = _live_endpoint_or_none(engine, autostart=False)
    if ep is None:
        # Force explicit single spawn even inside test workspaces when `daemon start` is invoked directly.
        ep = _spawn_and_wait(engine, timeout=15.0)
    if ep is None:
        raise CliError("SERVICE_ERROR", "daemon failed to become ready within 15 seconds",
                       f"check permissions on {engine.daemon_dir} and retry `daemon start`")
    return _http_call(ep, "GET", "/v1/status", timeout=2.0)


def cmd_daemon_stop(args, _sink):
    """Stop the resident daemon for this research directory cleanly."""
    engine = _engine_for_args(args)
    ep = _read_endpoint(engine)
    if ep is None:
        return {"running": False, "stopped": False, "reason": "not_running"}
    try:
        res = _http_call(ep, "POST", "/v1/stop", body={}, timeout=2.0)
        deadline = time.monotonic() + 3.0
        while _read_endpoint(engine) == ep:
            if time.monotonic() >= deadline:
                return {"running": True, "stopped": False, **res}
            time.sleep(0.05)
        return {"running": False, "stopped": True, **res}
    except OSError:
        try:
            with cache_store.file_lock(engine.daemon_dir / "daemon.lock", timeout=0):
                if _read_endpoint(engine) == ep:
                    cache_store.remove(engine.daemon_dir / "endpoint.json")
        except cache_store.LockTimeout as exc:
            raise CliError("SESSION_BUSY", "daemon is starting, stopping, or temporarily unreachable",
                           "retry `research_cli.py daemon stop` once the daemon responds") from exc
        return {"running": False, "stopped": False, "reason": "stale_endpoint_cleared"}


def cmd_daemon_status(args, _sink):
    """Return live daemon status when running, or offline state summary when stopped."""
    engine = _engine_for_args(args)
    ep = _read_endpoint(engine)
    if ep is not None:
        try:
            return _http_call(ep, "GET", "/v1/status", timeout=1.5)
        except (OSError, CliError):
            pass
    inbox = engine.inbox()
    return {
        "running": False,
        "research_dir": str(engine.root),
        "config": engine.load_config(),
        "counts_by_state": inbox["counts_by_state"],
        "actionable_total": inbox["actionable_total"],
        "active_auto_total": inbox["active_auto_total"],
    }


def cmd_daemon_config(args, _sink):
    """Inspect or update daemon scheduling parameters (quota floor, concurrency, backoff)."""
    engine = _engine_for_args(args)
    updates = {}
    for key in ("quota_floor", "concurrency", "multi_size", "max_unchanged_defers", "reconcile_interval_seconds"):
        val = getattr(args, key, None)
        if val is not None:
            updates[key] = val
    ep = _live_endpoint_or_none(engine, autostart=False)
    if updates:
        if ep is not None:
            return _http_call(ep, "POST", "/v1/config", body=updates, timeout=2.0)
        return engine.update_config(updates)
    if ep is not None:
        return _http_call(ep, "GET", "/v1/status", timeout=2.0)["config"]
    return engine.load_config()


def cmd_daemon_serve(args, _sink):
    """Foreground daemon server entry point invoked by ``daemon start``."""
    engine = _engine_for_args(args)
    server = DaemonServer(engine)
    info = server.start()
    try:
        while not server._stop_event.is_set():
            server._stop_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return {"stopped": True, "pid": info["pid"], "research_dir": info["research_dir"]}


def cmd_propose(args, _sink):
    """Submit a validated spec non-blockingly into AUTO_PREFLIGHT."""
    engine = _engine_for_args(args)
    spec_ref = getattr(args, "spec", None)
    if getattr(args, "stdin", False) or spec_ref == "-":
        raw = sys.stdin.read()
        source_path = None
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise CliError("INVALID_INPUT", f"stdin is not valid JSON: {exc}",
                           "pass valid spec JSON on stdin or pass a SPEC.json file path") from exc
    else:
        if not spec_ref:
            raise CliError("USAGE", "provide a spec file path or --stdin",
                           "run `research_cli.py propose SPEC.json`")
        src = Path(spec_ref).expanduser().resolve()
        if not src.is_file():
            raise CliError("NOT_FOUND", f"spec file not found: {src}",
                           "check the path to SPEC.json before running `propose`")
        source_path = str(src)
        try:
            payload = json.loads(src.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise CliError("INVALID_INPUT", f"cannot read spec {src.name}: {exc}",
                           "ensure the spec file is readable UTF-8 JSON") from exc

    if getattr(args, "name", None) and isinstance(payload, dict):
        payload = {**payload, "name": args.name}

    ep = _live_endpoint_or_none(engine, autostart=True)
    if ep is not None:
        return _http_call(ep, "POST", "/v1/propose", body={"spec": payload, "source_path": source_path}, timeout=5.0)
    return engine.propose(payload, source_path=source_path)


def cmd_inbox(args, _sink):
    """List runs currently paused at GATE_* interaction points and active AUTO counts."""
    engine = _engine_for_args(args)
    gate = getattr(args, "gate", None)
    limit = getattr(args, "limit", 20)
    ep = _live_endpoint_or_none(engine, autostart=False)
    if ep is not None:
        query = urllib.parse.urlencode({k: v for k, v in (("gate", gate), ("limit", limit)) if v is not None})
        return _http_call(ep, "GET", f"/v1/inbox?{query}", timeout=3.0)
    return engine.inbox(gate=gate, limit=limit)


def cmd_runs(args, _sink):
    """List state-machine runs under this workspace."""
    engine = _engine_for_args(args)
    st = getattr(args, "state", None)
    limit = getattr(args, "limit", 50)
    ep = _live_endpoint_or_none(engine, autostart=False)
    if ep is not None:
        query = urllib.parse.urlencode({k: v for k, v in (("state", st), ("limit", limit)) if v is not None})
        return _http_call(ep, "GET", f"/v1/runs?{query}", timeout=3.0)
    return engine.list_runs(state=st, limit=limit)


def cmd_inspect(args, _sink):
    """Inspect one run obeying the protocol-level Train/Test firewall."""
    engine = _engine_for_args(args)
    arm = getattr(args, "arm", None)
    ep = _live_endpoint_or_none(engine, autostart=False)
    if ep is not None:
        query = urllib.parse.urlencode({k: v for k, v in (("run_id", args.run_id), ("arm", arm)) if v is not None})
        return _http_call(ep, "GET", f"/v1/inspect?{query}", timeout=3.0)
    return engine.inspect_run(args.run_id, arm_label=arm)


def cmd_act(args, _sink):
    """Apply a gate transition decision to a run paused at a GATE_* state."""
    engine = _engine_for_args(args)
    params = {}
    for field in (
        "arms", "top", "verdict", "reason", "name", "category", "tags", "description",
        "trigger", "fields", "question", "expect", "branch", "dataset",
    ):
        val = getattr(args, field, None)
        if val is not None:
            params[field] = val
    if getattr(args, "yes", False):
        params["yes"] = True
    ep = _live_endpoint_or_none(engine, autostart=False)
    if ep is not None:
        return _http_call(ep, "POST", "/v1/act",
                          body={"run_id": args.run_id, "action": args.action, "params": params}, timeout=5.0)
    return engine.act(args.run_id, args.action, params)


def cmd_cancel(args, _sink):
    """Cancel an active run and revoke any live platform simulations."""
    engine = _engine_for_args(args)
    reason = getattr(args, "reason", None)
    ep = _live_endpoint_or_none(engine, autostart=False)
    if ep is not None:
        return _http_call(ep, "POST", "/v1/cancel", body={"run_id": args.run_id, "reason": reason}, timeout=5.0)
    return engine.cancel(args.run_id, reason=reason)


def cmd_wait(args, _sink):
    """Block efficiently until any run reaches an interaction gate or terminal state."""
    engine = _engine_for_args(args)
    run_id = getattr(args, "run_id", None)
    gate = getattr(args, "gate", None)
    max_wait = _wait_seconds(getattr(args, "max_wait", 30.0))
    ep = _live_endpoint_or_none(engine, autostart=False)
    if ep is not None:
        return _http_call(ep, "POST", "/v1/wait",
                          body={"run_id": run_id, "gate": gate, "max_wait_seconds": max_wait},
                          timeout=max_wait + 5.0)
    hit = _gate_ready(engine, run_id=run_id, gate=gate)
    if hit is not None:
        return {"woken": True, **hit}
    return {"woken": False, "reason": "daemon_not_running", "inbox": engine.inbox(gate=gate)}


def cmd_migrate_workspace(args, _sink):
    """Import legacy v3 backlog/ and backlog-results/ files into v4 runs/."""
    engine = _engine_for_args(args)
    return engine.migrate_workspace(dry_run=bool(getattr(args, "dry_run", False)))


def cmd_shelf_list(args, _sink):
    """List shelved candidates in submit-buffer.jsonl and report staleness."""
    return buffer_cmd.cmd_buffer_list(args, _sink)


def cmd_shelf_plan(args, _sink):
    """Compute a maximum-weight independent submission plan from shelved candidates."""
    data = buffer_cmd.cmd_buffer_list(args, _sink)
    rows = data.get("rows") or []
    planned = data.get("submission_plan")
    if planned is None:
        max_corr = float(getattr(args, "max_correlation", 0.7))
        planned = buffer_cmd._shelf_submission_plan(rows, argparse.Namespace(max_correlation=max_corr, cost_bps=getattr(args, "cost_bps", None))) if rows else {"selected": [], "rejected": []}
    return {"path": data.get("path"), "candidates": len(rows), "submission_plan": planned}
