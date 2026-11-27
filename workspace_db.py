"""SQLite WAL operational store for runs, slot journals, and Labs tickets."""

from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time

import cache_store
import research_paths

_DB_FILENAME = "workspace.db"
_PROCESS_LOCKS = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path(research_dir=None):
    """Return the path to ``<research_dir>/workspace.db``."""
    root = Path(research_dir or research_paths.ensure_dir()).expanduser().resolve()
    return root / _DB_FILENAME


def _thread_lock_for(path):
    key = str(Path(path).resolve())
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


def _init_schema(conn):
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if conn.execute("PRAGMA user_version").fetchone()[0] >= 1:
        return
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS kv_config (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            track TEXT,
            request_fingerprint TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            next_wake_at REAL,
            stalled INTEGER NOT NULL DEFAULT 0,
            doc_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runs_state_updated ON runs(state, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_runs_fingerprint ON runs(request_fingerprint);

        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            at TEXT NOT NULL,
            run_id TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT NOT NULL,
            trigger TEXT NOT NULL,
            detail_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, id);

        CREATE TABLE IF NOT EXISTS slot_journals (
            account_fingerprint TEXT PRIMARY KEY,
            journal_json TEXT NOT NULL,
            quota_json TEXT,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS labs_tickets (
            ticket_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            trigger_rule TEXT NOT NULL,
            region TEXT NOT NULL,
            universe TEXT NOT NULL,
            delay INTEGER NOT NULL,
            dataset TEXT,
            fields_json TEXT NOT NULL,
            questions_json TEXT NOT NULL,
            expected_metrics_json TEXT NOT NULL,
            branches_json TEXT NOT NULL,
            notebook_snippet TEXT NOT NULL,
            source_run_id TEXT,
            resolution_json TEXT,
            selected_branch TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_labs_status_updated ON labs_tickets(status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_labs_scope ON labs_tickets(region, universe, delay, status);
        PRAGMA user_version = 1;
        COMMIT;
        """
    )


@contextmanager
def connect(research_dir=None, *, create=True):
    """Open and cleanly close a SQLite WAL connection to ``workspace.db``."""
    target = db_path(research_dir)
    if not create and not target.exists():
        yield None
        return
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock_for(target):
        with closing(sqlite3.connect(str(target), timeout=30.0, isolation_level=None)) as conn:
            conn.row_factory = sqlite3.Row
            _init_schema(conn)
            yield conn


@contextmanager
def keepalive(research_dir):
    """Initialize SQLite and keep its WAL open across daemon requests."""
    target = db_path(research_dir)
    with _thread_lock_for(target):
        conn = sqlite3.connect(str(target), timeout=30.0, isolation_level=None, check_same_thread=False)
        try:
            _init_schema(conn)
        except BaseException:
            conn.close()
            raise
    try:
        yield
    finally:
        conn.close()


@contextmanager
def immediate_tx(research_dir=None, *, durable=False):
    """Execute SQLite statements inside an exclusive ``BEGIN IMMEDIATE`` transaction."""
    with connect(research_dir, create=True) as conn:
        if durable:
            conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


# ---------------------------------------------------------------------------
# Config & State-Machine Runs
# ---------------------------------------------------------------------------
def get_config(defaults, *, legacy=None, research_dir=None):
    """Load scheduling config merged over *defaults*."""
    cfg = {**defaults, **(legacy or {})}
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return cfg
        for row in conn.execute("SELECT key, value_json FROM kv_config WHERE key LIKE 'daemon.%'"):
            name = row["key"].removeprefix("daemon.")
            if name in cfg:
                try:
                    value = json.loads(row["value_json"])
                except ValueError:
                    continue
                if isinstance(value, (int, float)) and value >= 0:
                    cfg[name] = value
    return cfg


def set_config(cfg, *, research_dir=None):
    """Persist scheduling config key-value pairs into ``kv_config``."""
    now = _utc_now()
    with immediate_tx(research_dir) as conn:
        for key, value in cfg.items():
            conn.execute(
                """
                INSERT INTO kv_config (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                """,
                (f"daemon.{key}", json.dumps(value), now),
            )


def upsert_run(doc, *, from_state=None, trigger=None, detail=None, durable=False, research_dir=None):
    """Persist a run and its transition event in the same transaction."""
    with immediate_tx(research_dir, durable=durable) as conn:
        _write_run(conn, doc, from_state=from_state, trigger=trigger, detail=detail)


def _write_run(conn, doc, *, from_state=None, trigger=None, detail=None):
    now = doc.get("updated_at") or _utc_now()
    created = doc.get("created_at") or now
    blob = json.dumps(doc, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO runs (
            run_id, state, track, request_fingerprint,
            created_at, updated_at, next_wake_at, stalled, doc_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            state=excluded.state,
            track=excluded.track,
            request_fingerprint=excluded.request_fingerprint,
            updated_at=excluded.updated_at,
            next_wake_at=excluded.next_wake_at,
            stalled=excluded.stalled,
            doc_json=excluded.doc_json
        """,
        (
            doc["run_id"],
            doc.get("state") or "UNKNOWN",
            doc.get("track"),
            doc.get("request_fingerprint"),
            created,
            now,
            doc.get("next_wake_at"),
            int(bool(doc.get("stalled", False))),
            blob,
        ),
    )

    if trigger is not None and from_state != doc.get("state"):
        conn.execute(
            """INSERT INTO run_events (at, run_id, from_state, to_state, trigger, detail_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, doc["run_id"], from_state, doc["state"], trigger,
             None if detail is None else json.dumps(detail, ensure_ascii=False)),
        )


def get_run(run_id, *, research_dir=None):
    """Return the parsed run dict for *run_id*, or ``None`` if absent."""
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return None
        row = conn.execute("SELECT doc_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row["doc_json"])


def list_runs(*, state=None, states=None, limit=None, research_dir=None):
    """Query run documents ordered newest first by ``updated_at``."""
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return []
        clauses = []
        params = []
        if state:
            clauses.append("state = ?")
            params.append(state)
        elif states:
            placeholders = ",".join("?" for _ in states)
            clauses.append(f"state IN ({placeholders})")
            params.extend(states)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(int(limit))
        rows = conn.execute(
            f"SELECT doc_json FROM runs{where} ORDER BY updated_at DESC, run_id ASC{limit_sql}",
            params,
        ).fetchall()
        return [json.loads(r["doc_json"]) for r in rows]


def find_active_by_fingerprint(fingerprint, active_states, *, research_dir=None):
    """Return the active run matching *fingerprint*, or ``None``."""
    if not fingerprint or not active_states:
        return None
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return None
        placeholders = ",".join("?" for _ in active_states)
        row = conn.execute(
            f"SELECT doc_json FROM runs WHERE request_fingerprint = ? AND state IN ({placeholders}) LIMIT 1",
            [fingerprint, *sorted(active_states)],
        ).fetchone()
        return json.loads(row["doc_json"]) if row is not None else None


def run_counts(*, research_dir=None):
    """Count runs by state without loading their JSON documents."""
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return {}
        return dict(conn.execute("SELECT state, COUNT(*) FROM runs GROUP BY state"))


def _is_selection_receipt(path):
    prefix, marker, suffix = path.stem.rpartition("-selection")
    return bool(prefix and marker and (not suffix or (len(suffix) == 7 and suffix.startswith("-"))))


def legacy_run_documents(runs_dir):
    """Read validated legacy run files without creating or updating workspace state."""
    for path in sorted(Path(runs_dir).glob("*.json")):
        if path.name.startswith(".") or _is_selection_receipt(path):
            continue
        doc, err = cache_store.load_json(path)
        if not err and isinstance(doc, dict) and doc.get("run_id") == path.stem and doc.get("state"):
            yield path, doc


def import_legacy_run_files(runs_dir, *, research_dir=None, dry_run=False):
    """Import legacy run documents once, preserving original files for recovery."""
    with connect(research_dir, create=False) as conn:
        known = set() if conn is None else {row[0] for row in conn.execute("SELECT run_id FROM runs")}
    candidates = {p.stem for p in Path(runs_dir).glob("*.json") if not p.name.startswith(".") and not _is_selection_receipt(p)}
    if not candidates - known:
        return {"imported": 0, "skipped": 0}
    docs = {doc["run_id"]: doc for _, doc in legacy_run_documents(runs_dir) if doc["run_id"] not in known}
    skipped = len(candidates - known - docs.keys())
    if dry_run or not docs:
        return {"imported": len(docs), "skipped": skipped}
    imported = 0
    with immediate_tx(research_dir) as conn:
        for rid, doc in docs.items():
            if conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (rid,)).fetchone() is not None:
                continue
            _write_run(conn, doc)
            imported += 1
    return {"imported": imported, "skipped": skipped}


# ---------------------------------------------------------------------------
# Simulation Slot Journals & Quota Snapshots
# ---------------------------------------------------------------------------
def load_slot_journal(fingerprint, *, research_dir=None, legacy_path=None):
    """Load the slot journal for *fingerprint* from ``workspace.db`` or legacy JSON."""
    with connect(research_dir, create=False) as conn:
        if conn is not None:
            row = conn.execute(
                "SELECT journal_json FROM slot_journals WHERE account_fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if row is not None:
                try:
                    return json.loads(row["journal_json"])
                except ValueError as exc:
                    raise cache_store.CacheError("simulation slot journal in SQLite is invalid JSON") from exc
    legacy, error = (cache_store.load_json(legacy_path) if legacy_path is not None else (None, "missing"))
    if not error:
        return legacy
    if error != "missing":
        raise cache_store.CacheError(f"simulation slot journal cannot be read safely: {error}")
    return {"version": 1, "account_fingerprint": fingerprint, "reservations": {}}


def has_slot_reservations(*, research_dir=None):
    """Check whether SQLite contains any active slot reservations."""
    with connect(research_dir, create=False) as conn:
        return conn is not None and conn.execute(
            """
            SELECT 1 FROM slot_journals
            WHERE EXISTS (SELECT 1 FROM json_each(journal_json, '$.reservations'))
            LIMIT 1
            """
        ).fetchone() is not None


def save_slot_journal(fingerprint, journal, *, research_dir=None, import_only=False):
    """Persist *journal* atomically, preserving any current row during legacy import."""
    quota = journal.get("quota")
    with immediate_tx(research_dir, durable=import_only) as conn:
        if import_only and conn.execute(
            "SELECT 1 FROM slot_journals WHERE account_fingerprint = ?", (fingerprint,),
        ).fetchone() is not None:
            return
        conn.execute(
            """
            INSERT INTO slot_journals (account_fingerprint, journal_json, quota_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(account_fingerprint) DO UPDATE SET
                journal_json=excluded.journal_json,
                quota_json=excluded.quota_json,
                updated_at=excluded.updated_at
            """,
            (
                fingerprint,
                json.dumps(journal, ensure_ascii=False),
                None if quota is None else json.dumps(quota, ensure_ascii=False),
                time.time(),
            ),
        )


def latest_quota_snapshot(*, research_dir=None, legacy_dir=None):
    """Return the freshest recorded quota snapshot from ``slot_journals`` or legacy files."""
    latest = None
    migrated_accounts = set()
    with connect(research_dir, create=False) as conn:
        if conn is not None:
            for row in conn.execute("SELECT account_fingerprint, quota_json FROM slot_journals"):
                migrated_accounts.add(row["account_fingerprint"])
                if row["quota_json"] is None:
                    continue
                try:
                    quota = json.loads(row["quota_json"])
                except ValueError:
                    continue
                if isinstance(quota, dict) and type(quota.get("observed_at")) in (int, float):
                    if latest is None or quota["observed_at"] > latest["observed_at"]:
                        latest = quota
    slot_dir = Path(legacy_dir) if legacy_dir else (Path(research_dir or research_paths.ensure_dir()) / "simulation-slots")
    for path in sorted(slot_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        raw, err = cache_store.load_json(path)
        if err or not isinstance(raw, dict):
            continue
        fingerprint = raw.get("account_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint in migrated_accounts:
            continue
        quota = raw.get("quota")
        if isinstance(quota, dict) and type(quota.get("observed_at")) in (int, float):
            if latest is None or quota["observed_at"] > latest["observed_at"]:
                latest = quota
    return latest


# ---------------------------------------------------------------------------
# Shelf Projection
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# BRAIN Labs Diagnostic Tickets
# ---------------------------------------------------------------------------
def _row_to_ticket(row):
    return {
        "ticket_id": row["ticket_id"],
        "status": row["status"],
        "trigger_rule": row["trigger_rule"],
        "region": row["region"],
        "universe": row["universe"],
        "delay": int(row["delay"]),
        "dataset": row["dataset"],
        "fields": json.loads(row["fields_json"]),
        "questions": json.loads(row["questions_json"]),
        "expected_metrics": json.loads(row["expected_metrics_json"]),
        "branches": json.loads(row["branches_json"]),
        "notebook_snippet": row["notebook_snippet"],
        "source_run_id": row["source_run_id"],
        "resolution": None if row["resolution_json"] is None else json.loads(row["resolution_json"]),
        "selected_branch": row["selected_branch"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "resolved_at": row["resolved_at"],
    }


def insert_labs_ticket(ticket, *, run=None, from_state=None, research_dir=None):
    """Create a ticket and optionally move its associated run into the Labs gate atomically."""
    with immediate_tx(research_dir) as conn:
        conn.execute(
            """
            INSERT INTO labs_tickets (
                ticket_id, status, trigger_rule, region, universe, delay, dataset,
                fields_json, questions_json, expected_metrics_json, branches_json,
                notebook_snippet, source_run_id, resolution_json, selected_branch,
                created_at, updated_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket["ticket_id"],
                ticket["status"],
                ticket["trigger_rule"],
                ticket["region"].upper(),
                ticket["universe"].upper(),
                int(ticket["delay"]),
                ticket.get("dataset"),
                json.dumps(ticket["fields"], ensure_ascii=False),
                json.dumps(ticket["questions"], ensure_ascii=False),
                json.dumps(ticket["expected_metrics"], ensure_ascii=False),
                json.dumps(ticket["branches"], ensure_ascii=False),
                ticket["notebook_snippet"],
                ticket.get("source_run_id"),
                None if ticket.get("resolution") is None else json.dumps(ticket["resolution"], ensure_ascii=False),
                ticket.get("selected_branch"),
                ticket["created_at"],
                ticket["updated_at"],
                ticket.get("resolved_at"),
            ),
        )
        if run is not None:
            _write_run(conn, run, from_state=from_state, trigger="act:request-labs",
                       detail={"ticket_id": ticket["ticket_id"]})
    return ticket


def get_labs_ticket(ticket_id, *, research_dir=None):
    """Fetch a single BRAIN Labs ticket by *ticket_id*, or ``None`` if absent."""
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return None
        row = conn.execute("SELECT * FROM labs_tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        return None if row is None else _row_to_ticket(row)


def close_labs_ticket(ticket_id, updates, *, resumable_states, research_dir=None):
    """Close an OPEN ticket and restore its linked Labs-gate run in one transaction."""
    with immediate_tx(research_dir) as conn:
        row = conn.execute("SELECT * FROM labs_tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        if row is None or row["status"] != "OPEN":
            return (None if row is None else _row_to_ticket(row)), None, False
        merged = {**_row_to_ticket(row), **updates}
        transition = None
        run_id = merged["source_run_id"]
        if run_id:
            run = conn.execute("SELECT doc_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            doc = json.loads(run["doc_json"]) if run is not None else {}
            if doc.get("state") == "GATE_LABS" and doc.get("labs_ticket_id") == ticket_id:
                previous = doc.get("pre_labs_state")
                if previous not in resumable_states:
                    raise ValueError(f"unrecognized Labs resume state {previous!r}")
                info = {key: merged[key] for key in
                        ("ticket_id", "status", "resolution", "selected_branch", "resolved_at")}
                doc.update(state=previous, updated_at=merged["updated_at"], labs_resolution=info)
                _write_run(conn, doc, from_state="GATE_LABS", trigger=f"labs:{merged['status'].lower()}",
                           detail={"ticket_id": ticket_id})
                transition = {"run_id": run_id, "state": previous}
        conn.execute(
            """
            UPDATE labs_tickets
            SET status = ?, resolution_json = ?, selected_branch = ?, updated_at = ?, resolved_at = ?
            WHERE ticket_id = ?
            """,
            (
                merged["status"],
                None if merged.get("resolution") is None else json.dumps(merged["resolution"], ensure_ascii=False),
                merged.get("selected_branch"),
                merged["updated_at"],
                merged.get("resolved_at"),
                ticket_id,
            ),
        )
        return merged, transition, True


def list_labs_tickets(*, status=None, field=None, limit=50, research_dir=None):
    """List BRAIN Labs tickets ordered newest first."""
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return []
        predicates = []
        params = []
        if status:
            predicates.append("status = ?")
            params.append(status.upper())
        if field:
            predicates.append("EXISTS (SELECT 1 FROM json_each(fields_json) WHERE value = ? COLLATE NOCASE)")
            params.append(field)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        rows = conn.execute(
            f"SELECT * FROM labs_tickets{where} ORDER BY updated_at DESC, ticket_id ASC LIMIT ?",
            (*params, -1 if limit is None else limit),
        )
        return [_row_to_ticket(row) for row in rows]


def summarize_open_labs_tickets(*, limit, research_dir=None):
    """Return the open ticket count and a bounded projection without reading ticket bodies."""
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return 0, []
        conn.execute("BEGIN")
        count = conn.execute("SELECT COUNT(*) FROM labs_tickets WHERE status = 'OPEN'").fetchone()[0]
        rows = conn.execute(
            """SELECT ticket_id, trigger_rule, region, universe, delay, fields_json, source_run_id
               FROM labs_tickets WHERE status = 'OPEN'
               ORDER BY updated_at DESC, ticket_id ASC LIMIT ?""",
            (limit,),
        )
        return count, [{"fields": json.loads(row["fields_json"]), **{
            key: row[key] for key in ("ticket_id", "trigger_rule", "region", "universe", "delay", "source_run_id")
        }} for row in rows]


def find_open_blocking_tickets(region, universe, delay, fields, *, limit=20, research_dir=None):
    """Return a bounded result of matching open tickets in a scope without reading ticket bodies."""
    target_fields = {f.casefold(): f for f in fields if f}
    if not target_fields:
        return []
    with connect(research_dir, create=False) as conn:
        if conn is None:
            return []
        rows = conn.execute(
            """
            SELECT ticket_id, trigger_rule, fields_json
            FROM labs_tickets
            WHERE status = 'OPEN' AND region = ? AND universe = ? AND delay = ?
              AND EXISTS (
                SELECT 1 FROM json_each(fields_json) AS ticket_field
                JOIN json_each(?) AS target_field ON ticket_field.value = target_field.value COLLATE NOCASE
              )
            ORDER BY created_at ASC, ticket_id ASC LIMIT ?
            """,
            (str(region).upper(), str(universe).upper(), int(delay), json.dumps(list(target_fields)), limit),
        )
        return [{
            "ticket_id": row["ticket_id"],
            "trigger_rule": row["trigger_rule"],
            "overlapping_fields": sorted({target_fields[field.casefold()]
                                          for field in json.loads(row["fields_json"])
                                          if field.casefold() in target_fields}),
        } for row in rows]
