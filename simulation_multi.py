"""Durable parent and child state for native BRAIN multi-simulation requests."""

import hashlib
import json
import re
import time
import uuid

import machine_lib as ml
import simulation_slots


MAX_CHILDREN = 10
CONSULTANT_PARENT_LIMIT = 8
PARENT_POLL_FALLBACK_SECONDS = 30
PARENT_TIMING_FIELDS = (
    "multi_parent_accepted_at", "multi_last_pending_observed_at", "multi_parent_terminal_observed_at",
)
_PARENT_PENDING_STATUSES = frozenset({"PENDING", "QUEUED", "WAITING", "SIMULATING", "RUNNING"})
STATE_KEYS = (
    "multi_request_id", "multi_member", "multi_size", "multi_payload_sha",
    "multi_parent_url", "multi_parent_terminal", "multi_parent_status",
    "multi_children", "multi_children_complete", "multi_child_id", "multi_next_parent_poll_at",
    "multi_child_retry", "multi_terminal_evidence", "multi_failure_attribution",
)
_MAX_DIAGNOSTIC_LENGTH = 512
_PRIVATE_DIAGNOSTIC = re.compile(
    r"https?://|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|\b(?:bearer|basic)\s+\S+|"
    r"\b(?:authorization|cookie|credential|password|passwd|secret|token|api[_ -]?key|"
    r"(?:access|refresh|auth)[_ -]?token|session[_ -]?(?:id|key))"
    r"""["']?\s*[:=]""",
    re.IGNORECASE,
)


def _terminal_observation(payload, previous=None):
    """Keep bounded diagnostic text without copying response bodies or credentials."""
    status = payload["status"]
    observed = {"status": status}
    for key in ("message", "detail", "error"):
        value = payload.get(key)
        if isinstance(value, dict):
            value = value.get("message")
        if not isinstance(value, str):
            continue
        message = " ".join(value.split())
        if message and message.isprintable() and not _PRIVATE_DIAGNOSTIC.search(message):
            observed["message"] = message[:_MAX_DIAGNOSTIC_LENGTH]
            return observed
    if previous and previous.get("status") == status and previous.get("message"):
        observed["message"] = previous["message"]
    return observed


def payload_digest(payload):
    """Return a deterministic SHA-256 digest of a single simulation payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _settings_digest(payload):
    """Identify the settings a native parent would execute for this member."""
    settings = payload.get("settings") if isinstance(payload, dict) else None
    canonical = json.dumps(settings or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def request_groups(indices, payload_for, size):
    """Preserve order, keep repeats apart, and never mix settings in one parent.

    BRAIN applies one settings object per parent. A batch that mixed decays or
    universes silently ran every child with the parent's settings, so split on a
    settings change as well as on a repeated request.
    """
    batch = []
    seen = set()
    settings = None
    for index in indices:
        payload = payload_for(index)
        identity = payload_digest(payload)
        digest = _settings_digest(payload)
        if batch and (len(batch) == size or identity in seen or digest != settings):
            yield tuple(batch)
            batch, seen, settings = [], set(), None
        if not batch:
            settings = digest
        batch.append(index)
        seen.add(identity)
    if batch:
        yield tuple(batch)


class MultiRuns:
    """Manage a parent reservation and the one-to-many mapping in the arm records."""

    def __init__(self, session, records, payload_for, account, limit, deadline, emit, submission_event,
                 *, original_payload_for=None, poll_once=False):
        self.session = session
        self.records = records
        self.payload_for = payload_for
        self.original_payload_for = original_payload_for
        self._wire_payloads = {}
        self.account = account
        self.limit = limit
        self.deadline = deadline if deadline is not None else float("inf")
        self.poll_once = poll_once
        self.emit = emit
        self.submission_event = submission_event

    def _emit(self, kind, indices):
        rows = [{**self.records[index], "index": index} for index in indices]
        self.emit(kind, indices[0], {"records": rows})

    def _saved_wire_payload(self, index):
        saved_digest = self.records[index].get("multi_payload_sha")
        payload = self.payload_for(index)
        if payload_digest(payload) == saved_digest:
            return payload
        # Pre-formatting checkpoints identify their original wire syntax by hash.
        if self.original_payload_for is not None:
            original = self.original_payload_for(index)
            if payload_digest(original) == saved_digest:
                return original
        return None

    def restore(self):
        """Validate complete membership and recover parent URLs from the slot journal."""
        groups = {}
        for index, row in enumerate(self.records):
            if row.get("multi_request_id"):
                groups.setdefault(row["multi_request_id"], []).append(index)

        for group in groups.values():
            size = self.records[group[0]].get("multi_size")
            ordinals = [self.records[index].get("multi_member") for index in group]
            if (type(size) is not int or not 2 <= size <= MAX_CHILDREN
                    or any(type(member) is not int for member in ordinals)
                    or sorted(ordinals) != list(range(size))
                    or any(self.records[index].get("multi_size") != size for index in group)):
                raise ValueError("native multi-simulation checkpoint membership does not match its requests")
            payloads = {index: self._saved_wire_payload(index) for index in group}
            if None in payloads.values():
                raise ValueError("native multi-simulation checkpoint membership does not match its requests")
            self._wire_payloads.update(payloads)
            group.sort(key=lambda index: self.records[index]["multi_member"])
            indices = tuple(group)
            saved_url = self.records[indices[0]].get("multi_parent_url")
            parent = self._parent(indices)
            simulation_slots.reconcile_record(parent, self.account)
            self._save_reservation(indices, parent)
            if parent.get("submission_attempted") is False:
                self._clear(indices, None)
                continue
            if parent.get("progress_url"):
                self._save_parent_url(indices, parent["progress_url"])
                if saved_url != parent["progress_url"]:
                    self._emit("multi_submitted", indices)
            yield indices

    def _parent(self, indices):
        anchor = self.records[indices[0]]
        shadow = {
            "status": "COMPLETE" if anchor.get("multi_parent_terminal") else "UNKNOWN_SUBMISSION",
            "submission_attempted": True,
            "slot_reservation": anchor.get("slot_reservation"),
            "slot_reservation_state": anchor.get("slot_reservation_state"),
        }
        if anchor.get("multi_parent_url"):
            shadow["progress_url"] = anchor["multi_parent_url"]
            if not anchor.get("multi_parent_terminal"):
                shadow["status"] = "PENDING"
        return shadow

    def _save_reservation(self, indices, parent):
        anchor = self.records[indices[0]]
        anchor.update({key: parent.get(key) for key in ("slot_reservation", "slot_reservation_state")})

    def _save_parent_url(self, indices, url):
        for index in indices:
            row = self.records[index]
            row["multi_parent_url"] = url
            if row.get("status") == "UNKNOWN_SUBMISSION":
                row.update(status="PENDING", error=None)

    def _clear(self, indices, error, status="QUEUED"):
        for index in indices:
            row = self.records[index]
            row.update(dict.fromkeys(STATE_KEYS))
            for key in PARENT_TIMING_FIELDS:
                row.pop(key, None)
            row.update(status=status, submission_attempted=status != "QUEUED", error=error)
        self._emit("multi_pending", indices)

    def submit(self, indices):
        """Make one parent POST after journaling every arm as a single checkpoint event."""
        from requests.exceptions import HTTPError, RequestException

        payloads = [self.payload_for(index) for index in indices]
        request_id = uuid.uuid4().hex

        def checkpoint(reservation):
            for member, (index, payload) in enumerate(zip(indices, payloads)):
                self._wire_payloads[index] = payload
                row = self.records[index]
                row.update(dict.fromkeys(STATE_KEYS))
                row.update(status="UNKNOWN_SUBMISSION", submission_attempted=True, error=None,
                           multi_request_id=request_id, multi_member=member, multi_size=len(indices),
                           multi_payload_sha=payload_digest(payload))
            self._save_reservation(indices, {"slot_reservation": reservation,
                                             "slot_reservation_state": "submitting"})
            self._emit("multi_submitting", indices)

        quick = all(payload.get("settings", {}).get("simulationMode") == "QUICK" for payload in payloads)
        identity, slots = simulation_slots.prepare_submission(self.account, self.limit, checkpoint, quick_native=quick)
        if identity is None and slots["available"] == 0:
            for index in indices:
                self.records[index]["error"] = "shared simulation slots are occupied; reconcile or resume existing attempts"
            self._emit("multi_pending", indices)
            return "contended"

        parent = self._parent(indices)
        try:
            url = ml.submit_multi_simulation(
                self.session, payloads, deadline=self.deadline,
                on_event=lambda kind, payload: self.submission_event(kind, indices[0], payload),
            )
        except (ml.AuthError, ml.PermissionDenied) as exc:
            simulation_slots.reconcile_record(parent, self.account, outcome="refused")
            self._save_reservation(indices, parent)
            self._clear(indices, str(exc))
            raise
        except ml.RateLimited as exc:
            simulation_slots.reconcile_record(parent, self.account, outcome="refused")
            self._save_reservation(indices, parent)
            self._clear(indices, "%s: %s" % (exc.code, exc))
            return "blocked"
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 408:
                simulation_slots.reconcile_record(parent, self.account, outcome="refused")
                self._save_reservation(indices, parent)
                self._clear(indices, "multi-simulation submission rejected (HTTP %s)" % status, status="ERROR")
                return "skipped"
            self._set_unresolved(indices, "multi-simulation submission outcome needs reconciliation (HTTP %s)" % status)
            return "blocked"
        except (ml.BrainError, RequestException) as exc:
            self._set_unresolved(indices, "multi-simulation submission outcome needs reconciliation (%s)" % type(exc).__name__)
            return "blocked"

        accepted_at = time.time()
        simulation_slots.reconcile_record(parent, self.account, outcome="accepted", progress_url=url)
        self._save_reservation(indices, parent)
        self._save_parent_url(indices, url)
        self.records[indices[0]]["multi_parent_accepted_at"] = accepted_at
        self._emit("multi_submitted", indices)
        return "accepted"

    def pending_on_platform(self, indices):
        anchor = self.records[indices[0]]
        return not anchor.get("multi_parent_terminal")

    def needs_parent_read(self, indices):
        anchor = self.records[indices[0]]
        unresolved = any(self._unresolved(self.records[index]) for index in indices)
        return bool(anchor.get("multi_parent_url") and unresolved
                    and (self.pending_on_platform(indices)
                         or not anchor.get("multi_children_complete")))

    def needs_children(self, indices):
        anchor = self.records[indices[0]]
        return bool(anchor.get("multi_parent_terminal") and anchor.get("multi_children")
                    and any(self._unresolved(self.records[index]) for index in indices))

    @staticmethod
    def _unresolved(row):
        return not (row.get("alpha_id") or row.get("status") in simulation_slots.TERMINAL_STATUSES)

    def _set_unresolved(self, indices, error):
        for index in indices:
            row = self.records[index]
            if self._unresolved(row):
                row["error"] = error
        self._emit("multi_pending", indices)

    def poll_parent(self, indices):
        anchor = self.records[indices[0]]
        parent_id = ml.simulation_reference_id(anchor["multi_parent_url"])
        wait = max(0, (anchor.get("multi_next_parent_poll_at") or 0) - time.time())
        if wait:
            if self.poll_once or wait >= self.deadline - time.monotonic():
                return
            ml.sleep(wait)
        while True:
            try:
                payload, retry_after = ml.read_simulation_snapshot(
                    self.session, parent_id, deadline=self.deadline)
            except ml.BrainError as exc:
                if self.poll_once and isinstance(exc, ml.RateLimited) and exc.retry_after:
                    anchor["multi_next_parent_poll_at"] = time.time() + exc.retry_after
                self._set_unresolved(indices, "multi-simulation parent read failed (%s)" % exc.code)
                return
            if payload.get("id") not in (None, parent_id):
                self._set_unresolved(indices, "multi-simulation parent identity mismatch")
                return
            observed_at = time.time()
            status = payload.get("status")
            pending = not anchor.get("multi_parent_terminal") and (
                retry_after or (isinstance(status, str) and status in _PARENT_PENDING_STATUSES))
            if pending:
                anchor["multi_last_pending_observed_at"] = observed_at
            if not retry_after:
                terminal = isinstance(status, str) and status in simulation_slots.TERMINAL_STATUSES
                anchor["multi_next_parent_poll_at"] = (
                    None if terminal else observed_at + PARENT_POLL_FALLBACK_SECONDS)
                break
            if self.poll_once or retry_after >= self.deadline - time.monotonic():
                anchor["multi_next_parent_poll_at"] = observed_at + retry_after
                self._set_unresolved(indices, "multi-simulation parent is pending")
                return
            if pending:
                self._emit("multi_pending", indices)
            self.emit("polling", indices[0], {"retry_after": retry_after})
            ml.sleep(retry_after)

        if not isinstance(status, str) or status not in simulation_slots.TERMINAL_STATUSES:
            self._set_unresolved(indices, "multi-simulation parent is pending")
            return

        evidence = anchor.get("multi_terminal_evidence") or {}
        anchor["multi_terminal_evidence"] = {
            **evidence, "parent": _terminal_observation(payload, evidence.get("parent")),
        }
        if not anchor.get("multi_parent_terminal"):
            anchor.setdefault("multi_parent_terminal_observed_at", observed_at)
        children = payload.get("children")
        if (isinstance(children, list) and 0 < len(children) <= len(indices)
                and all(ml.valid_simulation_id(child) and child != parent_id for child in children)
                and len(set(children)) == len(children)):
            observed = list(dict.fromkeys([*(anchor.get("multi_children") or []), *children]))
            if len(observed) <= len(indices):
                anchor["multi_children"] = observed
                if len(children) == len(indices):
                    anchor["multi_children_complete"] = True
        for index in indices:
            row = self.records[index]
            row.update(multi_parent_terminal=True, multi_parent_status=status)
            if self._unresolved(row):
                row["error"] = (None if anchor.get("multi_children") else
                                "multi-simulation parent did not expose a valid child list")
        # Terminal evidence must be durable before dropping the reservation.
        self._emit("multi_pending", indices)
        parent = self._parent(indices)
        simulation_slots.reconcile_record(parent, self.account, outcome="terminal")
        self._save_reservation(indices, parent)
        self._emit("multi_pending", indices)

    # The child snapshot reports the executed settings, not the submitted
    # request: testPeriod and simulationMode never come back. Their absence is
    # unknown rather than different, so match on the identifying keys and on
    # every key the snapshot does report. Requiring the full request made every
    # real child fail to map ("did not expose a matching child", 2026-11-19).
    IDENTIFYING_SETTINGS = ("instrumentType", "region", "universe", "delay",
                            "neutralization", "truncation", "language")

    def _matches(self, index, payload):
        requested = self._wire_payloads[index]
        if payload.get("type") not in (None, requested["type"]):
            return False
        expression, _, _ = ml.regular_expression(payload)
        observed = payload.get("settings")
        if expression != requested["regular"] or not isinstance(observed, dict):
            return False
        if any(key not in observed for key in self.IDENTIFYING_SETTINGS):
            return False
        return all(observed[key] == value
                   for key, value in requested["settings"].items()
                   if key in observed)

    def collect_children(self, indices):
        """Match echoed child requests, allowing a partial parent result."""
        anchor = self.records[indices[0]]
        parent_id = ml.simulation_reference_id(anchor["multi_parent_url"])
        remaining = [index for index in indices if self._unresolved(self.records[index])]
        resolved_children = {self.records[index].get("multi_child_id") for index in indices
                             if index not in remaining}
        retries = dict(anchor.get("multi_child_retry") or {})
        evidence = anchor.get("multi_terminal_evidence") or {}
        known_children = anchor["multi_children"][:len(indices)]
        previous = evidence.get("children") or {}
        observations = {child: previous[child] for child in known_children if child in previous}
        matches = {index: [] for index in remaining}
        failures = set()
        for child in known_children:
            if child in resolved_children or retries.get(child, 0) > time.time() or time.monotonic() >= self.deadline:
                continue
            try:
                payload, retry_after = ml.read_simulation_snapshot(
                    self.session, child, deadline=self.deadline)
            except ml.RateLimited:
                break
            except ml.BrainError:
                continue
            if payload.get("id") not in (None, child) or payload.get("parent") not in (None, parent_id):
                continue
            if retry_after:
                retries[child] = time.time() + retry_after
            else:
                retries.pop(child, None)
                status = payload.get("status")
                if isinstance(status, str) and status in simulation_slots.TERMINAL_STATUSES:
                    observations[child] = _terminal_observation(payload, observations.get(child))
                    if (status in simulation_slots.TERMINAL_FAILURE_STATUSES
                            and payload.get("alpha") is None and payload.get("alpha_id") is None):
                        failures.add(child)
            for index in remaining:
                if self.records[index].get("multi_child_id") in (None, child) and self._matches(index, payload):
                    matches[index].append((child, payload, retry_after))
        anchor["multi_child_retry"] = retries
        if observations or "children" in evidence:
            anchor["multi_terminal_evidence"] = {**evidence, "children": observations}

        for index, children in matches.items():
            row = self.records[index]
            if len(children) != 1:
                if not (row.get("multi_child_id") in retries and not children):
                    row["error"] = ("multi-simulation child mapping is ambiguous" if children else
                                    "multi-simulation did not expose a matching child")
                continue
            child, payload, retry_after = children[0]
            row["multi_child_id"] = child
            status = payload.get("status")
            if retry_after or not isinstance(status, str) or status not in simulation_slots.TERMINAL_STATUSES:
                row["error"] = "multi-simulation child is pending"
            elif status in simulation_slots.TERMINAL_FAILURE_STATUSES:
                row.update(status=status, error=observations[child].get("message") or "multi-simulation child failed")
            elif ml.valid_simulation_id(payload.get("alpha")):
                row.update(status="PENDING", simulation_status=status, alpha_id=payload["alpha"],
                           metrics_pending=True, error=None)
            else:
                row["error"] = "multi-simulation child did not expose a valid Alpha ID"
        self._finish_unmatched_failures(indices, matches, failures)
        self._emit("multi_pending", indices)

    def _finish_unmatched_failures(self, indices, matches, failures):
        """Close a fully accounted parent without attributing anonymous children."""
        anchor = self.records[indices[0]]
        if not anchor.get("multi_children_complete"):
            return
        children = set(anchor["multi_children"])
        unmatched = [index for index in indices if self._unresolved(self.records[index])]
        if len(children) != len(indices) or not unmatched or any(matches[index] for index in unmatched):
            return
        resolved = {self.records[index].get("multi_child_id") for index in indices if index not in unmatched}
        if len(resolved) != len(indices) - len(unmatched) or not resolved <= children:
            return
        if not (children - resolved).issubset(failures):
            return
        for index in unmatched:
            self.records[index].update(
                status="ERROR", multi_failure_attribution="unattributed",
                error="all remaining multi-simulation children reported terminal failures without matching requests")
