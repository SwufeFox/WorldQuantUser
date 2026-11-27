"""Rebuildable SQLite projection of the append-only Alpha registry.

JSONL owns the evidence. The index stores offsets for specific payloads and
the narrow fields used by selection and aggregation. Its cursor and rows are
committed in one transaction after replaying durable ledger bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from functools import lru_cache

import alpha_registry
import fastexpr_ast
import machine_lib as ml
from check_evidence import platform_check_names, summarize_checks
from mechanism_identity import mechanism_key

_INDEX_VERSION = 4
_ANCHOR_BYTES = 4096
_FINISHED = alpha_registry.FINISHED_STATUSES | {"ACTIVE", "SUBMITTED", "UNSUBMITTED"}
_PYTHON_SETTINGS = ml.regular_wire_settings({**ml.DEFAULT_SETTINGS, "language": "PYTHON"})

_SUMMARY_TYPES = {
    "alpha_id": "TEXT",
    "role": "TEXT",
    "status": "TEXT",
    "region": "TEXT",
    "universe": "TEXT",
    "delay": "INTEGER",
    "simulation_mode": "TEXT",
    "neutralization": "TEXT",
    "decay": "INTEGER",
    "track": "TEXT",
    "family": "TEXT",
    "mechanism": "TEXT",
    "idea": "TEXT",
    "batch": "TEXT",
    "verdict": "TEXT",
    "recorded_at": "TEXT",
    "first_recorded_at": "TEXT",
    "first_measurement_observed_at": "TEXT",
    "measurement_observed_at": "TEXT",
    "first_checks_observed_at": "TEXT",
    "checks_observed_at": "TEXT",
    "checks_state": "TEXT",
    "evidence_metadata": "TEXT",
    "train_start": "TEXT",
    "test_start": "TEXT",
    "has_train_split": "INTEGER",
    "train_sharpe": "REAL",
    "train_fitness": "REAL",
    "train_turnover": "REAL",
    "train_margin": "REAL",
    "train_long_count": "REAL",
    "train_short_count": "REAL",
    "check_count": "INTEGER",
    "failed_checks": "TEXT",
    "unresolved_checks": "TEXT",
}
_UPSERT_COLUMNS = ("key", "first_offset", "line_offset", "line_size", "lookup_key",
                   "evidence_key", "request_hash", "request_settings",
                   "structural_block", "structural_skeleton", *_SUMMARY_TYPES)
_UPSERT = (
    "INSERT INTO registry_entries (" + ", ".join(_UPSERT_COLUMNS) + ") "
    "VALUES (" + ", ".join("?" for _ in _UPSERT_COLUMNS) + ") "
    "ON CONFLICT(key) DO UPDATE SET " +
    ", ".join(f"{name} = excluded.{name}" for name in _UPSERT_COLUMNS[2:])
)
_LATEST_LOOKUP = """
    NOT EXISTS (
        SELECT 1 FROM registry_entries AS newer
        WHERE newer.lookup_key = row.lookup_key AND newer.first_offset > row.first_offset
    )
"""
_LOOKUP_ORDER = """
    ORDER BY (SELECT MIN(peer.first_offset) FROM registry_entries AS peer
              WHERE peer.lookup_key = row.lookup_key)
"""
_STRUCTURAL = "row.structural_block = ? AND row.structural_skeleton = ? AND " + _LATEST_LOOKUP
_PAYLOAD = "row.key, row.line_offset, row.line_size, row.role, row.evidence_metadata"


class _UnindexableLedger(ValueError):
    """The legacy reader is needed to preserve dictionary semantics."""


def index_path(path):
    """Return the rebuildable SQLite index path alongside a JSONL ledger file."""
    return os.fspath(path) + ".index.sqlite3"


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _text(value):
    return value if isinstance(value, str) else None


def _number(value):
    return value if _finite(value) and -(2**63) <= value < 2**63 else None


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _status(value):
    return str(value or "").upper() or None


def summary_fields(entry):
    """Project recorded metadata and explicit train statistics into SQL columns."""
    settings = _mapping(entry.get("settings"))
    split = _mapping(entry.get("out_of_sample"))
    train = _mapping(split.get("train"))
    test = _mapping(split.get("test"))
    checks = alpha_registry._normalize_checks(entry.get("checks"))
    prior = entry.get("last_check_snapshot") or {}
    required = platform_check_names(prior.get("checks")) if entry.get("checks_state") == "partial" else ()
    check_summary = summarize_checks(checks, required=required)
    legacy_failures = entry.get("failing_checks")
    failures = set(alpha_registry._failing_checks(checks))
    if isinstance(legacy_failures, list):
        failures.update(name for name in legacy_failures if isinstance(name, str))
    unresolved = sorted(set(check_summary["pending"]) | set(check_summary["errored"]))

    return {
        **{name: _text(entry.get(name)) for name in
           ("alpha_id", "role", "family", "mechanism", "idea", "batch", "verdict", "recorded_at",
            "first_recorded_at", "first_measurement_observed_at", "measurement_observed_at",
            "first_checks_observed_at", "checks_observed_at", "checks_state")},
        "evidence_metadata": json.dumps(alpha_registry.evidence_state(entry), sort_keys=True),
        "status": _status(entry.get("status")),
        "region": _status(settings.get("region")),
        "universe": _status(settings.get("universe")),
        "delay": _number(settings.get("delay")),
        "simulation_mode": _status(settings.get("simulationMode")),
        "neutralization": _status(settings.get("neutralization")),
        "decay": _number(settings.get("decay")),
        "track": _text(entry.get("track")),
        "train_start": _text(train.get("startDate")),
        "test_start": _text(test.get("startDate")),
        "has_train_split": int(isinstance(split.get("train"), dict)),
        "train_sharpe": _number(train.get("sharpe")),
        "train_fitness": _number(train.get("fitness")),
        "train_turnover": _number(train.get("turnover")),
        "train_margin": _number(train.get("margin")),
        "train_long_count": _number(train.get("longCount")),
        "train_short_count": _number(train.get("shortCount")),
        "check_count": len(platform_check_names(checks)),
        "failed_checks": json.dumps(sorted(failures)),
        "unresolved_checks": json.dumps(unresolved),
    }


def settings_signature(settings):
    """Canonical compact JSON signature of resolved simulation settings."""
    settings = settings or {}
    if ml.is_python_language(settings.get("language")):
        settings = ml.regular_wire_settings(settings)
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def known_request_identity(entry):
    """Compute the resolved request identity only for a finished, fully known row."""
    expression = entry.get("expression")
    prior = entry.get("settings")
    if not isinstance(expression, str) or not expression or not isinstance(prior, dict):
        return None
    if _status(entry.get("status")) not in _FINISHED:
        return None
    metrics = _mapping(entry.get("metrics") or _mapping(entry.get("out_of_sample")).get("train"))
    if not any(_finite(metrics.get(key)) for key in ("sharpe", "fitness", "returns", "turnover")):
        return None
    defaults = _PYTHON_SETTINGS if ml.is_python_language(ml.alpha_language(entry)) else ml.DEFAULT_SETTINGS
    if any(prior.get(key) is None for key in defaults):
        return None
    try:
        resolved = ml.resolve_settings(prior)
    except (ValueError, TypeError):
        return None
    return (alpha_registry.expression_hash(expression, resolved.get("language")),
            settings_signature(resolved))


def _source_stamp(handle):
    if handle is None:
        return None
    stat = os.fstat(handle.fileno())
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _prefix_anchor(handle, size):
    if not size:
        return ""
    handle.seek(0)
    start = handle.read(min(size, _ANCHOR_BYTES))
    handle.seek(max(len(start), size - _ANCHOR_BYTES))
    end = handle.read(size - handle.tell())
    return hashlib.sha256(start + end).hexdigest()


def _request_lookup(identity):
    expression_hash, signature = identity
    return expression_hash, hashlib.sha256(signature.encode("utf-8")).digest()


@lru_cache(maxsize=4096)
def _expression_structure(expression, language):
    try:
        fields = tuple(fastexpr_ast.field_names(fastexpr_ast.parse_program(expression)))
    except (fastexpr_ast.ParseError, RecursionError):
        fields = ()
    return mechanism_key(expression, language) or (None, None), fields


class RegistryIndex:
    """Locked SQLite projection supporting incremental replay and indexed queries."""

    def __init__(self, path, connection, stack):
        self.path = path
        self.connection = connection
        self._stack = stack
        self._source = None

    def _create_schema(self):
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version == _INDEX_VERSION:
            return
        summary = ",\n".join(f"{name} {kind}" for name, kind in _SUMMARY_TYPES.items())
        self.connection.executescript(f"""
            BEGIN;
            DROP TABLE IF EXISTS registry_fields;
            DROP TABLE IF EXISTS registry_entries;
            DROP TABLE IF EXISTS registry_source;
            CREATE TABLE registry_source (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                stamp TEXT NOT NULL,
                anchor TEXT NOT NULL,
                partial INTEGER NOT NULL
            );
            CREATE TABLE registry_entries (
                key TEXT PRIMARY KEY,
                first_offset INTEGER NOT NULL,
                line_offset INTEGER NOT NULL,
                line_size INTEGER NOT NULL,
                lookup_key TEXT,
                evidence_key TEXT,
                request_hash TEXT,
                request_settings BLOB,
                structural_block TEXT,
                structural_skeleton TEXT,
                {summary}
            );
            CREATE TABLE registry_fields (
                key TEXT NOT NULL,
                field TEXT NOT NULL,
                PRIMARY KEY (key, field)
            ) WITHOUT ROWID;
            CREATE INDEX registry_field_name ON registry_fields (field, key);
            CREATE INDEX registry_entry_order ON registry_entries (first_offset);
            CREATE INDEX registry_lookup ON registry_entries (lookup_key, first_offset);
            CREATE INDEX registry_diagnostic_alpha ON registry_entries (alpha_id)
                WHERE role = 'diagnostic' AND alpha_id IS NOT NULL;
            CREATE INDEX registry_request ON registry_entries (request_hash, request_settings)
                WHERE request_hash IS NOT NULL;
            CREATE INDEX registry_structure ON registry_entries (structural_block, structural_skeleton)
                WHERE structural_block IS NOT NULL;
            CREATE INDEX registry_research_scope ON registry_entries
                (region, universe, delay, simulation_mode, train_sharpe DESC);
            PRAGMA user_version = {_INDEX_VERSION};
            COMMIT;
        """)

    def _open_source(self):
        # A pinned handle keeps existing readers on the same inode after replacement.
        try:
            return open(self.path, "rb")
        except FileNotFoundError:
            return None

    def refresh(self):
        """Reconcile a stable byte prefix while the registry's append lock is held."""
        if self._source is not None:
            self._source.close()
        source = self._source = self._open_source()
        if source is not None:
            self._stack.callback(source.close)
        stamp = _source_stamp(source)
        checkpoint = self.connection.execute(
            "SELECT stamp, anchor, partial FROM registry_source WHERE id = 1"
        ).fetchone()
        previous_stamp = json.loads(checkpoint[0]) if checkpoint else None
        if checkpoint and stamp == previous_stamp:
            return
        offset = 0
        can_replay = (stamp is not None and previous_stamp is not None
                      and stamp[:2] == previous_stamp[:2] and stamp[2] > previous_stamp[2]
                      and not checkpoint[2] and _prefix_anchor(source, previous_stamp[2]) == checkpoint[1])
        with self.connection:
            if can_replay:
                offset = previous_stamp[2]
            else:
                self.connection.execute("DELETE FROM registry_fields")
                self.connection.execute("DELETE FROM registry_entries")
            partial = False
            if source is not None:
                source.seek(offset)
                remaining = stamp[2] - offset
                while remaining:
                    line = source.readline(remaining)
                    if not line:
                        raise OSError("registry changed while rebuilding its index")
                    size = len(line)
                    key, entry = alpha_registry._parse_line(line)
                    if entry is not None:
                        self._project(key, entry, offset, size)
                    offset += size
                    remaining -= size
                    partial = not line.endswith(b"\n")
                anchor = _prefix_anchor(source, offset)
            else:
                anchor = ""
            if _source_stamp(source) != stamp:
                raise OSError("registry changed while rebuilding its index")
            self.connection.execute(
                "INSERT OR REPLACE INTO registry_source VALUES (1, ?, ?, ?)",
                (json.dumps(stamp), anchor, partial),
            )

    def _project(self, key, entry, offset, size):
        lookup = entry.get("alpha_id") or entry.get("key")
        evidence = entry.get("key") or entry.get("alpha_id")
        if (entry["key"] != key or not isinstance(key, str)
                or (lookup is not None and not isinstance(lookup, str))):
            raise _UnindexableLedger("nonstandard registry keys require the legacy reader")
        previous = self.connection.execute(
            "SELECT role, evidence_metadata FROM registry_entries WHERE key = ?", (key,)
        ).fetchone()
        entry = alpha_registry.fold_evidence(entry, json.loads(previous[1]) if previous else None)
        entry["role"] = alpha_registry.evidence_role({"role": previous[0]} if previous else None, entry)
        identity = known_request_identity(entry)
        request = _request_lookup(identity) if identity else (None, None)
        expression = entry.get("expression")
        structure, fields = (_expression_structure(expression, ml.alpha_language(entry))
                             if isinstance(expression, str) else ((None, None), ()))
        summary = summary_fields(entry)
        self.connection.execute(_UPSERT, (
            key, offset, offset, size, lookup or None, evidence or None, *request, *structure,
            *(summary[name] for name in _SUMMARY_TYPES),
        ))
        if previous:
            self.connection.execute("DELETE FROM registry_fields WHERE key = ?", (key,))
        self.connection.executemany(
            "INSERT INTO registry_fields (key, field) VALUES (?, ?)",
            ((key, field) for field in fields),
        )

    def begin_snapshot(self):
        """Pin rows before releasing the external append lock."""
        self.connection.execute("BEGIN")
        self.connection.execute("SELECT stamp FROM registry_source").fetchone()

    def refresh_after_append(self):
        try:
            self.refresh()
        except (sqlite3.Error, OSError, ValueError):
            # A durable JSONL write succeeds on its own. Reopen and replay on a later read.
            pass

    def _read_entry(self, row):
        key, offset, size, role, provenance = row
        self._source.seek(offset)
        read_key, entry = alpha_registry._parse_line(self._source.read(size))
        if read_key != key or entry is None:
            raise OSError("registry no longer matches the indexed evidence")
        entry = alpha_registry.fold_evidence(entry, json.loads(provenance))
        entry["role"] = role
        return entry

    def entry(self, key):
        row = self.connection.execute(
            f"SELECT {_PAYLOAD} FROM registry_entries AS row WHERE row.key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return self._read_entry(row)

    def exact_history(self, identity, *, limit=3):
        """Get the complete exact-evidence key list and decode only its preview."""
        rows = self.connection.execute(
            f"SELECT row.evidence_key, {_PAYLOAD} FROM registry_entries AS row "
            f"WHERE row.request_hash = ? AND row.request_settings = ? "
            f"AND {_LATEST_LOOKUP} {_LOOKUP_ORDER}",
            _request_lookup(identity),
        )
        keys, preview = [], []
        for key, *row in rows:
            keys.append(key)
            if len(preview) < limit:
                preview.append(self._read_entry(row))
        return keys, preview

    def mechanism_history(self, key, *, exclude_keys=(), limit=3):
        """Count structural history and decode only the requested preview rows."""
        if key is None:
            return 0, [], []
        base = f" FROM registry_entries AS row WHERE {_STRUCTURAL}"
        others = base + """
            AND NOT EXISTS (SELECT 1 FROM json_each(?) WHERE value IS row.evidence_key)
        """
        args = (*key, json.dumps(list(exclude_keys)))
        count = self.connection.execute("SELECT COUNT(*)" + others, args).fetchone()[0]
        active = self.connection.execute(
            "SELECT row.evidence_key" + base + " AND row.status = 'ACTIVE'" + _LOOKUP_ORDER,
            key,
        )
        preview = self.connection.execute(
            f"SELECT {_PAYLOAD}" + others + _LOOKUP_ORDER + " LIMIT ?", (*args, max(0, limit))
        )
        return count, [row[0] for row in active], [self._read_entry(row) for row in preview]

    def dataset_turnover(self, field_datasets):
        """Aggregate known training turnover once per ledger key and dataset."""
        if not field_datasets:
            return {}
        rows = self.connection.execute("""
            WITH matches AS (
                SELECT DISTINCT fields.key, catalog.value AS dataset
                FROM json_each(?) AS catalog
                CROSS JOIN registry_fields AS fields ON fields.field = catalog.key
            ), evidence AS (
                SELECT matches.dataset, rows.train_turnover AS turnover
                FROM matches JOIN registry_entries AS rows USING (key)
            ), totals AS (
                SELECT dataset, count(*) AS total, count(CASE WHEN turnover >= 0 THEN 1 END) AS n
                FROM evidence GROUP BY dataset
            ), ranked AS (
                SELECT dataset, turnover,
                       row_number() OVER (PARTITION BY dataset ORDER BY turnover) AS position,
                       count(*) OVER (PARTITION BY dataset) AS size
                FROM evidence WHERE turnover >= 0
            ), medians AS (
                SELECT dataset, avg(turnover) AS median
                FROM ranked
                WHERE position IN ((size + 1) / 2, (size + 2) / 2)
                GROUP BY dataset
            )
            SELECT totals.dataset, totals.n, medians.median, totals.total - totals.n
            FROM totals LEFT JOIN medians USING (dataset)
        """, (json.dumps(field_datasets, separators=(",", ":")),))
        return {dataset: {"n": n, "median": median, "missing_train": missing}
                for dataset, n, median, missing in rows}


@contextmanager
def open_index(path):
    """Open the local projection with its external append lock already held."""
    with ExitStack() as stack:
        try:
            connection = sqlite3.connect(index_path(path), timeout=3)
            stack.callback(connection.close)
            connection.execute("PRAGMA journal_mode = WAL")
            index = RegistryIndex(path, connection, stack)
            index._create_schema()
            index.refresh()
        except (sqlite3.Error, OSError, ValueError):
            index = None
        yield index


class LedgerRows(MutableMapping):
    """Load previous payloads only when an append or a caller needs them."""

    def __init__(self, index):
        self._index = index
        self._cached = {}
        self._misses = set()
        self._excluded = set()
        self._added = {}

    def __getitem__(self, key):
        if key not in self._cached:
            if key in self._excluded or key in self._misses or not isinstance(key, str):
                raise KeyError(key)
            try:
                self._cached[key] = self._index.entry(key)
            except KeyError:
                self._misses.add(key)
                raise
        return self._cached[key]

    def __setitem__(self, key, entry):
        if key not in self:
            self._added[key] = None
        self._cached[key] = entry

    def __delitem__(self, key):
        self[key]
        del self._cached[key]
        if key in self._added:
            del self._added[key]
        else:
            self._excluded.add(key)

    def __iter__(self):
        stored = self._index.connection.execute("SELECT key FROM registry_entries ORDER BY first_offset")
        for (key,) in stored:
            if key not in self._excluded:
                yield key
        yield from self._added

    def __len__(self):
        count = self._index.connection.execute("SELECT COUNT(*) FROM registry_entries").fetchone()[0]
        return count - len(self._excluded) + len(self._added)
