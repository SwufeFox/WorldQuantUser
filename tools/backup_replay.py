#!/usr/bin/env python3
"""Snapshot the research directory, verify it, and replay what it recorded.

Two different fears, two different mechanisms.

**Losing the data.** The ledger is append-only JSONL and every result file is a
plain document, but they sit in one directory with a cache that gets rewritten and
a pile of ``.before-*`` copies that only exist because nobody made a real snapshot.
`backup` writes one archive plus a manifest that describes it: every file's size and
sha256, and the ledger's own numbers (rows, keys, date range, verdict mix) so a
manifest can be read a year later without extracting anything. `verify` re-reads the
archive and re-hashes every member; it fails on a missing member *and* on an extra
one, because an archive that quietly gained a file is as suspect as one that lost
it.

**Losing the meaning.** Numbers quoted in a report were true at a moment, and the
ledger keeps moving. `asof` replays the append history offline: filter the rows to
one timestamp, keep the newest record per key, and you have the exact state a
document was written against. No session, no simulation.

**Losing the ground.** If the platform's numbers drifted, every conclusion inherits
the drift. `replay` re-simulates a selection through ``guarded_sim`` and diffs the
metrics against what was recorded, field by field, with a tolerance. It is opt-in
(``--run``) because it spends simulations, and it writes a report rather than
touching the ledger: a replay is evidence, not a new observation.

Usage:
    python3 tools/backup_replay.py backup                    # state profile
    python3 tools/backup_replay.py backup --profile ledger   # just the ledger
    python3 tools/backup_replay.py verify backups/x.tar.gz
    python3 tools/backup_replay.py restore backups/x.tar.gz --to /tmp/restore
    python3 tools/backup_replay.py asof --at 2026-10-25T00:00:00Z
    python3 tools/backup_replay.py replay --status ACTIVE --limit 3          # dry run
    python3 tools/backup_replay.py replay --ids akVznkO2,qMEWm7L2 --run      # costs sims
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import hashlib
import json
import os
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alpha_registry  # noqa: E402
import cache_store  # noqa: E402
import research_paths  # noqa: E402

#: What goes into an archive.
#:
#: ``ledger``  the append-only record and nothing else: small, cheap, worth running
#:             after every batch.
#: ``state``   everything that cannot be regenerated -- the ledger, the field-unit
#:             table, result documents, recorded PnL series, checkpoints. The
#:             metadata cache is excluded because it is fetched again on demand.
#: ``full``    the cache too, for a machine that has to work offline.
PROFILES = {
    "ledger": ("alpha-registry.jsonl", "field-units.json"),
    "state": None,  # everything except the exclusions below
    "full": None,
}

#: Directory names never worth archiving, whatever the profile.
EXCLUDED_DIRS = {"__pycache__", "backups", ".git", ".pytest_cache"}
#: File suffixes that are build noise or ephemeral locks.
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".lock", ".tmp", ".part")

#: ``state`` additionally drops the cache; ``full`` keeps it.
CACHE_DIR = "cache"

#: Never archived, in any profile. ``cache/session.json`` is a BRAIN session
#: cookie; a backup is a file that gets copied, mailed and restored onto other
#: machines, which is the last place a live credential should travel. Restoring a
#: machine costs one login, so there is nothing to trade here.
SECRET_FILES = frozenset({"session.json", ".env", "cookies.json", "credentials.json",
                          "token.json", ".netrc"})


# ------------------------------------------------------------------ file walking


def collect(root, profile="state", extra_exclude=()):
    """Every file the profile covers, as sorted relative paths.

    Sorted so two backups of unchanged data produce the same order, which makes a
    diff of two manifests readable instead of a wall of shuffled lines.
    """
    if profile not in PROFILES:
        raise ValueError("unknown profile %r (choose from %s)"
                         % (profile, ", ".join(sorted(PROFILES))))
    wanted = PROFILES[profile]
    excluded = set(extra_exclude)
    out = []
    for directory, dirnames, filenames in os.walk(root):
        relative_dir = os.path.relpath(directory, root)
        dirnames[:] = sorted(name for name in dirnames
                             if name not in EXCLUDED_DIRS
                             and name not in excluded
                             and not (profile == "state" and relative_dir == "."
                                      and name == CACHE_DIR))
        for name in filenames:
            if name.endswith(EXCLUDED_SUFFIXES) or name in SECRET_FILES:
                continue
            relative = os.path.normpath(os.path.join(relative_dir, name)).replace(os.sep, "/")
            if relative.startswith(".."):
                continue
            if wanted is not None and relative not in wanted:
                continue
            out.append(relative)
    return sorted(out)


def secret_files(root):
    """Relative paths of credential-shaped files that were deliberately left out."""
    found = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        for name in filenames:
            if name in SECRET_FILES:
                found.append(os.path.normpath(
                    os.path.join(os.path.relpath(directory, root), name)).replace(os.sep, "/"))
    return sorted(found)


def fingerprint(path):
    """``(bytes, sha256)`` for one file, read in chunks so a big file stays cheap."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def ledger_stats(path):
    """The numbers a manifest should carry about the ledger itself.

    Read line by line rather than through ``load_entries``: this runs during a
    backup, a damaged line should be counted and reported, not raised, and the point
    is to describe the file that was archived rather than the view the tooling
    prefers.
    """
    stats = {"rows": 0, "damaged": 0, "keys": 0, "dated": 0, "undated": 0,
             "oldest": None, "newest": None, "by_verdict": {}, "by_status": {},
             "book": 0, "keys_by_verdict": {}}
    keys = set()
    last_status, last_verdict = {}, {}
    verdicts, statuses = collections.Counter(), collections.Counter()
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError:
        return None
    with handle:
        for line in handle:
            if not line.strip():
                continue
            stats["rows"] += 1
            try:
                row = json.loads(line)
            except ValueError:
                stats["damaged"] += 1
                continue
            if not isinstance(row, dict):
                stats["damaged"] += 1
                continue
            key = row.get("key") or row.get("alpha_id") or ""
            keys.add(key)
            # Row level and key level are different questions and used to be
            # conflated here: 117 rows say ACTIVE because the same alpha was filed
            # at several universes, while the book is 48. A manifest that reports
            # 117 is a manifest that will be quoted as 117.
            last_status[key] = (row.get("status") or "").upper()
            last_verdict[key] = row.get("verdict") or "-"
            stamp = row.get("recorded_at")
            if stamp:
                stats["dated"] += 1
                stats["oldest"] = stamp if stats["oldest"] is None else min(stats["oldest"], stamp)
                stats["newest"] = stamp if stats["newest"] is None else max(stats["newest"], stamp)
            else:
                stats["undated"] += 1
            verdicts[row.get("verdict") or "-"] += 1
            statuses[row.get("status") or "-"] += 1
    stats["keys"] = len(keys)
    stats["by_verdict"] = dict(verdicts.most_common())
    stats["by_status"] = dict(statuses.most_common())
    stats["book"] = sum(1 for status in last_status.values() if status == "ACTIVE")
    stats["keys_by_verdict"] = dict(collections.Counter(last_verdict.values()).most_common())
    return stats


# ---------------------------------------------------------------------- manifest


def build_manifest(root, files, profile, label=None):
    """Describe an archive: every member's size and hash, plus what the ledger says.

    The ledger block is what makes a manifest readable a year later without
    extracting anything -- rows, distinct keys, date range, verdict mix, and the book
    counted per key. A file list alone tells you the bytes survived, not what they
    meant.
    """
    manifest = {
        "version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "profile": profile,
        "label": label,
        "source": os.path.abspath(root),
        "files": [],
        "totals": {"files": 0, "bytes": 0},
    }
    for relative in files:
        path = os.path.join(root, relative)
        try:
            size, digest = fingerprint(path)
            mtime = os.path.getmtime(path)
        except OSError as error:
            manifest["files"].append({"path": relative, "error": str(error)})
            continue
        manifest["files"].append({"path": relative, "bytes": size,
                                  "sha256": digest, "mtime": mtime})
        manifest["totals"]["files"] += 1
        manifest["totals"]["bytes"] += size
    ledger = ledger_stats(os.path.join(root, "alpha-registry.jsonl"))
    if ledger is not None:
        manifest["ledger"] = ledger
    excluded = secret_files(root)
    if excluded:
        manifest["excluded_secrets"] = excluded
    return manifest


def _archive_name(profile, label, when):
    stem = "research-%s-%s" % (profile, label or "snapshot")
    return "%s-%s.tar.gz" % (stem, when.strftime("%Y%m%dT%H%M%SZ"))


def write_archive(root, files, manifest, target):
    """Write ``target`` (tar.gz) in a deterministic order, then hash and verify it.

    The archive is verified before it is reported as written: an unreadable backup
    that claims success is worse than no backup, because it stops you making a
    second one.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        for relative in files:
            path = os.path.join(root, relative)
            if not os.path.isfile(path):
                continue
            archive.add(path, arcname=relative, recursive=False)
    size, digest = fingerprint(target)
    manifest["archive"] = {"path": os.path.abspath(target), "bytes": size,
                           "sha256": digest}
    problems = verify_archive(target, manifest)
    if problems:
        raise RuntimeError("archive failed verification: %s" % "; ".join(problems[:3]))
    # The sidecar is written here rather than by the caller so that an archive can
    # never exist without the manifest that describes it -- restore and verify both
    # look for it, and a backup that needs a second step to become verifiable is a
    # backup that will be verified never.
    cache_store.write_json_atomic(manifest_path(target), manifest, indent=1)
    return manifest["archive"]


def verify_archive(archive, manifest=None):
    """Re-hash an archive and its members; return a list of problems (empty is good).

    ``manifest`` may be passed to skip the sidecar lookup, which is what
    :func:`write_archive` does before the sidecar exists.
    """
    problems = []
    if manifest is None:
        manifest = load_manifest(archive)
        if manifest is None:
            return ["no manifest next to %s" % os.path.basename(archive)]
    try:
        size, digest = fingerprint(archive)
    except OSError as error:
        return ["cannot read archive: %s" % error]
    expected = (manifest.get("archive") or {}).get("sha256")
    if expected and expected != digest:
        problems.append("archive sha256 %s does not match manifest %s"
                        % (digest[:12], expected[:12]))
    declared = {item["path"]: item for item in manifest.get("files") or []
                if "sha256" in item}
    seen = set()
    try:
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                if not member.isfile():
                    continue
                seen.add(member.name)
                record = declared.get(member.name)
                if record is None:
                    problems.append("member %s is not in the manifest" % member.name)
                    continue
                if member.size != record["bytes"]:
                    problems.append("member %s is %d bytes, manifest says %d"
                                    % (member.name, member.size, record["bytes"]))
                    continue
                extracted = handle.extractfile(member)
                chunk_digest = hashlib.sha256()
                while True:
                    chunk = extracted.read(1 << 20)
                    if not chunk:
                        break
                    chunk_digest.update(chunk)
                if chunk_digest.hexdigest() != record["sha256"]:
                    problems.append("member %s hashes differently" % member.name)
    except (OSError, tarfile.TarError) as error:
        problems.append("cannot open archive: %s" % error)
        return problems
    missing = sorted(set(declared) - seen)
    for name in missing[:10]:
        problems.append("member %s is in the manifest but not the archive" % name)
    if len(missing) > 10:
        problems.append("... and %d more missing" % (len(missing) - 10))
    return problems


def prune(directory, profile, label, keep, protect=None):
    """Delete all but the newest ``keep`` archives of one profile+label.

    Sorted by filename, which is the UTC stamp, so "newest" needs no stat call and
    stays correct when files are copied around. The archive just written is never a
    candidate, and a sidecar is deleted with its archive -- a manifest without an
    archive would make `verify` fail on a file nobody can see.
    """
    if not keep or keep <= 0:
        return []
    pattern = os.path.join(directory, "research-%s-%s-*.tar.gz"
                           % (profile, label or "snapshot"))
    found = sorted(glob.glob(pattern))
    expired = found[:-keep] if len(found) > keep else []
    removed = []
    for path in expired:
        if protect and os.path.abspath(path) == os.path.abspath(protect):
            continue
        for target in (path, manifest_path(path)):
            try:
                os.remove(target)
                removed.append(target)
            except OSError:
                continue
    return removed


def manifest_path(archive):
    return archive + ".manifest.json"


def load_manifest(archive):
    for candidate in (manifest_path(archive),):
        try:
            with open(candidate) as handle:
                return json.load(handle)
        except (OSError, ValueError):
            continue
    return None


# ----------------------------------------------------------------------- restore


def restore(archive, target, force=False, dry_run=False, manifest=None):
    """Extract an archive after verifying it.

    Refuses to overwrite an existing file unless ``force``, and lists what it would
    replace first when ``dry_run``. Extraction happens into ``target`` only: a member
    whose name escapes the directory is refused rather than normalised, because a
    path that tries to leave is a damaged archive, not a path to fix.
    """
    manifest = manifest or load_manifest(archive)
    problems = verify_archive(archive, manifest)
    if problems:
        return {"ok": False, "problems": problems, "written": []}
    target = os.path.abspath(target)
    written, skipped, refused = [], [], []
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            destination = os.path.abspath(os.path.join(target, member.name))
            if not destination.startswith(target + os.sep):
                refused.append(member.name)
                continue
            if os.path.exists(destination) and not force:
                skipped.append(member.name)
                continue
            written.append(member.name)
            if dry_run:
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            source = handle.extractfile(member)
            with open(destination, "wb") as out:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
    if written and not dry_run:
        with open(os.path.join(target, ".restored.json"), "w") as handle:
            json.dump({"archive": os.path.abspath(archive),
                       "restored_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "files": written}, handle, indent=1)
    return {"ok": True, "problems": [], "written": written,
            "skipped": skipped, "refused": refused, "dry_run": dry_run}


# -------------------------------------------------------------------------- asof


def rows_asof(rows, at=None):
    """The ledger as it stood at ``at``: newest record per key, nothing later.

    Filtering first and deduplicating second is what makes this exact: a key that
    was rewritten after ``at`` keeps its earlier value, which is the whole point of
    a point-in-time view.
    """
    cutoff = _parse_stamp(at) if at else None
    kept = {}
    order = []
    for row in rows:
        if cutoff is not None:
            stamp = row.get("recorded_at")
            if not stamp:
                continue
            if _parse_stamp(stamp) > cutoff:
                continue
        key = row.get("key") or row.get("alpha_id") or alpha_registry.expression_hash(
            row.get("expression") or "")
        if key not in kept:
            order.append(key)
        kept[key] = row
    return [kept[key] for key in order]


def _parse_stamp(text):
    if isinstance(text, dt.datetime):
        return text if text.tzinfo else text.replace(tzinfo=dt.timezone.utc)
    raw = str(text).strip().replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def summarize(rows):
    """Count a point-in-time view the way :func:`ledger_stats` counts the live file.

    ``keys`` is the number of records after deduplication and ``book`` is how many of
    those end ACTIVE. Both are key-level on purpose: the row-level version of the
    same number is roughly twice as large and answers a question nobody asked.
    """
    verdicts = collections.Counter(row.get("verdict") or "-" for row in rows)
    statuses = collections.Counter((row.get("status") or "-").upper() for row in rows)
    dates = [row.get("recorded_at") for row in rows if row.get("recorded_at")]
    return {
        "keys": len(rows),
        "by_verdict": dict(verdicts.most_common()),
        "by_status": dict(statuses.most_common()),
        "oldest": min(dates) if dates else None,
        "newest": max(dates) if dates else None,
        "book": statuses.get("ACTIVE", 0),
    }


# ------------------------------------------------------------------------ replay


#: Metric fields worth diffing. Anything absent on either side is reported as
#: missing rather than as a difference of zero.
REPLAY_FIELDS = ("sharpe", "fitness", "turnover", "returns", "drawdown")


def compare(recorded, fresh, tolerance=1e-9):
    """Field-by-field diff of two metric dicts."""
    recorded = recorded or {}
    fresh = fresh or {}
    out = {"missing": [], "differences": {}, "max_delta": 0.0, "identical": True}
    for field in REPLAY_FIELDS:
        left, right = recorded.get(field), fresh.get(field)
        if left is None or right is None:
            if left is not None or right is not None:
                out["missing"].append(field)
                out["identical"] = False
            continue
        try:
            delta = abs(float(right) - float(left))
        except (TypeError, ValueError):
            out["missing"].append(field)
            out["identical"] = False
            continue
        if delta > tolerance:
            out["differences"][field] = {"recorded": left, "fresh": right,
                                         "delta": delta}
            out["max_delta"] = max(out["max_delta"], delta)
            out["identical"] = False
    return out


def replay(entries, simulate=None, tolerance=1e-9, session=None, labels=None):
    """Re-simulate ``entries`` and diff each against what was recorded.

    ``simulate`` is injected so the diff logic is testable without the platform; it
    defaults to ``guarded_sim.simulate``, which is the only supported way to
    simulate in this repo and does the name checks the platform would otherwise do
    for us -- by failing a batch.
    """
    if simulate is None:
        sys.path.insert(0, HERE)
        import guarded_sim  # noqa: PLC0415 - optional dependency, only for --run
        simulate = guarded_sim.simulate

    results = []
    for entry in entries:
        expression = entry.get("expression") or ""
        settings = entry.get("settings") or {}
        record = {"alpha_id": entry.get("alpha_id"), "key": entry.get("key"),
                  "expression": expression, "settings": settings}
        try:
            produced = simulate([expression], settings, session=session,
                                labels=[entry.get("alpha_id") or "replay"])
        except Exception as error:  # noqa: BLE001 - one failure must not stop the run
            record["error"] = "%s: %s" % (type(error).__name__, error)
            results.append(record)
            continue
        fresh = produced[0] if produced else {}
        record["fresh_alpha_id"] = fresh.get("alpha_id")
        record["status"] = fresh.get("status")
        record["comparison"] = compare(entry.get("metrics"), fresh.get("metrics"),
                                       tolerance)
        record["fresh_metrics"] = fresh.get("metrics")
        if entry.get("alpha_id") and fresh.get("alpha_id") \
                and entry["alpha_id"] != fresh.get("alpha_id"):
            record["identity_changed"] = True
        results.append(record)
    return results


def select(entries, ids=None, status=None, verdict=None, limit=None, sample=None):
    """Pick ledger entries to replay, with an explicit and printed rule.

    ``sample`` is seeded by nothing: a random sample of a research ledger is for
    spot checks, and a reproducible one would be a different promise.
    """
    chosen = list(entries)
    if ids:
        wanted = [item.strip() for item in ids.split(",") if item.strip()]
        chosen = [entry for entry in chosen
                  if entry.get("alpha_id") in wanted or entry.get("key") in wanted]
    if status:
        wanted = {item.strip().upper() for item in status.split(",") if item.strip()}
        chosen = [entry for entry in chosen
                  if (entry.get("status") or "").upper() in wanted]
    if verdict:
        wanted = {item.strip().lower() for item in verdict.split(",") if item.strip()}
        chosen = [entry for entry in chosen
                  if (entry.get("verdict") or "").lower() in wanted]
    if sample:
        import random  # noqa: PLC0415 - only used when asked for
        random.shuffle(chosen)
        chosen = chosen[:sample]
    if limit:
        chosen = chosen[:limit]
    return chosen


# ---------------------------------------------------------------------- reporting


def report_backup(manifest, stream=None):
    """Print what was archived, in the terms the numbers are usually quoted in."""
    stream = sys.stdout if stream is None else stream
    ledger = manifest.get("ledger") or {}
    print("archived %d files, %.1f MB uncompressed"
          % (manifest["totals"]["files"], manifest["totals"]["bytes"] / 1e6),
          file=stream)
    if ledger:
        print("ledger: %d rows / %d keys, %s .. %s"
              % (ledger.get("rows", 0), ledger.get("keys", 0),
                 (ledger.get("oldest") or "-")[:19], (ledger.get("newest") or "-")[:19]),
              file=stream)
        print("verdicts: %s"
              % ", ".join("%s %d" % item for item in ledger.get("by_verdict", {}).items()),
              file=stream)
        print("book: %d ACTIVE keys (%d rows carry an ACTIVE status)"
              % (ledger.get("book", 0), (ledger.get("by_status") or {}).get("ACTIVE", 0)),
              file=stream)
        if ledger.get("damaged"):
            print("WARNING: %d damaged lines were archived as-is" % ledger["damaged"],
                  file=stream)
    if manifest.get("excluded_secrets"):
        print("left out on purpose (credentials): %s"
              % ", ".join(manifest["excluded_secrets"]), file=stream)
    archive = manifest.get("archive") or {}
    if archive:
        print("archive: %s (%.1f MB, sha256 %s)"
              % (archive.get("path"), archive.get("bytes", 0) / 1e6,
                 (archive.get("sha256") or "")[:12]), file=stream)


def report_replay(results, stream=None):
    """Summarise a replay by identical / drifted / failed, and name the worst drift.

    "Drifted" is not a verdict on the alpha -- it is a statement that the platform's
    answer changed, which is the thing worth knowing before trusting any older number.
    """
    stream = sys.stdout if stream is None else stream
    total = len(results)
    errored = [item for item in results if item.get("error")]
    drifted = [item for item in results
               if item.get("comparison") and not item["comparison"]["identical"]]
    identical = total - len(errored) - len(drifted)
    print("replayed %d: %d identical, %d drifted, %d failed"
          % (total, identical, len(drifted), len(errored)), file=stream)
    if drifted:
        worst = max(drifted, key=lambda item: item["comparison"]["max_delta"])
        print("largest drift: %s (max |delta| %.3g on %s)"
              % (worst.get("alpha_id"), worst["comparison"]["max_delta"],
                 ", ".join(sorted(worst["comparison"]["differences"]))), file=stream)
        for item in drifted[:10]:
            fields = ", ".join("%s %+.4f" % (name, body["delta"])
                               for name, body in item["comparison"]["differences"].items())
            print("  %-10s %s" % (item.get("alpha_id"), fields), file=stream)
    for item in errored[:5]:
        print("  FAILED %-10s %s" % (item.get("alpha_id"), item["error"][:120]),
              file=stream)
    changed = [item for item in results if item.get("identity_changed")]
    if changed:
        print("%d expression/settings pairs now return a different alpha id"
              % len(changed), file=stream)
    return {"total": total, "identical": identical, "drifted": len(drifted),
            "failed": len(errored)}


# ---------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    backup = sub.add_parser("backup", help="write a verified archive")
    backup.add_argument("--profile", default="state", choices=sorted(PROFILES))
    backup.add_argument("--out", default=None, help="default: <research>/backups")
    backup.add_argument("--label", default=None, help="name to put in the filename")
    backup.add_argument("--root", default=None)
    backup.add_argument("--exclude", action="append", default=[])
    backup.add_argument("--keep", type=int, default=0,
                        help="keep only the newest N archives of this profile+label")

    verify = sub.add_parser("verify", help="re-hash an archive against its manifest")
    verify.add_argument("archive")

    restore_cmd = sub.add_parser("restore", help="verify, then extract")
    restore_cmd.add_argument("archive")
    restore_cmd.add_argument("--to", required=True)
    restore_cmd.add_argument("--force", action="store_true")
    restore_cmd.add_argument("--dry-run", action="store_true")

    asof = sub.add_parser("asof", help="the ledger as it stood at one timestamp")
    asof.add_argument("--at", default=None, help="ISO timestamp; omit for 'now'")
    asof.add_argument("--json", default=None, help="write the reconstructed rows here")

    replay_cmd = sub.add_parser("replay", help="re-simulate and diff metrics")
    replay_cmd.add_argument("--ids", default=None)
    replay_cmd.add_argument("--status", default=None)
    replay_cmd.add_argument("--verdict", default=None)
    replay_cmd.add_argument("--sample", type=int, default=None)
    replay_cmd.add_argument("--limit", type=int, default=None)
    replay_cmd.add_argument("--tolerance", type=float, default=1e-9)
    replay_cmd.add_argument("--run", action="store_true",
                            help="actually simulate; without it this is a dry run")
    replay_cmd.add_argument("--json", default=None)

    args = parser.parse_args(argv)

    if args.command == "backup":
        root = args.root or research_paths.ensure_dir()
        files = collect(root, args.profile, args.exclude)
        if not files:
            print("nothing to archive under %s" % root, file=sys.stderr)
            return 2
        manifest = build_manifest(root, files, args.profile, args.label)
        out_dir = args.out or os.path.join(root, "backups")
        target = os.path.join(out_dir, _archive_name(args.profile, args.label,
                                                     dt.datetime.now(dt.timezone.utc)))
        write_archive(root, files, manifest, target)
        report_backup(manifest)
        removed = prune(out_dir, args.profile, args.label, args.keep, protect=target)
        if removed:
            print("pruned %d older file(s)" % len(removed))
        return 0

    if args.command == "verify":
        manifest = load_manifest(args.archive)
        problems = verify_archive(args.archive, manifest)
        if problems:
            print("FAILED: %d problem(s)" % len(problems))
            for problem in problems[:20]:
                print("  " + problem)
            return 1
        print("ok: %d files match the manifest, archive sha256 %s"
              % (len((manifest or {}).get("files") or []),
                 ((manifest or {}).get("archive") or {}).get("sha256", "")[:12]))
        return 0

    if args.command == "restore":
        outcome = restore(args.archive, args.to, force=args.force,
                          dry_run=args.dry_run)
        if not outcome["ok"]:
            print("refused: %d problem(s)" % len(outcome["problems"]))
            for problem in outcome["problems"][:10]:
                print("  " + problem)
            return 1
        verb = "would write" if args.dry_run else "wrote"
        print("%s %d files to %s" % (verb, len(outcome["written"]), args.to))
        if outcome["skipped"]:
            print("skipped %d existing files (pass --force to overwrite)"
                  % len(outcome["skipped"]))
        if outcome["refused"]:
            print("refused %d members whose path escapes the target"
                  % len(outcome["refused"]))
        return 0

    if args.command == "asof":
        rows = _raw_rows(args.at)
        state = summarize(rows)
        print("as of %s: %d keys, %d ACTIVE"
              % (args.at or "now (everything)", state["keys"], state["book"]))
        print("  verdicts: %s"
              % ", ".join("%s %d" % item for item in state["by_verdict"].items()))
        print("  window: %s .. %s"
              % ((state["oldest"] or "-")[:19], (state["newest"] or "-")[:19]))
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(rows, handle, ensure_ascii=False, indent=1)
            print("  wrote %s" % args.json)
        return 0

    if args.command == "replay":
        entries = alpha_registry.load_entries()
        chosen = select(entries, ids=args.ids, status=args.status,
                        verdict=args.verdict, limit=args.limit, sample=args.sample)
        if not chosen:
            print("nothing selected", file=sys.stderr)
            return 2
        print("selected %d of %d ledger entries" % (len(chosen), len(entries)))
        for entry in chosen[:10]:
            print("  %-10s %s" % (entry.get("alpha_id"),
                                  (entry.get("expression") or "").replace("\n", " ")[:88]))
        if len(chosen) > 10:
            print("  ... %d more" % (len(chosen) - 10))
        if not args.run:
            print("\ndry run: pass --run to spend %d simulation(s)" % len(chosen))
            return 0
        import machine_lib as ml  # noqa: PLC0415 - only needed when simulating
        session = ml.login()
        results = replay(chosen, tolerance=args.tolerance, session=session)
        summary = report_replay(results)
        if args.json:
            with open(args.json, "w") as handle:
                json.dump({"summary": summary, "results": results}, handle,
                          ensure_ascii=False, indent=1)
            print("wrote %s" % args.json)
        return 0 if summary["drifted"] == 0 and summary["failed"] == 0 else 1

    return 2


def _raw_rows(at=None):
    """Rows straight from the JSONL, deduplicated as of ``at``."""
    path = alpha_registry.default_registry_path()
    rows = []
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError:
        return []
    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows_asof(rows, at)


if __name__ == "__main__":
    raise SystemExit(main())