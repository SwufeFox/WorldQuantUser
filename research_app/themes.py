"""Project observed platform theme records into one catalog.

Themes are an incentive dimension rather than a filter: the platform reports
which themes an alpha matches and the multiplier attached to each, and
``rules_snapshot`` already stores those observations with their source,
timestamp and scope. This module only reads that history and folds it into one
entry per theme id, so research can see which regions a theme has been observed
in and how fresh the evidence is.
"""

from __future__ import annotations

import json
from pathlib import Path

import research_paths

SNAPSHOT_FILENAME = "rules-snapshot.jsonl"
DEFAULT_LIMIT = 50


def snapshot_path(root=None):
    """Return the rules-snapshot file that keeps the observed theme records."""
    base = Path(root) if root else Path(research_paths.research_dir())
    return base / SNAPSHOT_FILENAME


def catalog(root=None):
    """Fold every observed theme record into one entry per theme id."""
    path = snapshot_path(root)
    themes = {}
    scanned = 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict) or record.get("kind") != "themes":
                    continue
                scanned += 1
                observed = record.get("observed_at")
                region = (record.get("scope") or {}).get("region")
                for item in record.get("themes") or []:
                    if not isinstance(item, dict):
                        continue
                    identity = item.get("id")
                    if not isinstance(identity, str) or not identity.strip():
                        continue
                    entry = themes.setdefault(identity, {
                        "id": identity,
                        "name": None,
                        "multiplier": None,
                        "regions": [],
                        "observations": 0,
                        "first_observed_at": None,
                        "last_observed_at": None,
                        "last_result": None,
                    })
                    entry["observations"] += 1
                    if item.get("name") and not entry["name"]:
                        entry["name"] = item["name"]
                    if item.get("multiplier") is not None:
                        entry["multiplier"] = item["multiplier"]
                    if isinstance(region, str) and region and region not in entry["regions"]:
                        entry["regions"].append(region)
                    if observed and (entry["first_observed_at"] is None
                                     or observed < entry["first_observed_at"]):
                        entry["first_observed_at"] = observed
                    if entry["last_observed_at"] is None or (observed or "") >= entry["last_observed_at"]:
                        entry["last_observed_at"] = observed
                        entry["last_result"] = record.get("result")
    rows = sorted(themes.values(), key=lambda row: row["id"])
    rows.sort(key=lambda row: row["last_observed_at"] or "", reverse=True)
    for row in rows:
        row["regions"] = sorted(row["regions"])
    return {"path": str(path), "scanned_records": scanned, "themes": rows}


def cmd_themes_list(args, _sink):
    """List observed themes with their regions, multipliers and freshness."""
    limit = getattr(args, "limit", None)
    data = catalog(getattr(args, "research_dir", None))
    rows = data["themes"]
    if isinstance(limit, int) and limit >= 0:
        rows = rows[:limit]
    return {"path": data["path"], "count": len(data["themes"]),
            "scanned_records": data["scanned_records"], "themes": rows}
