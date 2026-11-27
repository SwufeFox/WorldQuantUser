# Backing up the research directory, and replaying what it recorded

Three fears, three mechanisms. They are separate on purpose: a backup that also
promises reproducibility is a backup nobody can check.

| Fear | Verb | Costs a session? |
| --- | --- | --- |
| Losing the files | `backup` / `verify` / `restore` | no |
| Losing the meaning ("this number was true on 2026-10-25") | `asof` | no |
| Losing the ground (the platform's numbers drifted) | `replay` | yes, one simulation per row |

```
python3 tools/backup_replay.py backup                     # state profile
python3 tools/backup_replay.py backup --profile ledger    # the ledger alone
python3 tools/backup_replay.py verify backups/x.tar.gz
python3 tools/backup_replay.py restore backups/x.tar.gz --to /tmp/restore --dry-run
python3 tools/backup_replay.py asof --at 2026-10-25T00:00:00Z
python3 tools/backup_replay.py replay --ids ALPHA_ID --run
```

## Profiles

| Profile | Contents |
| --- | --- |
| `ledger` | `alpha-registry.jsonl`, `field-units.json` |
| `state` | Research files under the selected root, excluding its `cache/` and the common exclusions |
| `full` | `state` plus the research cache, subject to the same common exclusions |

The default is `state`: the cache is fetched again on demand, the PnL series and the
ledger are not.

The `state` and `full` profiles use SQLite's backup API to capture committed
transactions from `workspace.db` and `pnl/meta.sqlite`, including transactions
still in WAL. SQLite WAL, shared-memory, and rollback journal files are excluded.
The PnL store holds its publication lock while the backup stages the manifest,
referenced Parquet files, and legacy JSON. A missing committed file aborts the
backup. The ledger is staged under its append lock. Each store is captured
consistently, at its own point in time. Pause workspace writers if you need a
single cutoff across the database, PnL, and ledger. The output directory needs
temporary space for the staged files as well as the compressed archive.

All profiles exclude known credential files including `cache/session.json`,
`.env`, and `daemon/endpoint.json`. The manifest lists detected exclusions under
`excluded_secrets`. Notes, results, and source files may still contain sensitive
content.

An archive covers the selected root. Standalone `simulate` defaults to
`.brain_checkpoints/` in the invocation directory. To keep those checkpoints in
scope, pass `--checkpoint` with a path under the research root.

## What an archive is

`<out>/research-<profile>-<label>-<UTC>.tar.gz` plus a sidecar
`.manifest.json`. The manifest is the part worth reading:

* every member's path, size, mtime and sha256;
* the archive's own size and sha256;
* the ledger's row count, distinct keys, date range, verdict mix, and latest
  `ACTIVE` count per key;
* anything deliberately excluded.

Member paths in the manifest and archive use `/` on every platform. Windows
backups therefore verify against the same names that tar stores, and nested
results restore with the same layout on Windows and POSIX.

Rows and distinct keys are deliberately different counts. An alpha can be filed
under several settings or recorded repeatedly as its status changes. The manifest
therefore labels both units explicitly and counts the book per key, so a row count
cannot be mistaken for the number of distinct alphas.

## Verification is inside the write

`backup` re-opens the archive it just wrote and re-hashes every member before it
prints success. An unreadable backup that claims to be fine is worse than no backup,
because it stops you making a second one.

`verify` checks the archive hash and compares its members with the manifest.

* a member whose hash or size differs;
* a member in the manifest that is missing from the archive;
* a member in the archive that is not in the manifest — an archive that quietly
  gained a file is as suspect as one that lost it.

`restore` verifies first, then extracts. Restore archives containing a workspace
SQLite database or PnL files into a new research directory to preserve their
consistency. The command rejects existing workspace databases, SQLite sidecars,
PnL directories, or daemon endpoints in the destination for these archives.
For ordinary files, `--force` allows overwriting and `--dry-run` lists what would
be written. Paths that escape the destination are refused.

## Point-in-time replay (`asof`)

Point-in-time views use the dated history retained in the ledger. Apply the
cutoff before keeping the newest record per key. A revision after the cutoff
leaves the earlier value visible.

An explicit cutoff omits undated rows. Reconstruction uses the keys saved in the
ledger, including historical expression hash formats.

`asof --json` writes the reconstructed rows, so a report can be checked against the
state it was written from instead of against today's ledger.

## Drift check (`replay`)

Re-simulates selected rows through `guarded_sim` with stored source and
settings, then compares `sharpe`, `fitness`, `turnover`, `returns`, and
`drawdown` field by field. Python rows need a stored language marker and currently
verified account capability. The platform runs with today's data and rules.

* It is a **dry run unless `--run`** is given, because it spends simulations.
* It prints a report. `--json PATH` saves the report separately from the ledger.
* A metric present on only one side is reported as missing. A metric absent on
  both sides is omitted from the comparison.
* A different alpha id for the same expression and settings is reported separately:
  that is an identity change, not a numeric drift.
* One failure does not stop the batch.

Replay reports distinguish identical results, numeric drift, missing metrics,
identity changes and failed simulations. Keep those reports in the research
directory so they remain tied to the selected records and observation time.

## Limitations

* No incremental or deduplicated archives: every backup is a full copy in a format
  that can be opened with `tar`.
* Archives are unencrypted. Treat them as copies of private research data.
* Verification expects regular files. File symlinks selected under the root can
  cause verification to fail because tar stores them as links. `--exclude NAME`
  can omit a directory by name, including the files inside it.
* `restore` extracts a whole archive; there is no single-file restore.
* `replay` re-simulates one expression at a time, so a large sample is slow; it is a
  spot check, not a nightly job.
