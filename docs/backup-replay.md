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
python3 tools/backup_replay.py replay --ids akVznkO2 --run
```

## Profiles

| Profile | Contents | Measured 2026-11-01 |
| --- | --- | --- |
| `ledger` | `alpha-registry.jsonl`, `field-units.json` | 2 files, 4.9 MB → 0.4 MB |
| `state` | everything except `cache/` — results, PnL series, checkpoints, notes | 1855 files, 41.5 MB → 5.4 MB, 6 s |
| `full` | `state` plus the metadata cache, for a machine that has to work offline | +11 MB |

The default is `state`: the cache is fetched again on demand, the PnL series and the
ledger are not.

**A session cookie is never archived, in any profile.** `cache/session.json` is a
live BRAIN session; an archive is a file that gets copied, mailed and restored onto
other machines. Restoring a machine costs one login, so there is nothing to trade.
The manifest records what was left out (`excluded_secrets`) so the omission is
visible rather than remembered.

## What an archive is

`<out>/research-<profile>-<label>-<UTC>.tar.gz` plus a sidecar
`.manifest.json`. The manifest is the part worth reading:

* every member's path, size, mtime and sha256;
* the archive's own size and sha256;
* the ledger's numbers — rows, distinct keys, date range, verdict mix, and the book
  counted **per key**;
* anything deliberately excluded.

Two counts in there are deliberately different. On 2026-11-01 the ledger held
**2134 rows over 1267 keys**, and **117 rows carry an `ACTIVE` status while the book
is 48 keys** — the same alpha filed at several universes. A manifest that reports
117 is a manifest that will be quoted as 117, so both numbers are printed with their
单位 spelled out.

## Verification is inside the write

`backup` re-opens the archive it just wrote and re-hashes every member before it
prints success. An unreadable backup that claims to be fine is worse than no backup,
because it stops you making a second one.

`verify` fails on **three** things, not one:

* a member whose hash or size differs;
* a member in the manifest that is missing from the archive;
* a member in the archive that is not in the manifest — an archive that quietly
  gained a file is as suspect as one that lost it.

`restore` verifies first, then extracts. It will not overwrite an existing file
without `--force`, refuses members whose path escapes the target directory (that is
a damaged archive, not a path to normalise), and `--dry-run` lists what it would
write.

## Point-in-time replay (`asof`)

The ledger is append-only, so any past view is recoverable: take the rows up to a
timestamp and keep the newest record per key. **Filter first, deduplicate second** —
that order is the whole feature, because a key rewritten after the cutoff has to
keep the value it had before it.

Measured, which is also a sanity check on the ledger itself:

| As of | Keys | Book |
| --- | --- | --- |
| 2026-10-19 | 245 | 5 |
| 2026-10-22 | 866 | 32 |
| 2026-10-26 | 1241 | 48 |
| 2026-10-31 | 1267 | 48 |

`asof --json` writes the reconstructed rows, so a report can be checked against the
state it was written from instead of against today's ledger.

## Drift check (`replay`)

Re-simulates selected rows through `guarded_sim` — the only supported simulation
entry point — and diffs `sharpe`, `fitness`, `turnover`, `returns`, `drawdown`
against what was recorded, field by field, with a tolerance.

* It is a **dry run unless `--run`** is given, because it spends simulations.
* It **writes a report, not the ledger**. A replay is evidence, not a new
  observation; appending it would double every key.
* A metric missing on either side is reported as *missing*, never as a difference of
  zero.
* A different alpha id for the same expression and settings is reported separately:
  that is an identity change, not a numeric drift.
* One failure does not stop the batch.

First run, 2026-11-01: `akVznkO2` and `qMEWm7L2`, **2 replayed, 2 identical, 0
drifted, 0 failed** — 3m37s for two simulations.

## Limitations

* No incremental or deduplicated archives: every backup is a full copy. At 5 MB that
  is not worth the complexity, and a simpler format is one you can open with `tar`.
* No encryption. The exclusion list is the security boundary, so treat an archive as
  a copy of the research directory minus credentials.
* `restore` extracts a whole archive; there is no single-file restore.
* `replay` re-simulates one expression at a time, so a large sample is slow; it is a
  spot check, not a nightly job.