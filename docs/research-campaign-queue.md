# Research campaign delivery

Use the research directory for the account's catalog cache, ledger and quota journal. Use `--campaign-dir` for a nested campaign that owns its own `backlog` and `backlog-results`.

## Enumerate one region and universe

```shell
python research_cli.py catalog inventory --region KOR --universe TOP600 --stream --save /path/to/kor-fields.json
```

The inventory enumerates every dataset and field for the selected region, universe, instrument type and delay. Each row has the dataset ID, field type, coverage and date coverage. Coverage can be zero or null when the value is unavailable. Inspect `discovery.complete` and each dataset's scan state before drawing conclusions from missing rows. A complete cached catalog is identified as cached. For a fresh catalog fetch, use `--refresh-meta`. Catalog reads consume no simulations.

## Publish a QUICK pilot or expand a supported mechanism

```shell
python research_cli.py campaign publish pilot.json --campaign-dir /path/to/campaign --multi-size 10
python research_cli.py campaign publish expansion.json --campaign-dir /path/to/campaign --multi-size 10 \
  --phase expansion --pilot-batch /path/to/campaign/backlog-results/pilot-batch.json
```

Publish each regional scope separately. The pilot has 16 arms and the expansion has at least 16. Every arm needs an idea, each normalized expression skeleton can occur at most eight times, and at least three actual signal-field datasets must have current account-and-scope evidence. Before the file is visible to a worker, the command runs `probe --check --strict-gate` against the same bytes and verifies dataset admission. The response includes the exact backlog destination, top-level listing or observed claim, request fingerprint and dataset evidence.

For expansion, set an explicit `mechanism` on each arm. Each expanded mechanism needs at least one completed QUICK arm from the saved 16-arm pilot on the same track and scope with train Sharpe ≥1.0, positive train Fitness and train turnover ≤35%. Test-window metrics do not enter this decision. Use FULL separately to confirm any shortlisted candidate and obtain platform checks.

The execution runner manages platform capacity and Multi permissions. Pass `--multi-size 10` on the runner as well when using the generic campaign command. Publication never starts a simulation.

## Diagnose delivery and read quota

```shell
python research_cli.py campaign diagnose --campaign-dir /path/to/campaign
```

The offline report distinguishes fresh top-level specs, saved checkpoints, claims, failed specs and the approved reserve. The runner saves a failure record under `backlog/failed/.reasons` before moving an unsuccessful spec. The diagnostic can still surface prior failures without a saved reason and expired native-dispatch deadlines. The host refill reports an empty approved reserve explicitly.

Quota comes from simulation POST response headers. The account's existing journal lock grants exclusive access to each write. Each writer merges against the current saved value inside the lock. Within an established reset window an out-of-order larger balance cannot overwrite a smaller one. A fresh window requires a new balance and reset boundary. Use completed child tasks to calculate throughput, and mechanism decisions to measure research output.
