"""Calculate a bounded, train-only research review from the ledger projection."""

from datetime import date
import json
from pathlib import Path

import alpha_registry
import machine_lib as ml
import research_paths
from registry_index import index_path

from .runtime import CliError, protect_output


_GROUPS = {
    "mechanism": "nullif(trim(mechanism), '')",
    "family": "nullif(trim(family), '')",
    "neutralization": "neutralization",
    "batch": "nullif(trim(batch), '')",
    "scope": """coalesce(region, '?') || '/' || coalesce(universe, '?')
                || '/D' || coalesce(delay, '?') || '/' || coalesce(simulation_mode, '?')""",
}
_STAGES = (
    "active_or_submitted", "diagnostic", "unfinished", "hard_failure", "checks_missing",
    "training_unknown", "weak_training", "positions_unknown", "empty_book",
    "cost_unknown", "cost_exhausted", "high_turnover", "research_screen",
)
_COHORT_TABLE = "temp.ledger_analysis_cohort"
_FINISHED = "'COMPLETE', 'WARNING', 'UNSUBMITTED'"


def _rows(connection, sql, parameters):
    cursor = connection.execute(sql, parameters)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def _cohort(args):
    filters = {
        "region": args.region.upper() if args.region else None,
        "universe": args.universe.upper() if args.universe else None,
        "delay": args.delay,
        "simulation_mode": args.mode,
    }
    where = [f"{column} = :{column}" for column, value in filters.items() if value is not None]
    if args.since:
        try:
            filters["since"] = date.fromisoformat(args.since).isoformat()
        except ValueError as exc:
            raise CliError("INVALID_INPUT", "--since requires a calendar date", "use YYYY-MM-DD") from exc
        where.append("recorded_at >= :since")
    filters = {column: value for column, value in filters.items() if value is not None}
    return (" WHERE " + " AND ".join(where) if where else ""), filters


def _yearly(row, pnl_dir):
    from .pnl_snapshot import yearly_train_batch

    return yearly_train_batch([row], pnl_dir)[0]


def _review(connection, where, parameters, group):
    # Several views share one assessment. Materialize only hot fields once,
    # keeping repeated scans free of payloads, AST keys or date validation.
    connection.execute(f"DROP TABLE IF EXISTS {_COHORT_TABLE}")
    connection.execute(f"""
        CREATE TEMP TABLE {_COHORT_TABLE} AS
        WITH cohort AS (
            SELECT key, alpha_id, role, status, region, universe, delay, simulation_mode,
                   neutralization, decay, family, idea, batch, recorded_at, train_start,
                   test_start, has_train_split, train_sharpe, train_fitness, train_turnover,
                   train_margin, train_long_count, train_short_count, check_count,
                   failed_checks, unresolved_checks, {group} AS group_value,
                   nullif(trim(mechanism), '') AS mechanism_label,
                   json_array_length(failed_checks) AS n_failed,
                   json_array_length(unresolved_checks) AS n_unresolved,
                   10000 * train_margin - :cost_bps AS net_margin_bps
            FROM registry_entries{where}
        ), assessed AS (
            SELECT *, CASE
                WHEN has_train_split IS NOT 1 OR train_sharpe IS NULL
                    OR date(substr(train_start, 1, 10), '+0 days') IS NOT substr(train_start, 1, 10)
                    OR date(substr(test_start, 1, 10), '+0 days') IS NOT substr(test_start, 1, 10)
                    OR train_start IS NULL OR test_start IS NULL
                    OR substr(train_start, 1, 4) = '0000' OR substr(test_start, 1, 4) = '0000'
                    OR substr(train_start, 1, 10) >= substr(test_start, 1, 10) THEN 'training_unknown'
                WHEN train_sharpe < :min_train_sharpe THEN 'weak_training'
                WHEN train_long_count IS NULL OR train_short_count IS NULL THEN 'positions_unknown'
                WHEN train_long_count <= 0 OR train_short_count <= 0 THEN 'empty_book'
                WHEN train_turnover IS NULL OR train_margin IS NULL THEN 'cost_unknown'
                WHEN net_margin_bps <= 0 THEN 'cost_exhausted'
                WHEN train_turnover > :max_turnover THEN 'high_turnover'
                ELSE 'research_screen'
            END AS metrics_stage
            FROM cohort
        )
        SELECT *, CASE
            WHEN status IN ('ACTIVE', 'SUBMITTED') THEN 'active_or_submitted'
            WHEN role = 'diagnostic' OR EXISTS (
                SELECT 1 FROM registry_entries AS diagnostic
                WHERE diagnostic.role = 'diagnostic' AND diagnostic.alpha_id = assessed.alpha_id
            ) THEN 'diagnostic'
            WHEN status IS NULL OR status NOT IN ({_FINISHED})
                OR nullif(alpha_id, '') IS NULL THEN 'unfinished'
            WHEN n_failed > 0 THEN 'hard_failure'
            WHEN check_count = 0 THEN 'checks_missing'
            ELSE metrics_stage
        END AS stage
        FROM assessed
    """, parameters)
    funnel = dict(connection.execute(f"SELECT stage, count(*) FROM {_COHORT_TABLE} GROUP BY stage"))
    counts = _rows(connection, f"""
        SELECT count(*) AS entries, count(DISTINCT alpha_id) AS distinct_alpha_ids,
               count(DISTINCT mechanism_label) AS distinct_mechanism_labels,
               count(*) FILTER (WHERE mechanism_label IS NULL) AS mechanism_unlabeled,
               count(*) FILTER (WHERE simulation_mode IS NULL) AS mode_unknown,
               count(*) FILTER (WHERE metrics_stage = 'training_unknown') AS training_unknown
        FROM {_COHORT_TABLE}
    """, {})[0]
    failures = _rows(connection, f"""
        SELECT checks.value AS check_name, count(*) AS entries,
               count(*) FILTER (WHERE n_failed = 1) AS sole_failure_entries,
               count(*) FILTER (WHERE n_failed = 1 AND metrics_stage = 'research_screen'
                                    AND n_unresolved = 0) AS otherwise_screened_entries
        FROM {_COHORT_TABLE} AS review, json_each(review.failed_checks) AS checks
        WHERE stage = 'hard_failure'
        GROUP BY checks.value
        ORDER BY entries DESC, check_name
        LIMIT :limit
    """, parameters)
    groups = _rows(connection, f"""
        SELECT group_value AS label, count(*) AS entries,
               count(*) FILTER (WHERE stage = 'research_screen') AS screened,
               count(*) FILTER (WHERE stage = 'research_screen' AND n_unresolved = 0) AS screened_with_resolved_stored_checks,
               count(*) FILTER (WHERE stage = 'hard_failure') AS hard_failures,
               count(*) FILTER (WHERE stage IN ('active_or_submitted', 'diagnostic', 'unfinished')) AS outside_screen,
               count(*) FILTER (WHERE metrics_stage <> 'training_unknown') AS entries_with_training,
               avg(train_sharpe) FILTER (WHERE metrics_stage <> 'training_unknown') AS mean_train_sharpe
        FROM {_COHORT_TABLE}
        GROUP BY group_value
        ORDER BY screened DESC, entries DESC, label
        LIMIT :limit
    """, parameters)
    return counts, {stage: funnel.get(stage, 0) for stage in _STAGES}, failures, groups


def _representatives(connection, parameters, *, leads=False):
    eligible = ("stage = 'hard_failure' AND n_failed = 1 AND n_unresolved = 0 "
                "AND metrics_stage = 'research_screen'" if leads else "stage = 'research_screen'")
    ordering = "train_sharpe DESC, net_margin_bps DESC, recorded_at DESC, key"
    rows = _rows(connection, f"""WITH by_alpha AS (
        SELECT *, row_number() OVER (PARTITION BY alpha_id ORDER BY {ordering}) AS alpha_rank
        FROM {_COHORT_TABLE} WHERE {eligible}
    ), ranked AS (
        SELECT *, row_number() OVER (PARTITION BY mechanism_label ORDER BY {ordering}) AS mechanism_rank
        FROM by_alpha WHERE alpha_rank = 1
    )
    SELECT key, alpha_id, region, universe, delay, simulation_mode, neutralization, decay,
           family, mechanism_label AS mechanism, idea, batch, recorded_at, train_start, test_start,
           train_sharpe, train_fitness, train_turnover, train_margin, net_margin_bps,
           train_long_count, train_short_count, failed_checks, unresolved_checks
    FROM ranked
    WHERE mechanism_rank = 1
    ORDER BY {ordering}
    LIMIT :limit
    """, parameters)
    for row in rows:
        row["failed_checks"] = json.loads(row["failed_checks"])
        row["unresolved_checks"] = json.loads(row["unresolved_checks"])
    return rows


def cmd_ledger_analyze(args, _sink):
    """Screen the local ledger, explain losses, and produce bounded evidence packets."""
    if (not 1 <= args.limit <= 100 or args.cost_bps < 0
            or args.max_turnover is not None and args.max_turnover < 0
            or args.delay is not None and args.delay < 0):
        raise CliError("INVALID_INPUT", "invalid analysis limit, cost, turnover or delay",
                       "use a limit of 1 to 100 and non-negative cost, turnover and delay")
    where, filters = _cohort(args)
    registry = Path(args.registry or alpha_registry.default_registry_path()).expanduser().resolve()
    projection = index_path(registry)
    protect_output(args, registry, projection, projection + "-wal", projection + "-shm",
                   str(registry) + ".lock")
    if not registry.exists():
        raise CliError("NOT_FOUND", "the local research ledger does not exist",
                       "select a workspace with recorded backtests or pass --registry")
    parameters = {**filters, "cost_bps": args.cost_bps, "min_train_sharpe": args.min_train_sharpe,
                  "max_turnover": args.max_turnover, "limit": args.limit}
    with alpha_registry.indexed_registry(registry) as index:
        if index is None:
            raise CliError("INVALID_INPUT", "the local ledger projection is unavailable",
                           "check that the ledger and its directory can be read and indexed")
        connection = index.connection
        counts, funnel, failures, groups = _review(connection, where, parameters, _GROUPS[args.group_by])
        candidates = _representatives(connection, parameters)
        one_failure = _representatives(connection, parameters, leads=True)

    pnl_dir = Path(research_paths.resolve("pnl"))
    annual = {}
    if args.yearly:
        from . import pnl_snapshot

        protect_output(args, pnl_dir / "meta.sqlite", pnl_dir / "meta.sqlite-wal",
                       pnl_dir / "meta.sqlite-shm", pnl_dir / "compact.lock")

        for row in candidates:
            if ml.valid_simulation_id(row["alpha_id"]):
                name = row["alpha_id"] + ".json"
                protect_output(args, pnl_dir / name, pnl_dir / ".meta" / name)
        yearly_rows = pnl_snapshot.yearly_train_batch(candidates, pnl_dir)
        for row, yearly_result in zip(candidates, yearly_rows):
            row["yearly_train"] = yearly_result
            state = yearly_result["state"]
            annual[state] = annual.get(state, 0) + 1
    return {
        "source": "latest_registry_entry_per_key",
        "filters": filters,
        "screen": {
            "basis": "explicit_training_metrics_and_stored_checks",
            "min_train_sharpe": args.min_train_sharpe,
            "max_turnover": args.max_turnover,
            "deducted_cost_bps": args.cost_bps,
            "margin_definition": "train_margin * 10000 - deducted_cost_bps",
            "representatives": "one per declared mechanism label; unlabeled entries share one bucket",
            "platform_submission_status": "not_checked",
        },
        "counts": counts,
        "funnel": funnel,
        "failure_drivers": failures,
        "group_by": args.group_by,
        "groups": groups,
        "representatives": candidates,
        "single_failure_leads": one_failure,
        "yearly_train": {"enabled": args.yearly, "representative_counts": annual},
    }
