"""BRAIN account sessions, catalog metadata, simulations and Alpha results.

Use load_session to reuse a private, account-bound session. Catalog datasets,
fields and categories are cached per account and query scope.

request_json handles ordinary API requests. Simulation progress and Alpha
recordsets have endpoint-specific readiness protocols. simulate_alphas uses
durable reservations to continue accepted work after interruption.

Connection setup imports the HTTP client on demand. Command help and offline
previews can run without it. DataFrame helpers load ``polars`` on demand."""

import re
import threading
import time
from os import environ
from pathlib import Path
from time import sleep

import cache_store
import research_paths

_diagnostic_handler = None

SUBMIT_UNLOCK_ENV = "ALLOW_AGENT_SUBMIT"

#: Base payload for ``POST /simulations``. Callers override individual keys;
#: see ``resolve_settings`` for merge + validation rules.
DEFAULT_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 6,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "testPeriod": "P0Y",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": False,
}

#: Regions the API accepts. Which ones actually return data depends on the
#: account's permissions (a TUTORIAL account only sees USA).
REGIONS = [
    "USA", "GLB", "EUR", "ASI", "CHN", "HKG", "JPN", "KOR", "TWN", "IND",
    "AMR", "EMEA", "APAC",
]

#: Known neutralization values accepted by the API. **A hint list, not a local gate.**
#: The platform owns this enum; well-formed uppercase tokens outside this list are still
#: sent so new platform options work without client code changes, while malformed tokens
#: fail locally as input typos.
NEUTRALIZATIONS = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY",
                   "STATISTICAL", "FAST"]

#: Shape of a settings enum token: upper case, digits, underscores.
_SETTING_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Non-exhaustive list of common universes, used for input validation hints.
UNIVERSES = [
    "TOP3000", "TOP1000", "TOP600", "TOP500", "TOP200", "TOP100",
    "TOPSP500", "TOPSP1000", "MINVOL1M", "ILLIQUID_MINVOL1M",
]

_DOTENV_LOADED = False

_DOTENV_LOCK = threading.Lock()

DEFAULT_METADATA_TTL = float(environ.get("BRAIN_METADATA_TTL", 24 * 3600) or 0)

_metadata_policy = {"enabled": True, "max_age": DEFAULT_METADATA_TTL, "refresh": False}

BRAIN_API_URL = "https://api.worldquantbrain.com"

QUICK_ORPHAN_SWEEP_SECONDS = 5

DEFAULT_LOCAL_CONCURRENCY = 3

SPLIT_METRIC_KEYS = (
    "sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
    "longCount", "shortCount", "pnl",
)

#: Values the platform accepts for an alpha's category. Discovered by probing the
#: API: an invalid value returns 400 without listing the choices, so the set is
#: assembled from what is accepted.
ALPHA_CATEGORIES = ("ANALYST", "FUNDAMENTAL", "PRICE_VOLUME", "SENTIMENT", "VOLUME")

ALPHA_COLORS = ("RED", "YELLOW", "GREEN", "BLUE", "PURPLE")

_UNSET = object()

# Re-export cohesive brain_core submodules behind the canonical machine_lib facade.
from brain_core.errors import (  # noqa: E402
    set_diagnostic_handler, note, BrainError, MissingCredentials, AuthError, CaptchaRequired, NotFound, RateLimited, SimulationError, SubmissionDisabled, agent_submit_allowed, ServiceError, ReadInterrupted, AuthenticationUnavailable, AuthenticationDeadline, SessionBusy, PermissionDenied,
)
from brain_core.auth import (  # noqa: E402
    _load_dotenv, configured_username, get_credentials, login, authenticate, is_python_language, normalize_python_source, alpha_language, regular_expression, session_cache_path, _session_payload, save_session, _session_from_entry, session_status, invalidate_session, logout_session, fetch_user_profile, fetch_competitions, fetch_user_surface, fetch_consultant_summary, login_lock_timeout, isolated_session, load_session,
)
from brain_core.catalog import (  # noqa: E402
    configure_metadata_cache, metadata_cache_policy, metadata_cache_path, read_cached_field_snapshot, migrate_legacy_datafield_cache, cached_metadata, CatalogAccess, _catalog_account, catalog_field_cache_paths, _cached_account_catalog, get_datasets, _get_datasets, _as_results, _catalog_pages, fetch_operators, _fetch_operators, fetch_operator_doc, _fetch_operator_doc, fetch_data_categories, _fetch_data_categories, get_datafields, _get_datafields, _filter_datafields, search_datafield, get_dataset, get_datafield, field_ids, fields_to_df, process_datafields,
)
from brain_core.transport import (  # noqa: E402
    _SIMULATION_ID, valid_simulation_id, simulation_reference_id, _validated_progress_url, _quick_progress_target, _request_timeout, _retry_after_seconds, _TRANSIENT_READ_STATUSES, _transient_read_error, _wait_for_read_retry, _get_poll_response, _poll_json_when_ready, request_json,
)
from brain_core.simulations import (  # noqa: E402
    read_simulation_snapshot, _read_simulation_snapshot, cancel_simulation, simulation_progress_receipt, recover_accepted_slots, recover_quick_orphans, attach_quick_location, poll_simulation, submit_single_simulation, submit_multi_simulation, _has_multi_simulation_permission, _submit_simulation_request, fetch_alpha_metrics, is_quick_simulation, alpha_classification_ids, defer_alpha_metrics, normalize_setting_keys, resolve_settings, _REGULAR_UNSENT_SETTINGS, _PYTHON_UNSENT_SETTINGS, _regular_unsent_settings, _omit_regular_settings, regular_wire_settings, format_simulation_payload, _single_simulation_data, regular_settings_evidence, _ALPHA_ENTRY_KEYS, normalize_alphas, alpha_candidate_ids, alpha_families, alpha_labels, concurrency_plan, simulate_alphas,
)
from brain_core.alphas import (  # noqa: E402
    _OPERATOR_MENTION, _OPERATOR_FATAL, note_operator_rejection, _CEILING_CHECKS, reconcile_check, _extract_checks, fetch_alpha_checks, _performance_request, fetch_merged_performance, _recordset_paths, _read_recordset, fetch_alpha_recordsets, _PNL_EVIDENCE_FORMAT, _pnl_digest, _matching_pnl_evidence, pnl_cache_status, _write_pnl_cache, fetch_daily_pnl, read_alpha_submission_status, submit_alpha, list_alphas, extract_split, set_alpha_metadata, _CORRELATION_KINDS, _fetch_alpha_correlation_endpoint, fetch_self_correlation, fetch_prod_correlation, build_prod_correlation_archive, fetch_power_pool_correlation, power_pool_correlation_verdict, self_correlation_verdict,
)
from brain_core.client import (  # noqa: E402
    BrainClient,
)
