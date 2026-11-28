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

# Resolve the core modules lazily. Importing all of them here would make the
# package unimportable in the other direction, because every brain_core module
# imports this one for its constants and error types.
_CORE_MODULES = ("errors", "transport", "auth", "catalog", "client", "simulations", "alphas")


def __getattr__(name):
    """Expose a core symbol through this facade, importing its module on first use."""
    import importlib

    for module_name in _CORE_MODULES:
        module = importlib.import_module("brain_core.%s" % module_name)
        try:
            value = getattr(module, name)
        except AttributeError:
            continue
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
