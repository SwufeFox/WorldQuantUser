"""Stateful BrainClient context-manager facade over BRAIN API operations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from functools import cache
from os import environ
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import cache_store
from check_evidence import normalize_checks, platform_check_names
import machine_lib as _ml
import research_paths


class BrainClient:
    """Ergonomic, session-managed Python facade for WorldQuant BRAIN.

    Lazily loads the account-bound session cache on first network access and
    automatically refreshes once if BRAIN rejects an expired session cookie.
    Offline calls such as ``simulate(..., dry_run=True)`` never touch the network.
    """

    def __init__(self, session=None, *, cache_path=None, use_cache=True, refresh=False):
        self._session = session
        self._external_session = session is not None
        self._cache_path = cache_path
        self._use_cache = use_cache
        self._refresh = refresh

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
        return False

    def close(self):
        if self._session is not None and not self._external_session:
            close_fn = getattr(self._session, "close", None)
            if callable(close_fn):
                close_fn()
            self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = _ml.load_session(
                cache_path=self._cache_path,
                use_cache=self._use_cache,
                refresh=self._refresh,
            )
            self._refresh = False
        return self._session

    def _call(self, fn, *args, **kwargs):
        retried = False
        while True:
            try:
                return fn(self.session, *args, **kwargs)
            except _ml.AuthError as exc:
                if retried or self._external_session or exc.code != "AUTH_FAILED":
                    raise
                retried = True
                self._session = _ml.load_session(
                    cache_path=self._cache_path, use_cache=False, refresh=True
                )

    def _catalog_access(self):
        if self._external_session:
            return self._session
        return _ml.CatalogAccess(_ml.configured_username(), lambda work: self._call(work))

    # --- Account & session ---
    def profile(self, *, deadline=None):
        return self._call(_ml.fetch_user_profile, deadline=deadline)

    def competitions(self, *, mine=True, competition_id=None, deadline=None):
        return self._call(
            _ml.fetch_competitions, mine=mine, competition_id=competition_id, deadline=deadline
        )

    def consultant(self, *, deadline=None):
        return self._call(_ml.fetch_consultant_summary, deadline=deadline)

    def logout(self, *, remote=False):
        result = _ml.logout_session(
            self._session, cache_path=self._cache_path, remote=remote
        )
        self.close()
        return result

    # --- Catalog ---
    def datasets(self, **kwargs):
        return _ml.get_datasets(self._catalog_access(), **kwargs)

    def dataset(self, dataset_id, **kwargs):
        return _ml.get_dataset(self._catalog_access(), dataset_id, **kwargs)

    def fields(self, dataset_id="", **kwargs):
        return _ml.get_datafields(self._catalog_access(), dataset_id=dataset_id, **kwargs)

    def field(self, field_id, **kwargs):
        return _ml.get_datafield(self._catalog_access(), field_id, **kwargs)

    def operators(self, search="", group="", **kwargs):
        return self._call(_ml.fetch_operators, search=search, group=group, **kwargs)

    def operator_doc(self, name, **kwargs):
        return self._call(_ml.fetch_operator_doc, name, **kwargs)

    def categories(self, **kwargs):
        return _ml.fetch_data_categories(self._catalog_access(), **kwargs)

    # --- Simulation ---
    def simulate(self, expressions, settings=None, *, mode=None, dry_run=False, **kwargs):
        items = [expressions] if isinstance(expressions, (str, dict)) else list(expressions)
        merged = dict(settings or {})
        if mode is not None:
            merged["simulationMode"] = str(mode).upper()
        resolved = _ml.resolve_settings(merged)
        if dry_run:
            return _ml.simulate_alphas(None, items, resolved, dry_run=True, **kwargs)
        return self._call(_ml.simulate_alphas, items, resolved, dry_run=False, **kwargs)

    def simulate_one(self, expression, settings=None, *, mode=None, dry_run=False, **kwargs):
        rows = self.simulate([expression], settings=settings, mode=mode, dry_run=dry_run, **kwargs)
        return rows[0] if rows else None

    def cancel_simulation(self, reference, *, deadline=None):
        return self._call(_ml.cancel_simulation, reference, deadline=deadline)

    # --- Alphas & correlations ---
    def alphas(self, **kwargs):
        return self._call(_ml.list_alphas, **kwargs)

    def alpha(self, alpha_id, **kwargs):
        return self._call(_ml.fetch_alpha_metrics, alpha_id, **kwargs)

    def pnl(self, alpha_id, **kwargs):
        return self._call(_ml.fetch_daily_pnl, alpha_id, **kwargs)

    def recordsets(self, alpha_id, names=(), **kwargs):
        return self._call(_ml.fetch_alpha_recordsets, alpha_id, names=names, **kwargs)

    def performance(self, alpha_id, **kwargs):
        return self._call(_ml.fetch_merged_performance, alpha_id, **kwargs)

    def check(self, alpha_id, **kwargs):
        return self._call(_ml.fetch_alpha_checks, alpha_id, **kwargs)

    def self_corr(self, alpha_id, **kwargs):
        return self._call(_ml.fetch_self_correlation, alpha_id, **kwargs)

    def prod_corr(self, alpha_id, **kwargs):
        return self._call(_ml.fetch_prod_correlation, alpha_id, **kwargs)

    def pp_corr(self, alpha_id, **kwargs):
        return self._call(_ml.fetch_power_pool_correlation, alpha_id, **kwargs)

    def describe(self, alpha_id, **kwargs):
        return self._call(_ml.set_alpha_metadata, alpha_id, **kwargs)

    def submit(self, alpha_id, confirm=False, **kwargs):
        return self._call(_ml.submit_alpha, alpha_id, confirm=confirm, **kwargs)
