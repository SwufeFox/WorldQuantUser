"""Run ordinary tests, including CLI subprocesses, without remote networking."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import offline_network


def pytest_configure(config):
    workspace = TemporaryDirectory(prefix="worldquant-pytest-")
    config.add_cleanup(workspace.cleanup)
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    patch.setenv(offline_network.WORKSPACE_ENV, workspace.name)
    patch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent), prepend=os.pathsep)
    offline_network.isolate_environment(patch.setenv, workspace.name)
    offline_network.install(patch.setattr)
