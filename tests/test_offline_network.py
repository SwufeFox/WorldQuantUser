"""The network guard applies before endpoint lookup, including in CLI subprocesses."""

import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from offline_network import RemoteNetworkBlocked


def test_remote_endpoint_is_blocked():
    with socket.socket() as sock, pytest.raises(RemoteNetworkBlocked):
        sock.connect(("remote.invalid", 443))
    with pytest.raises(RemoteNetworkBlocked):
        socket.getaddrinfo("remote.invalid", 443)


def test_cli_subprocess_inherits_offline_policy():
    root = Path(__file__).resolve().parent.parent
    # The usual CLI contract helper strips BRAIN_* from the child environment.
    env = {key: value for key, value in os.environ.items() if not key.startswith("BRAIN_")}
    script = """
import os, runpy, socket, sys
from offline_network import RemoteNetworkBlocked, WORKSPACE_ENV
assert os.environ['BRAIN_RESEARCH_DIR'] == os.environ[WORKSPACE_ENV]
assert os.environ['BRAIN_USERNAME'] == os.environ['BRAIN_PASSWORD'] == ''
with socket.socket() as sock:
    try:
        sock.connect(('remote.invalid', 443))
    except RemoteNetworkBlocked:
        pass
    else:
        raise AssertionError('remote socket reached from a CLI subprocess')
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    proc = subprocess.run([sys.executable, "-c", script, str(root / "research_cli.py"), "help"],
                          cwd=root, env=env, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
