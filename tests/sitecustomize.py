"""Apply the pytest network policy in Python subprocesses before CLI import."""

import os

import offline_network


workspace = os.environ.get(offline_network.WORKSPACE_ENV)
if workspace:
    offline_network.isolate_environment(os.environ.__setitem__, workspace, child=True)
    offline_network.install(setattr)
