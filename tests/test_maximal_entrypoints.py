"""Only maximal is exposed, and deferred-copy inputs never switch engines."""
import subprocess
import sys

import pytest

from pipeguard import check_module
from pipeguard.engine import Executor
from test_checker import load
from test_deferred_cp_arrival import deferred_case


@pytest.mark.parametrize('name', ['crosscheck', 'exhaustive', 'unknown'])
def test_removed_engine_is_rejected(name):
    with pytest.raises(ValueError, match='only maximal'):
        check_module(load('safe_handoff.json'), engine=name)
    run = subprocess.run([sys.executable, '-m', 'pipeguard', 'check', 'unused.json',
                          '--engine', name], capture_output=True, text=True)
    assert run.returncode == 2
    assert 'invalid choice' in run.stderr


def test_no_exploration_entrypoint():
    assert not hasattr(Executor, 'explore_all')


@pytest.mark.parametrize('commit', [False, True])
def test_expanded_deferred_copy_uses_incremental_maximal(commit, monkeypatch):
    def forbid_scan(*args, **kwargs):
        raise AssertionError('deferred copy fell back to a scanning scheduler')
    monkeypatch.setattr(Executor, 'explore', forbid_scan)
    report = check_module(deferred_case(commit=commit).expand_reference())
    assert report['verdict'] == 'SAFE', report
    stats = report['statistics']
    assert stats['engine'] == 'maximal'
    assert stats['explored_states'] == 0
    assert stats['full_enabled_scans'] == 0
    assert stats['state_clones'] == 0
