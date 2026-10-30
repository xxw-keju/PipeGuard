"""Commit depth is not a protocol error; only the observed prefix orders use."""
from copy import deepcopy

import pytest

from pipeguard.checker import check_module
from pipeguard.model import module_from_dict, normalize_module
from pipeguard.engine import Executor
from pipeguard.parametric import normalize_parametric_module
from test_checker import cp_async_module, tma_store_module, wgmma_module
from test_parametric import _single_stream_parametric


def grouped_program(protocol, count, wait_count, read_group):
    if protocol == 'CP_ASYNC':
        raw = deepcopy(cp_async_module(count, wait_count, read_group).raw)
        raw['events'][-1]['type'] = 'MemRead'
        return raw
    if protocol == 'WGMMA':
        raw = deepcopy(wgmma_module(wait_operands=False, overwrite_operand=False).raw)
        fence, issue, commit, wait, read = raw['events']
        destination = raw['regions'][1]
        events = [fence]
    else:
        raw = deepcopy(tma_store_module(read_only_wait=False, overwrite_source=False).raw)
        issue, commit, wait, read = raw['events']
        destination = raw['regions'][1]
        events = []
    raw['regions'] = [raw['regions'][0]]
    for group in range(count):
        region = dict(destination, id=f'destination{group}', offset=group*destination['extent'])
        raw['regions'].append(region)
        events.extend([
            dict(issue, id=f'issue{group}', destination_region=region['id']),
            dict(commit, id=f'commit{group}', group_seq=group),
        ])
    events.extend([dict(wait, wait_group_count=wait_count),
                   dict(read, region=f'destination{read_group}')])
    for seq, event in enumerate(events):
        event['seq'] = seq
    raw['events'] = events
    return raw


@pytest.mark.parametrize('protocol', ['CP_ASYNC', 'WGMMA', 'TMA_STORE'])
@pytest.mark.parametrize('count', [2, 9, 32])
@pytest.mark.parametrize('coverage', ['all', 'oldest', 'uncovered'])
@pytest.mark.parametrize('representation', ['expanded', 'parametric'])
def test_wait_prefix_independent_of_group_depth(protocol, count, coverage, representation):
    wait_count = 0 if coverage == 'all' else 1
    read_group = 0 if coverage == 'oldest' else count-1
    raw = grouped_program(protocol, count, wait_count, read_group)
    # Omitted and obsolete tiny capacities have identical semantics.
    if count == 2:
        raw['resources'][0]['capacity'] = 1
    else:
        raw['resources'][0].pop('capacity', None)
    if representation == 'parametric':
        raw = _single_stream_parametric(raw)
    report = check_module(module_from_dict(raw), engine='maximal')
    expected = ('SYNC_HAZARD' if protocol == 'WGMMA' else 'DATA_HAZARD') if coverage == 'uncovered' else 'SAFE'
    assert report['verdict'] == expected, report.get('diagnostics')
    if coverage == 'uncovered':
        assert report['diagnostics'][0]['code'] == ('WGMMA_RESULT_NOT_READY' if protocol == 'WGMMA' else 'UNORDERED_CONFLICT')


@pytest.mark.parametrize('representation', ['expanded', 'parametric'])
def test_bulk_read_wait_does_not_advance_global_completion(representation):
    raw = grouped_program('TMA_STORE', 32, 0, 31)
    raw['resources'][0].pop('capacity', None)
    raw['events'][-2]['wait_mode'] = 'READ_ONLY'
    if representation == 'parametric':
        raw = _single_stream_parametric(raw)
    assert check_module(module_from_dict(raw))['verdict'] == 'DATA_HAZARD'


@pytest.mark.parametrize('representation', ['expanded', 'parametric'])
def test_nine_copies_wait_eight_only_orders_oldest(representation):
    # Same commit/wait shape as tilekernels-9b264da9's compiled prologue.
    for consumed, expected in [(0, 'SAFE'), (1, 'DATA_HAZARD'), (8, 'DATA_HAZARD')]:
        raw = grouped_program('CP_ASYNC', 9, 8, consumed)
        if representation == 'parametric':
            raw = _single_stream_parametric(raw)
        assert check_module(module_from_dict(raw))['verdict'] == expected


@pytest.mark.parametrize('protocol', ['CP_ASYNC', 'WGMMA', 'TMA_STORE'])
@pytest.mark.parametrize('representation', ['expanded', 'parametric'])
def test_normalization_discards_legacy_group_capacity(protocol, representation):
    raw = grouped_program(protocol, 2, 0, 1)
    raw['resources'][0]['capacity'] = 1
    if representation == 'parametric':
        raw = _single_stream_parametric(raw)
    module = module_from_dict(raw)
    normalized = (normalize_parametric_module if representation == 'parametric' else normalize_module)(module)
    assert 'capacity' not in normalized['resources'][0]
    assert check_module(module_from_dict(normalized))['verdict'] == 'SAFE'


def test_exploration_budget_exhaustion_is_inconclusive(monkeypatch):
    monkeypatch.setattr(Executor, 'MAX_STATES', 1)
    report = check_module(cp_async_module(2, overwrite_group=None), engine='exhaustive')
    assert report['verdict'] == 'INCONCLUSIVE'
    assert report['diagnostics'][0]['code'] == 'EXPLORATION_LIMIT'
