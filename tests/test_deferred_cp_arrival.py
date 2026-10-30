"""noinc is an asynchronous completion signal, never a program-thread wait."""
from copy import deepcopy

import pytest

from pipeguard.checker import check_module
from pipeguard.parametric import parametric_module_from_dict
from pipeguard.model import PGIRError
from test_parametric import parametric_memory_module


def deferred_case(*, commit=False, early=None, generic=False, count=1, parity=0,
                  double=False, late_commit=False, reuse=False):
    raw = parametric_memory_module()
    raw['resources'] = [
        {'id': 'bar', 'kind': 'mbarrier', 'participants': [{'stream': 'lane', 'instances': [0, 1]}]},
        {'id': 'cp', 'kind': 'async_group', 'protocol': 'CP_ASYNC',  'scope': 'CTA',
         'participants': [{'stream': 'lane', 'instances': [0]}]},
    ]
    raw['regions'] += [
        {'id': 'src', 'allocation_id': 'input', 'address_space': 'GLOBAL', 'offset': 0, 'extent': 4},
        {'id': 'side', 'allocation_id': 'other-shared', 'address_space': 'SHARED', 'offset': 0, 'extent': 4},
    ]
    events = []
    def add(name, kind, lane, **fields):
        e = dict(id=name, type=kind, stream='lane', order=len(events),
                 execution_scope='INSTANCE', instances=[lane], fields=fields)
        if lane == 1:
            e['dependencies'] = [{'template': 'init', 'relation': 'ALL_INSTANCES'}]
        events.append(e)
    def arrive(name, generation=0):
        add(name, 'CpAsyncMBarrierArrive', 0, resource='bar', async_resource='cp',
            group_owner='$self', lifecycle_epoch=0, generation=generation, count=1)
    def wait(name, generation=0, lane=1):
        add(name, 'MBarrierWaitParity', lane, resource='bar', lifecycle_epoch=0,
            generation=generation, raw_parity=parity if not generation else generation % 2,
            parity_evidence=dict(derivation_kind='PHASE_SSA', logical_stream_id=f'wait:{lane}',
                                 lifecycle_epoch=0, stream_rank=generation, polling_coalesced=True))
    add('init', 'MBarrierInit', 0, resource='bar', lifecycle_epoch=0, count=count)
    if generic:
        add('generic', 'MemWrite', 0, region='side')
    add('copy', 'CpAsyncIssue', 0, resource='cp', group_owner='$self', source_region='src',
        destination_region='shared', bytes=4)
    if commit:
        add('commit', 'CpAsyncCommitGroup', 0, resource='cp', group_owner='$self', group_seq=0)
    arrive('noinc')
    if double:
        arrive('extra')
    if early:
        add('early', early, 0, region='shared')
    if late_commit:
        add('late-commit', 'CpAsyncCommitGroup', 0, resource='cp', group_owner='$self', group_seq=0)
    wait('wait')
    add('consume', 'MemRead', 1, region='side' if generic else 'shared')
    if reuse:
        wait('producer-wait', lane=0)
        arrive('next-noinc', 1)
        wait('next-wait', 1)
    raw['event_templates'] = events
    return parametric_module_from_dict(raw)


@pytest.mark.parametrize('options,expected', [
    ({}, 'SAFE'),
    ({'commit': True}, 'SAFE'),
    ({'late_commit': True}, 'SAFE'),
    ({'early': 'MemRead'}, 'DATA_HAZARD'),
    ({'early': 'MemWrite'}, 'DATA_HAZARD'),
    ({'generic': True}, 'DATA_HAZARD'),
    ({'count': 2}, 'SYNC_HAZARD'),
    ({'parity': 1}, 'SYNC_HAZARD'),
    ({'double': True}, 'SYNC_HAZARD'),
])
def test_deferred_arrival_direct_and_independent_oracle(options, expected):
    module = deferred_case(**options)
    direct = check_module(module, engine='maximal')
    assert direct['verdict'] == expected, direct.get('diagnostics')
    if options.get('parity'):
        with pytest.raises(PGIRError, match='raw wait parity'):
            module.expand_reference()
        return
    reference = check_module(module.expand_reference(), engine='exhaustive')
    assert reference['verdict'] == expected, reference.get('diagnostics')


def test_noinc_reuse_needs_successful_observation():
    module = deferred_case(reuse=True)
    assert check_module(module, engine='maximal')['verdict'] == 'SAFE'
    raw = deepcopy(module.raw)
    raw['event_templates'] = [e for e in raw['event_templates'] if e['id'] != 'producer-wait']
    assert check_module(parametric_module_from_dict(raw), engine='maximal')['verdict'] == 'SYNC_HAZARD'


@pytest.mark.parametrize('observe,expected', [(True, 'SAFE'), (False, 'DATA_HAZARD')])
def test_mbarrier_observation_orders_copies_across_groups(observe, expected):
    base = deferred_case(commit=True).raw
    base['resources'][1]['capacity'] = 2
    templates = []
    # Separate barriers make lack of completion observation independent of
    # barrier reuse. Without handoff, the next copy overwrites an in-flight
    # destination: a memory conflict, independent of the legacy capacity.
    base['resources'] = [base['resources'][1]]
    for iteration in range(3):
        bar = f'bar{iteration}'
        base['resources'].append({'id': bar, 'kind': 'mbarrier',
                                  'participants': [{'stream': 'lane', 'instances': [0]}]})
        for source in deferred_case(commit=True).raw['event_templates']:
            if source['id'] == 'consume' or source['id'] == 'wait' and not observe:
                continue
            item = deepcopy(source)
            item['id'] += str(iteration)
            item['order'] = len(templates)
            item['instances'] = [0]
            item.pop('dependencies', None)
            fields = item['fields']
            if fields.get('resource') == 'bar': fields['resource'] = bar
            if 'group_seq' in fields: fields['group_seq'] = iteration
            if 'parity_evidence' in fields:
                fields['parity_evidence']['logical_stream_id'] = bar
            templates.append(item)
    base['event_templates'] = templates
    module = parametric_module_from_dict(base)
    assert check_module(module, engine='maximal')['verdict'] == expected
    assert check_module(module.expand_reference(), engine='exhaustive')['verdict'] == expected


@pytest.mark.parametrize('observe,expected', [(True, 'SAFE'), (False, 'DATA_HAZARD')])
def test_noinc_does_not_implicitly_commit_group(observe, expected):
    raw = deepcopy(deferred_case().raw)
    if not observe:
        raw['event_templates'] = [e for e in raw['event_templates'] if e['id'] != 'wait']
        for e in raw['event_templates']:
            if e['id'] == 'consume':
                e['instances'] = [0]
                e.pop('dependencies', None)
    raw['event_templates'].insert(-1, dict(id='empty-group-wait', type='CpAsyncWaitGroup', stream='lane',
        order=99, execution_scope='INSTANCE', instances=[0],
        fields=dict(resource='cp', group_owner='$self', wait_group_count=0)))
    for order, event in enumerate(raw['event_templates']): event['order'] = order
    module = parametric_module_from_dict(raw)
    assert check_module(module, engine='maximal')['verdict'] == expected
    assert check_module(module.expand_reference(), engine='exhaustive')['verdict'] == expected
