import hashlib
import json
from pathlib import Path

import pytest

from pipeguard.ptx import apply_device_checks, check_wgmma_register_fences
from pipeguard.checker import check_module
from pipeguard.model import module_from_dict
from pipeguard.cli import main

MMA = 'wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%f0,%f1,%f2,%f3}, %rd0, %rd1, 1, 1, 1, 0, 0;'
FENCE = 'wgmma.fence.sync.aligned;'


def kernel(body):
    return '.visible .entry kernel() {\n' + body + '\n}'


def safe_module_with_ptx(text):
    raw = json.loads((Path(__file__).parent / 'fixtures/safe_handoff.json').read_text())
    raw['frontend_provenance']['final_ptx'] = {
        'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
    }
    return module_from_dict(raw)


@pytest.mark.parametrize('engine', ['maximal', 'exhaustive', 'crosscheck'])
@pytest.mark.parametrize('body, enabled_verdict', [
    (FENCE + '\nmov.f32 %f0, 0f3f800000;\n' + MMA, 'SYNC_HAZARD'),
    (FENCE + '\nL_loop:\n' + MMA + '\n@%p bra L_loop;', 'UNSUPPORTED'),
    (FENCE + '\n{ mov.f32 %f0, 0f3f800000; }\n' + MMA, 'UNSUPPORTED'),
])
def test_register_check_is_opt_in_for_embedded_and_external_ptx(engine, body, enabled_verdict):
    text = kernel(body)
    module = safe_module_with_ptx(text)
    for kwargs in ({}, {'final_ptx': text}):
        result = check_module(module, engine=engine, **kwargs)
        assert result['verdict'] == 'SAFE'
        assert result['validation_scope'] == 'PGIR'
        assert result['device_validation']['wgmma_register_fences'] == 'DISABLED'
        assert not result['diagnostics']
        assert 'text' not in result['manifest']['frontend_provenance']['final_ptx']
        enabled = check_module(module, engine=engine, check_ptx_registers=True, **kwargs)
        assert enabled['verdict'] == enabled_verdict
        assert enabled['validation_scope'] == 'PGIR_AND_FINAL_PTX_WGMMA_REGISTERS'
    assert module.raw['frontend_provenance']['final_ptx']['text'] == text


@pytest.mark.parametrize('command', ['check', 'explain'])
@pytest.mark.parametrize('external', [False, True])
def test_cli_register_check_is_opt_in(tmp_path, capsys, command, external):
    text = kernel(FENCE + '\nmov.f32 %f0, 0f3f800000;\n' + MMA)
    raw = safe_module_with_ptx(text).raw
    options = []
    if external:
        raw['frontend_provenance'].pop('final_ptx')
        ptx = tmp_path / 'kernel.ptx'
        ptx.write_text(text)
        options = ['--final-ptx', str(ptx)]
    source = tmp_path / 'pgir.json'
    source.write_text(json.dumps(raw))
    argv = [command, str(source), *options]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['device_validation']['wgmma_register_fences'] == 'DISABLED'
    assert main([*argv, '--check-ptx-registers']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['verdict'] == 'SYNC_HAZARD'
    assert result['diagnostics'][0]['code'] == 'WGMMA_REGISTER_FENCE_MISSING'


@pytest.mark.parametrize('middle', ['', 'wgmma.wait_group.sync.aligned 0;', 'fence.proxy.async.shared::cta;'])
def test_late_accumulator_write_needs_register_fence(middle):
    result = check_wgmma_register_fences(kernel(FENCE + '\nmov.f32 %f0, 0f3f800000;\n' + middle + '\n' + MMA))
    assert result['status'] == 'CHECKED'
    assert result['diagnostics'][0]['code'] == 'WGMMA_REGISTER_FENCE_MISSING'
    assert result['diagnostics'][0]['details']['prior_accesses'] == {'%f0': 3}


@pytest.mark.parametrize('body', [
    'mov.f32 %f0, 0f3f800000;\n' + FENCE + '\n' + MMA,
    FENCE + '\n' + MMA + '\n' + MMA,
    FENCE + '\nmov.f32 %unrelated, 0f3f800000;\n' + MMA,
    FENCE + '\n// mov.f32 %f0, 0f3f800000;\n' + MMA,
    FENCE + '\nmov.pred %p, 0;\n@%p mov.f32 %f0, 0f3f800000;\n' + MMA,
])
def test_correct_fence_and_same_shape_accumulator_chaining(body):
    result = check_wgmma_register_fences(kernel(body))
    assert result['status'] == 'CHECKED'
    assert result['diagnostics'] == []


def test_first_mma_requires_fence_even_without_prior_mov():
    assert check_wgmma_register_fences(kernel(MMA))['diagnostics'][0]['code'] == 'WGMMA_REGISTER_FENCE_MISSING'


def test_ordinary_read_also_needs_fence_before_reuse():
    result = check_wgmma_register_fences(kernel(FENCE + '\n' + MMA + '\nwgmma.wait_group.sync.aligned 0;\nst.global.f32 [%rd4], %f0;\n' + MMA))
    assert len(result['diagnostics']) == 1
    assert '%f0' in result['diagnostics'][0]['details']['prior_accesses']


def test_input_a_register_write_requires_fence():
    mma = MMA.replace('%rd0,', '{%r0,%r1,%r2,%r3},')
    result = check_wgmma_register_fences(kernel(FENCE + '\nmov.b32 %r2, 0;\n' + mma))
    assert '%r2' in result['diagnostics'][0]['details']['prior_accesses']


@pytest.mark.parametrize('body', [
    FENCE + '\n@%p bra L_skip;\nmov.f32 %f0, 0f3f800000;\nL_skip:\n' + MMA,
    FENCE + '\n@%p mov.f32 %f0, 0f3f800000;\n' + MMA,
    '@%p ' + FENCE + '\n' + MMA,
    FENCE + '\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16;',
])
def test_unresolved_path_is_not_treated_as_straight_line_proof(body):
    result = check_wgmma_register_fences(kernel(body))
    assert result['status'] == 'UNSUPPORTED'
    assert result['diagnostics'][0]['code'] == 'PTX_WGMMA_REGISTER_ORDER_UNRESOLVED'


def test_device_finding_keeps_independent_data_hazard():
    text = kernel(FENCE + '\nmov.f32 %f0, 0f3f800000;\n' + MMA)
    evidence = {'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    original = {'verdict': 'DATA_HAZARD', 'diagnostics': [{'code': 'UNORDERED_CONFLICT'}]}
    result = apply_device_checks(original, evidence)
    assert result['verdict'] == 'DATA_HAZARD'
    assert [d['code'] for d in result['diagnostics']] == ['UNORDERED_CONFLICT', 'WGMMA_REGISTER_FENCE_MISSING']
    assert len(original['diagnostics']) == 1
    result = apply_device_checks({'verdict': 'SAFE', 'diagnostics': []}, evidence)
    assert result['verdict'] == 'SYNC_HAZARD'
    evidence['text'] += '\n'
    assert apply_device_checks({'verdict': 'SAFE'}, evidence)['verdict'] == 'UNSUPPORTED'


def test_backedge_after_last_textual_mma_cannot_hide_next_iteration_access():
    text = kernel(FENCE + '\nL_loop:\n' + MMA + '\nmov.f32 %f0, 0f3f800000;\n@%p bra L_loop;')
    assert check_wgmma_register_fences(text)['status'] == 'UNSUPPORTED'


def test_input_register_read_read_does_not_invent_fence_requirement():
    mma = MMA.replace('%rd0,', '{%r0,%r1,%r2,%r3},')
    assert check_wgmma_register_fences(kernel(FENCE + '\n' + mma + '\n' + mma))['diagnostics'] == []


@pytest.mark.parametrize('text', [
    '.visible .entry kernel() { ' + MMA + ' }',
    kernel(FENCE + '\n{ mov.f32 %f0, 0f3f800000; }\n' + MMA),
    kernel(FENCE + '\nst.global.f32 [%rd4],\n%f0;\n' + MMA),
    kernel(FENCE + '\n' + MMA.replace('%f0', 'named_register')),
])
def test_unparsed_register_access_cannot_be_certified(text):
    assert check_wgmma_register_fences(text)['status'] == 'UNSUPPORTED'
