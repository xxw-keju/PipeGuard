"""Completion, phase and collective-participation regressions from real captures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeguard.checker import check_module
from pipeguard.frontends import triton_snapshot_to_module, tilelang_snapshot_to_module
from test_frontend_capture_tools import (
    FIXTURES, TRITON_PYTHON, TILELANG_PYTHON, _capture, _run_capture,
)
from triton_capture import TritonCapture, CaptureError, _final_ptx_proxy_fence_evidence


def test_plain_arrival_rounds_are_shared_between_partitions():
    capture = TritonCapture.__new__(TritonCapture)
    capture.barrier_init_counts = {('bar', 0): 2}
    capture.tma_issue_generation = {}
    capture.output_operations = []
    for actor in ('producer', 'consumer'):
        for phase in (0, 1):
            for kind in ('arrive', 'wait_parity'):
                capture.output_operations.append(dict(op='triton.mbarrier.' + kind,
                    actor=actor, resource='bar', lifecycle_epoch=0,
                    generation=phase if kind == 'wait_parity' else 99, count=1))
    capture._assign_plain_arrival_generations()
    assert [e['generation'] for e in capture.output_operations if e['op'].endswith('.arrive')] == [0, 1, 0, 1]


def test_ambiguous_unsynchronized_partition_rounds_fail_closed():
    capture = TritonCapture.__new__(TritonCapture)
    capture.barrier_init_counts = {('bar', 0): 1}
    capture.tma_issue_generation = {}
    capture.output_operations = [dict(op='triton.mbarrier.arrive', actor=actor,
        resource='bar', lifecycle_epoch=0, generation=99, count=1) for actor in ('a', 'b')]
    with pytest.raises(CaptureError, match='concurrent arrival prefixes'):
        capture._assign_plain_arrival_generations()


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason='pinned Triton unavailable')
@pytest.mark.parametrize('initial,expected', [(2, 'SAFE'), (1, 'SYNC_HAZARD')])
def test_two_plain_arrivals_respect_the_phase_quorum(tmp_path, initial, expected):
    text = (FIXTURES / 'triton_mbarrier.ttgir.mlir').read_text()
    arrival = next(line for line in text.splitlines(keepends=True) if 'ttng.arrive_barrier' in line)
    text = text.replace('init_barrier %bar, 1', f'init_barrier %bar, {initial}').replace(arrival, arrival * 2)
    path = tmp_path / 'two-arrivals.mlir'; path.write_text(text)
    _, snapshot = _capture(TRITON_PYTHON, 'triton_capture.py', path, 'two-arrivals')
    arrivals = [op for op in snapshot['operations'] if op['op'] == 'triton.mbarrier.arrive']
    assert [op['generation'] for op in arrivals] == ([0, 0] if initial == 2 else [0, 1])
    assert check_module(triton_snapshot_to_module(snapshot), engine='maximal')['verdict'] == expected


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason='pinned Triton unavailable')
@pytest.mark.parametrize('instructions,expected', [
    ('@p bar.sync 1, 128;', 'SYNC_HAZARD'),
    ('@p bar.sync 1, 128; @!p bar.sync 1, 128;', 'SAFE'),
    ('@p bar.sync 1, 128; @p bar.sync 2, 128; @!p bar.sync 2, 128; @!p bar.sync 1, 128;', 'SYNC_HAZARD'),
])
def test_named_inline_barriers_match_invocations_per_lane(tmp_path, instructions, expected):
    text = (FIXTURES / 'triton_effectful_inline_asm.ttgir.mlir').read_text()
    path = tmp_path / 'named.mlir'
    path.write_text(text.replace('@p bar.sync 0;', instructions).replace('$1, 16;', '$1, 64;'))
    _, snapshot = _capture(TRITON_PYTHON, 'triton_capture.py', path, 'named')
    assert check_module(triton_snapshot_to_module(snapshot), engine='maximal')['verdict'] == expected


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason='pinned Triton unavailable')
def test_terminal_uncommitted_copies_are_not_safe_in_direct_engine(tmp_path):
    text = (FIXTURES / 'triton_cp_async.ttgir.mlir').read_text()
    text = '\n'.join(line for line in text.splitlines() if not any(word in line for word in ('%group =', '%wait =', '%loaded =')))
    path = tmp_path / 'missing-commit.mlir'; path.write_text(text)
    _, snapshot = _capture(TRITON_PYTHON, 'triton_capture.py', path, 'missing-commit')
    result = check_module(triton_snapshot_to_module(snapshot), engine='maximal')
    assert result['verdict'] == 'SYNC_HAZARD'
    assert result['diagnostics'][0]['code'] == 'MISSING_CP_ASYNC_COMMIT'


def noinc_ir(count=32, gap=''):
    return '''#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
 tt.func public @copy(%base: !tt.ptr<i32>) {
 %zero = arith.constant 0 : i32
 %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
 %p = tt.splat %base : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
 %src = tt.addptr %p, %range : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
 %dst = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
 %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
 ttng.init_barrier %bar, COUNT : !ttg.memdesc<1xi64, #shared, #smem, mutable>
 %copy = ttg.async_copy_global_to_local %src, %dst : tensor<32x!tt.ptr<i32>, #blocked> -> <32xi32, #shared, #smem, mutable>
 ttng.async_copy_mbarrier_arrive %bar {noIncrement} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
 GAP
 ttng.wait_barrier %bar, %zero : !ttg.memdesc<1xi64, #shared, #smem, mutable>
 %read = ttg.local_load %dst : !ttg.memdesc<32xi32, #shared, #smem, mutable> -> tensor<32xi32, #blocked>
 tt.return
 }
}
'''.replace('COUNT', str(count)).replace('GAP', gap)


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason='pinned Triton unavailable')
@pytest.mark.parametrize('count,expected', [(32, 'SAFE'), (64, 'SYNC_HAZARD')])
def test_noinc_tracks_uncommitted_copies(tmp_path, count, expected):
    path = tmp_path / 'noinc.mlir'; path.write_text(noinc_ir(count))
    _, snapshot = _capture(TRITON_PYTHON, 'triton_capture.py', path, 'noinc')
    assert check_module(triton_snapshot_to_module(snapshot), engine='maximal')['verdict'] == expected


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason='pinned Triton unavailable')
@pytest.mark.parametrize('variant', ['intervening-read', 'incrementing'])
def test_noinc_never_invents_blocking_completion_for_general_programs(tmp_path, variant):
    gap = '%early = ttg.local_load %dst : !ttg.memdesc<32xi32, #shared, #smem, mutable> -> tensor<32xi32, #blocked>'
    text = noinc_ir(gap=gap) if variant == 'intervening-read' else noinc_ir().replace('{noIncrement}', '')
    path = tmp_path / 'noinc-unresolved.mlir'; path.write_text(text)
    result = _run_capture(TRITON_PYTHON, 'triton_capture.py', path, 'noinc-unresolved')
    if variant == 'incrementing':
        assert result.returncode == 2
        assert json.loads(result.stdout)['diagnostics'][0]['code'] == 'TRITON_CP_ASYNC_MBARRIER_UNRESOLVED'
    else:
        assert result.returncode == 0, result.stderr
        assert check_module(triton_snapshot_to_module(json.loads(result.stdout)), engine='maximal')['verdict'] == 'DATA_HAZARD'


def test_multi_fence_ptx_maps_groups_and_rejects_control_flow(tmp_path):
    text = '.visible .entry kernel() {\n' + ('fence.proxy.async.shared::cta;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16;\nwgmma.commit_group.sync.aligned;\n' * 2) + '}\n'
    path = tmp_path / 'groups.ptx'; path.write_text(text)
    assert _final_ptx_proxy_fence_evidence(path)['proxy_fence_before_groups'] == [True, True]
    path.write_text(text.replace('wgmma.commit_group', 'bra label;\nwgmma.commit_group', 1))
    with pytest.raises(CaptureError, match='straight-line'):
        _final_ptx_proxy_fence_evidence(path)


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason='pinned Triton unavailable')
@pytest.mark.parametrize('threshold,expected', [(64, 'SYNC_HAZARD'), (128, 'SAFE')])
def test_inline_wgmma_wait_keeps_physical_participation(tmp_path, threshold, expected):
    text = (FIXTURES / 'triton_wgmma_fenced.ttgir.mlir').read_text()
    line = next(line for line in text.splitlines() if 'ttng.warp_group_dot_wait' in line)
    replacement = f'    %waited = tt.elementwise_inline_asm "{{ .reg .pred p; .reg .u32 tid; mov.u32 tid, %tid.x; setp.lt.u32 p, tid, {threshold}; @p wgmma.wait_group.sync.aligned 0; mov.b32 $0, $1; }}" {{constraints = "=f,f", packed_element = 1 : i32, pure = false}} %2 : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #mma>'
    path = tmp_path / 'inline-wgmma.mlir'; path.write_text(text.replace(line, replacement))
    _, snapshot = _capture(TRITON_PYTHON, 'triton_capture.py', path, 'inline-wgmma')
    assert check_module(triton_snapshot_to_module(snapshot), engine='maximal')['verdict'] == expected


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason='pinned Triton unavailable')
@pytest.mark.parametrize('publish_before_write,expected', [(False, 'SAFE'), (True, 'DATA_HAZARD')])
def test_partition_arrival_publishes_peer_writes_only_before_arrive(tmp_path, publish_before_write, expected):
    text = (FIXTURES / 'triton_partition_arrive_handoff.ttgir.mlir').read_text()
    if publish_before_write:
        lines = text.splitlines(keepends=True)
        store = next(i for i, line in enumerate(lines) if 'ttg.local_store' in line)
        assert 'ttng.arrive_barrier' in lines[store + 1]
        lines[store], lines[store + 1] = lines[store + 1], lines[store]
        text = ''.join(lines)
    path = tmp_path / 'handoff.mlir'; path.write_text(text)
    evidence = json.loads((FIXTURES / 'triton_partition_arrive_handoff.binding.json').read_text())
    evidence['source_ir_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = tmp_path / 'binding.json'; binding.write_text(json.dumps(evidence))
    _, snapshot = _capture(TRITON_PYTHON, 'triton_capture.py', path, 'handoff', binding_evidence=binding)
    assert check_module(triton_snapshot_to_module(snapshot), engine='maximal')['verdict'] == expected


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason='pinned TileLang unavailable')
@pytest.mark.parametrize('threads,expected', [(64, 'SYNC_HAZARD'), (128, 'SAFE')])
def test_tilelang_aligned_wait_requires_all_128_threads(tmp_path, threads, expected):
    import subprocess
    code = '''import sys
import tilelang
from tilelang import tvm
T = tvm.tirx
tx = T.Var("tx", "int32")
iv = T.IterVar(tvm.ir.Range(0, 128), tx, T.IterVar.ThreadIndex, "threadIdx.x")
wait = T.Evaluate(T.Call("handle", tvm.ir.Op.get("tl.warpgroup_wait"), [T.IntImm("int32", 0)]))
body = T.AttrStmt(iv, "thread_extent", 128, T.IfThenElse(tx < int(sys.argv[2]), wait, None))
f = T.PrimFunc([], body).with_attr("target", tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})).with_attr("thread_extent", {"threadIdx.x": 128})
open(sys.argv[1], "w").write(tvm.ir.save_json(tvm.IRModule({"kernel": f})))
'''
    path = tmp_path / 'wait.json'
    generated = subprocess.run([str(TILELANG_PYTHON), '-c', code, str(path), str(threads)], capture_output=True)
    assert generated.returncode == 0, generated.stderr.decode()
    _, snapshot = _capture(TILELANG_PYTHON, 'tilelang_capture.py', path, 'aligned-wait')
    assert check_module(tilelang_snapshot_to_module(snapshot), engine='maximal')['verdict'] == expected


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason='pinned TileLang unavailable')
def test_tilelang_noinc_does_not_order_intervening_memory(tmp_path):
    import subprocess
    code = '''import sys
import tilelang
from tilelang import tvm
T = tvm.tirx
m = tvm.ir.load_json(open(sys.argv[1]).read())
f = next(iter(m.functions.values()))
stores = []
T.stmt_functor.post_order_visit(f.body, lambda node: stores.append(node) if isinstance(node, T.BufferStore) else None)
def rewrite(node):
    if isinstance(node, T.Evaluate) and isinstance(node.value, T.Call) and node.value.op.name == "tl.ptx_cp_async_barrier_noinc":
        return T.SeqStmt([node, T.BufferStore(stores[0].buffer, T.FloatImm("float32", 1.0), [T.IntImm("int32", 0)])])
body = T.stmt_functor.ir_transform(f.body, None, rewrite, ["tirx.Evaluate"])
open(sys.argv[2], "w").write(tvm.ir.save_json(tvm.IRModule({"kernel": f.with_body(body)})))
'''
    path = tmp_path / 'noinc-intervening.json'
    subprocess.run([str(TILELANG_PYTHON), '-c', code, str(FIXTURES / 'tilelang_cp_async_noinc_device.tvm.json'), str(path)], check=True, capture_output=True)
    result = _run_capture(TILELANG_PYTHON, 'tilelang_capture.py', path, 'noinc-intervening')
    assert result.returncode == 0, result.stderr
    assert check_module(tilelang_snapshot_to_module(json.loads(result.stdout)), engine='maximal')['verdict'] == 'DATA_HAZARD'
