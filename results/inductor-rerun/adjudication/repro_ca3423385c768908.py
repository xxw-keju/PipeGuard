"""Run ca3423385c768908 with out_ptr0 pre-filled with NaN; any NaN or
mismatch in out_ptr3 means a thread read out_ptr0 before another warp's
store reached it. Variant 'barrier' inserts tl.debug_barrier() between the
store loop and the reload loop."""
import importlib.util, sys, tempfile, torch
from pathlib import Path
src = Path(sys.argv[1]).read_text()
variant = sys.argv[2]
runs = int(sys.argv[3])
if variant == "barrier":
    src = src.replace("    tmp12 = tl.sum(_tmp12, 1)[:, None]\n", "    tl.debug_barrier()\n    tmp12 = tl.sum(_tmp12, 1)[:, None]\n", 1)
    assert "debug_barrier" in src
path = Path(tempfile.mkdtemp()) / f"k_{variant}.py"
path.write_text(src)
spec = importlib.util.spec_from_file_location(path.stem, path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn = m.triton_red_fused_native_layer_norm_backward_permute_86
fn = getattr(fn, "fn", fn) if not hasattr(fn, "__getitem__") else fn
torch.manual_seed(0)
dev = "cuda"
N = 9408 * 96
h = lambda n: torch.randn(n, device=dev, dtype=torch.float16)
io, a0, a1, a2, a3, a4 = h(N), h(N), h(N), h(N), h(N), h(96)
a5 = torch.randn(9408, device=dev); a6 = torch.rand(9408, device=dev) + 0.5
out0 = torch.empty(N, device=dev); out3 = torch.empty(N, device=dev, dtype=torch.float16)
def launch(fill):
    out0.fill_(fill); out3.zero_()
    fn[(1176,)](io, a0, a1, a2, a3, a4, a5, a6, out0, out3, 9408, 96, XBLOCK=8, RBLOCK=128, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    return out3.clone()
launch(0.0)
ref = None
# reference: out_ptr0 pre-filled with its own final contents, so a stale read is harmless
final0 = out0.clone()
out0.copy_(final0); out3.zero_()
fn[(1176,)](io, a0, a1, a2, a3, a4, a5, a6, out0, out3, 9408, 96, XBLOCK=8, RBLOCK=128, num_warps=4, num_stages=1)
torch.cuda.synchronize(); ref = out3.clone()
bad_runs = 0; bad_elems = 0
for _ in range(runs):
    got = launch(float("nan"))
    diff = (got != ref) | torch.isnan(got)
    count = int(diff.sum())
    if count:
        bad_runs += 1; bad_elems += count
print(f"variant={variant} runs={runs} bad_runs={bad_runs} bad_elements={bad_elems}")
