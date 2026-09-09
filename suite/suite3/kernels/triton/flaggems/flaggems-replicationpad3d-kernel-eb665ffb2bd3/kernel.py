# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import triton
import triton.language as tl

@triton.jit
def replicationpad3d_kernel(x_ptr, out_ptr, D_in, H_in, W_in, D_out, H_out, W_out, pad_l, pad_t, pad_f, stride_xn, stride_xc, stride_xd, stride_xh, stride_xw, stride_on, stride_oc, stride_od, stride_oh, stride_ow, C, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ncd = tl.program_id(2)
    d_idx = pid_ncd % D_out
    nc_idx = pid_ncd // D_out
    c_idx = nc_idx % C
    n_idx = nc_idx // C
    iz = d_idx - pad_f
    iz = tl.where(iz < 0, 0, iz)
    iz = tl.where(iz > D_in - 1, D_in - 1, iz)
    x_base_ptr = x_ptr + n_idx * stride_xn + c_idx * stride_xc + iz * stride_xd
    out_base_ptr = out_ptr + n_idx * stride_on + c_idx * stride_oc + d_idx * stride_od
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    iy = offs_h - pad_t
    iy = tl.where(iy < 0, 0, iy)
    iy = tl.where(iy > H_in - 1, H_in - 1, iy)
    ix = offs_w - pad_l
    ix = tl.where(ix < 0, 0, ix)
    ix = tl.where(ix > W_in - 1, W_in - 1, ix)
    x_offset = iy[:, None] * stride_xh + ix[None, :] * stride_xw
    out_offset = offs_h[:, None] * stride_oh + offs_w[None, :] * stride_ow
    mask = (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)
    vals = tl.load(x_base_ptr + x_offset, mask=mask)
    tl.store(out_base_ptr + out_offset, vals, mask=mask)

def _resolve_kernel(**pipeguard_bindings):
    return replicationpad3d_kernel

def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get('arguments', {}))
    build = case.get('build', {})
    arguments.update(build.get('resolver_kwargs', {}))
    return _resolve_kernel(**arguments)
