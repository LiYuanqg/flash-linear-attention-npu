"""Staged Triton kernels matching the NPU V0 -> C0 fusion schedule.

Inner tensors are head-first BNSD. V0 is AIV-equivalent: safe-gate correct g,
then chunk cumsum, then qg/kbg/vb/kg. C0 is AIC-equivalent: u = A @ vb and
w = A @ kbg with A reused. Official FLA remains sequence-major at its own API.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp, exp2
from fla.ops.utils.softplus import softplus

from common import RCP_LN2


@triton.jit(do_not_specialize=["T"])
def v0_kernel(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    g_corr,
    gk,
    qg,
    kbg,
    vb,
    kg,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE,
    LOWER_BOUND,
    USE_GATE: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = beta + (i_b * HV + i_hv) * T + o_t
    b_b = tl.load(p_b, mask=m_t, other=0.0).to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + ((i_b * HV + i_hv) * T + o_t)[:, None] * V + o_v[None, :]
        p_vb = vb + ((i_b * HV + i_hv) * T + o_t)[:, None] * V + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
        tl.store(p_vb, (b_v * b_b[:, None]).to(p_vb.dtype.element_ty), mask=m_v)

    last_local = tl.minimum(BT, T - i_t * BT) - 1
    b_A = 0.0
    if USE_GATE:
        b_A = tl.load(A_log + i_hv).to(tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]

        p_g = g + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]
        b_s = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
        if USE_GATE:
            if HAS_BIAS:
                b_bias = tl.load(dt_bias + i_hv * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_s = b_s + b_bias[None, :]
            if SAFE_GATE:
                b_gate = LOWER_BOUND * tl.sigmoid(exp(b_A) * b_s)
            else:
                b_gate = -exp(b_A) * softplus(b_s)
        else:
            b_gate = b_s

        b_gk = tl.cumsum(b_gate, axis=0) * SCALE
        p_gk = gk + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]
        tl.store(p_gk, b_gk.to(p_gk.dtype.element_ty), mask=m_tk)

        b_e2 = exp2(b_gk)
        p_q = q + ((i_b * H + i_h) * T + o_t)[:, None] * K + o_k[None, :]
        p_k = k + ((i_b * H + i_h) * T + o_t)[:, None] * K + o_k[None, :]
        p_qg = qg + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]
        p_kbg = kbg + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]
        p_kg = kg + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]

        b_q = tl.load(p_q, mask=m_tk, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_tk, other=0.0).to(tl.float32)
        tl.store(p_qg, (b_q * b_e2).to(p_qg.dtype.element_ty), mask=m_tk)
        tl.store(p_kbg, (b_k * b_b[:, None] * b_e2).to(p_kbg.dtype.element_ty), mask=m_tk)

        b_gn = tl.sum(tl.where((tl.arange(0, BT) == last_local)[:, None], b_gk, 0.0), axis=0)
        b_kg = b_k * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_gk), 0.0)
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_tk)
        p_g_corr = g_corr + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]
        tl.store(p_g_corr, b_gate.to(p_g_corr.dtype.element_ty), mask=m_tk)


@triton.jit(do_not_specialize=["T"])
def c0_kernel(
    A,
    kbg,
    vb,
    w,
    u,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // HV
    i_hv = i_bh % HV

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_A = tl.arange(0, BT)
    m_A = m_t[:, None] & (o_A[None, :] < tl.minimum(BT, T - i_t * BT))
    p_A = A + ((i_b * HV + i_hv) * T + o_t)[:, None] * BT + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_vb = vb + ((i_b * HV + i_hv) * T + o_t)[:, None] * V + o_v[None, :]
        p_u = u + ((i_b * HV + i_hv) * T + o_t)[:, None] * V + o_v[None, :]
        b_vb = tl.load(p_vb, mask=m_v, other=0.0)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_kbg = kbg + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]
        p_w = w + ((i_b * HV + i_hv) * T + o_t)[:, None] * K + o_k[None, :]
        b_kbg = tl.load(p_kbg, mask=m_k, other=0.0)
        b_w = tl.dot(b_A, b_kbg)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


def _slice_time(tensor: torch.Tensor, start: int, end: int, time_dim: int) -> torch.Tensor:
    sl = [slice(None)] * tensor.ndim
    sl[time_dim] = slice(start, end)
    return tensor[tuple(sl)]


def run_staged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    chunk_size: int,
    use_gate: bool,
    safe_gate: bool,
    lower_bound: float,
    cu_seqlens: list[int] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """All tensors are head-first BNSD: q/k `[B,HK,T,K]`, v `[B,HV,T,V]`, g `[B,HV,T,K]`, A `[B,HV,T,BT]`."""
    if cu_seqlens is not None:
        cu = [int(x) for x in cu_seqlens]
        if q.shape[0] != 1:
            raise ValueError("varlen staged path requires B=1")
        names = ("g_corr", "gk", "qg", "kbg", "vb", "kg", "w", "u")
        outputs = {name: None for name in names}
        for start, end in zip(cu[:-1], cu[1:]):
            part = run_staged(
                _slice_time(q, start, end, 2), _slice_time(k, start, end, 2),
                _slice_time(v, start, end, 2), _slice_time(g, start, end, 2),
                _slice_time(beta, start, end, 2), _slice_time(A, start, end, 2),
                A_log, dt_bias,
                chunk_size=chunk_size, use_gate=use_gate,
                safe_gate=safe_gate, lower_bound=lower_bound,
            )
            for name in names:
                if outputs[name] is None:
                    full_shape = list(part[name].shape)
                    full_shape[2] = q.shape[2]
                    outputs[name] = torch.empty(full_shape, device=part[name].device, dtype=part[name].dtype)
                outputs[name][:, :, start:end] = part[name]
        return outputs

    B, H, T, K = k.shape
    HV, V = v.shape[1], v.shape[-1]
    BT = chunk_size
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    NT = triton.cdiv(T, BT)

    gk = torch.empty(B, HV, T, K, device=g.device, dtype=torch.float32)
    g_corr = torch.empty(B, HV, T, K, device=g.device, dtype=torch.float32)
    qg = torch.empty(B, HV, T, K, device=q.device, dtype=q.dtype)
    kbg = torch.empty(B, HV, T, K, device=k.device, dtype=k.dtype)
    vb = torch.empty(B, HV, T, V, device=v.device, dtype=v.dtype)
    kg = torch.empty(B, HV, T, K, device=k.device, dtype=k.dtype)
    w = torch.empty(B, HV, T, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)

    dummy_bias = g if dt_bias is None else dt_bias
    v0_kernel[(NT, B * HV)](
        q, k, v, g, beta, A_log, dummy_bias,
        g_corr, gk, qg, kbg, vb, kg,
        T, H, HV, K, V, BT, BK, BV,
        RCP_LN2, lower_bound,
        USE_GATE=use_gate,
        SAFE_GATE=safe_gate,
        HAS_BIAS=dt_bias is not None and use_gate,
        num_warps=4,
        num_stages=2,
    )
    c0_kernel[(NT, B * HV)](
        A, kbg, vb, w, u,
        T, HV, K, V, BT, BK, BV,
        num_warps=4,
        num_stages=2,
    )
    return {"g_corr": g_corr, "gk": gk, "qg": qg, "kbg": kbg, "vb": vb, "kg": kg, "w": w, "u": u}


def run_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    chunk_size: int,
    use_gate: bool,
    safe_gate: bool,
    lower_bound: float,
    cu_seqlens: list[int] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Official FLA path: kda_gate_chunk_cumsum / chunk_local_cumsum + recompute_w_u_fwd.

    FLA public tensors are sequence-major BSND. Convert BNSD -> BSND at the call site.
    Does not expose workspace kbg/vb. Varlen is run per sequence so leftover
    chunks and packed T>GPU memory still fit; each call is the official fused kernels.
    """
    from fla.ops.kda.gate import kda_gate_chunk_cumsum
    from fla.ops.kda.wy_fast import recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2 as FLA_RCP_LN2

    if cu_seqlens is not None:
        cu = [int(x) for x in cu_seqlens]
        if q.shape[0] != 1:
            raise ValueError("varlen FLA path requires B=1")
        names = ("gk", "qg", "kg", "w", "u")
        outputs = {}
        for start, end in zip(cu[:-1], cu[1:]):
            part = run_fla(
                _slice_time(q, start, end, 1), _slice_time(k, start, end, 1),
                _slice_time(v, start, end, 1), _slice_time(g, start, end, 1),
                _slice_time(beta, start, end, 1), _slice_time(A, start, end, 1),
                A_log, dt_bias,
                chunk_size=chunk_size, use_gate=use_gate,
                safe_gate=safe_gate, lower_bound=lower_bound,
            )
            for name in names:
                if name not in outputs:
                    full_shape = list(part[name].shape)
                    full_shape[1] = q.shape[1]
                    outputs[name] = torch.empty(full_shape, device=part[name].device, dtype=part[name].dtype)
                outputs[name][:, start:end] = part[name]
        return outputs

    cu = None
    if use_gate:
        gk = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=FLA_RCP_LN2,
            chunk_size=chunk_size,
            lower_bound=lower_bound if safe_gate else None,
            cu_seqlens=cu,
        )
    else:
        gk = chunk_local_cumsum(
            g=g,
            scale=FLA_RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu,
        )
    w, u, qg, kg = recompute_w_u_fwd(
        q=q, k=k, v=v, beta=beta, A=A, gk=gk,
        cu_seqlens=cu,
    )
    return {"gk": gk, "qg": qg, "kg": kg, "w": w, "u": u}
