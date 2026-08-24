"""Staged Triton kernels matching the NPU merge_fwd_bwd schedule.

Cube stages do only `h = M @ h`. Vector stages do only `h = h + He`.
Those two stages repeat for each remaining rank. Adjacent cube updates
cannot share one AIC stage because the next MM needs this stage's h.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from common import pack_ag_hm


@triton.jit
def v_copy_kernel(
    he,
    h,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_h = tl.program_id(1)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    p_he = he + i_h * K * V + o_k[:, None] * V + o_v[None, :]
    b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    if STATE_V_FIRST:
        p_h = h + i_h * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_h, tl.trans(b_he).to(p_h.dtype.element_ty), mask=m_v[:, None] & m_k[None, :])
    else:
        p_h = h + i_h * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_he.to(p_h.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])


@triton.jit
def c_mm_kernel(
    m,
    h,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_h = tl.program_id(1)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V

    p_m = m + i_h * K * K + o_k[:, None] * K + o_k[None, :]
    b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)

    if STATE_V_FIRST:
        p_h = h + i_h * V * K + o_v[:, None] * K + o_k[None, :]
        b_h = tl.load(p_h, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_h = tl.dot(b_h, tl.trans(b_m))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_v[:, None] & m_k[None, :])
    else:
        p_h = h + i_h * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_h = tl.dot(b_m, b_h)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])


@triton.jit
def v_add_kernel(
    he,
    h,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_h = tl.program_id(1)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    p_he = he + i_h * K * V + o_k[:, None] * V + o_v[None, :]
    b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    if STATE_V_FIRST:
        p_h = h + i_h * V * K + o_v[:, None] * K + o_k[None, :]
        b_h = tl.load(p_h, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_h = b_h + tl.trans(b_he)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_v[:, None] & m_k[None, :])
    else:
        p_h = h + i_h * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_h = b_h + b_he
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_k[:, None] & m_v[None, :])


def _empty_h(he: torch.Tensor, state_v_first: bool) -> torch.Tensor:
    _, hv, k_dim, v_dim = he.shape
    if state_v_first:
        return torch.zeros(hv, v_dim, k_dim, dtype=torch.float32, device=he.device)
    return torch.zeros(hv, k_dim, v_dim, dtype=torch.float32, device=he.device)


def run_staged(
    he: torch.Tensor,
    m: torch.Tensor,
    h0: torch.Tensor | None,
    *,
    reverse: bool,
    state_v_first: bool,
) -> dict[str, torch.Tensor]:
    """he/m are [R, HV, K, V] / [R, HV, K, K] on GPU."""
    ranks, hv, k_dim, v_dim = he.shape
    bk = triton.next_power_of_2(k_dim)
    bv = 64 if v_dim >= 64 else triton.next_power_of_2(v_dim)
    grid = (triton.cdiv(v_dim, bv), hv)
    order = list(range(ranks - 1, -1, -1) if reverse else range(ranks))
    stages: dict[str, torch.Tensor] = {}
    launch = dict(num_warps=4, num_stages=2)
    if h0 is None:
        h = _empty_h(he, state_v_first)
        v_copy_kernel[grid](
            he[order[0]], h, hv, k_dim, v_dim, bk, bv, state_v_first, **launch,
        )
        stages["h_v0"] = h.clone()
        start = 1
        pair_id = 1
    else:
        h = h0.to(torch.float32).contiguous()
        start = 0
        pair_id = 0
    for rank in order[start:]:
        c_mm_kernel[grid](
            m[rank], h, hv, k_dim, v_dim, bk, bv, state_v_first, **launch,
        )
        stages[f"h_c{pair_id}"] = h.clone()
        v_add_kernel[grid](
            he[rank], h, hv, k_dim, v_dim, bk, bv, state_v_first, **launch,
        )
        stages[f"h_v{pair_id}"] = h.clone()
        pair_id += 1
    stages["h"] = h
    return stages


def run_fla(
    he: torch.Tensor,
    m: torch.Tensor,
    h0: torch.Tensor | None,
    *,
    reverse: bool,
    state_v_first: bool,
) -> dict[str, torch.Tensor]:
    """Official FLA CP merge. h0 is not used in CP mode; reverse flips rank order."""
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    if h0 is not None:
        raise ValueError("FLA CP merge_fwd_bwd does not take h0; use staged backend")
    work_he = he.flip(0).contiguous() if reverse else he.contiguous()
    work_m = m.flip(0).contiguous() if reverse else m.contiguous()
    ranks, hv, k_dim, v_dim = work_he.shape
    ag_hm = pack_ag_hm(work_he.to(torch.float32), work_m.to(torch.float32))
    h = _empty_h(work_he, state_v_first)
    bk = triton.next_power_of_2(k_dim)

    def grid(meta):
        return (triton.cdiv(v_dim, meta["BV"]), hv)

    merge_fwd_bwd_kernel[grid](
        h=h,
        ag_hm=ag_hm,
        pre_or_post_num_ranks=ranks,
        rank=ranks,
        seq_offsets=None,
        init_offsets=None,
        h0_seq_ids=None,
        h0=None,
        HV=hv,
        K=k_dim,
        V=v_dim,
        BK=bk,
        FORWARD=True,
        INTRACARD_MODE=False,
        NUM_SEQ_ENTRIES=0,
        STATE_V_FIRST=state_v_first,
    )
    return {"h": h}
