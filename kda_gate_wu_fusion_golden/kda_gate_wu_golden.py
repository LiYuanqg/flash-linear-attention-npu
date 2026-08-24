"""CPU formulas for the KDA gate-cumsum + recompute_w_u_fwd fusion.

Logical pipeline is two stages:

    V0 (vector): safe-gate correct g, then cumsum, then qg/kbg/vb/kg
    C0 (cube):   u = A @ vb,  w = A @ kbg

Shapes below are one (batch, HV head, chunk) tile unless noted:

    g/gk/qg/kbg/kg : [BT, K]
    beta           : [BT]
    vb/u           : [BT, V]
    A              : [BT, BT]
    q/k            : [BT, K] after GQA expand HK -> HV
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common import RCP_LN2, RCP_LN2_F64, expand_hk_to_hv


def activate_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    use_gate: bool,
    safe_gate: bool,
    lower_bound: float,
) -> torch.Tensor:
    """g is [B, HV, T, K], A_log is [HV], dt_bias is [HV, K]."""
    if not use_gate:
        return g
    x = g
    if dt_bias is not None:
        x = x + dt_bias.view(1, g.shape[1], 1, g.shape[-1])
    eig = torch.exp(A_log.view(1, g.shape[1], 1, 1))
    if safe_gate:
        return lower_bound * torch.sigmoid(eig * x)
    return -eig * F.softplus(x)


def chunk_cumsum(gate: torch.Tensor, chunk_size: int, scale: float) -> torch.Tensor:
    """Chunk-local prefix sum along T. gate/gk are [B, HV, T, K]."""
    batch, hv, tokens, k_dim = gate.shape
    gk = torch.empty_like(gate)
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        gk[:, :, start:end] = torch.cumsum(gate[:, :, start:end], dim=2) * scale
    return gk


def stage_v0(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    chunk_size: int,
    use_gate: bool,
    safe_gate: bool,
    lower_bound: float,
    rcp_ln2: float | None = None,
) -> dict[str, torch.Tensor]:
    """V0: safe-gate correct g, then kda_gate_chunk_cumsum(g_corr), then qg/kbg/vb/kg.

    Inputs are head-first:
      q/k [B, HK, T, K], v [B, HV, T, V], g [B, HV, T, K], beta [B, HV, T]
    Outputs stay head-first with HV on the head axis.
    """
    work_dtype = g.dtype
    q = q.to(work_dtype)
    k = k.to(work_dtype)
    v = v.to(work_dtype)
    beta = beta.to(work_dtype)
    A_log = A_log.to(work_dtype)
    if dt_bias is not None:
        dt_bias = dt_bias.to(work_dtype)

    hv = g.shape[1]
    tokens = g.shape[2]
    scale = RCP_LN2_F64 if work_dtype == torch.float64 else RCP_LN2
    if rcp_ln2 is not None:
        scale = rcp_ln2

    g_corr = activate_gate(
        g, A_log, dt_bias,
        use_gate=use_gate, safe_gate=safe_gate, lower_bound=lower_bound,
    )
    gk = chunk_cumsum(g_corr, chunk_size, scale)
    e2 = torch.exp2(gk)

    q_hv = expand_hk_to_hv(q, hv)
    k_hv = expand_hk_to_hv(k, hv)
    beta_k = beta.unsqueeze(-1)

    qg = q_hv * e2
    kbg = k_hv * beta_k * e2
    vb = v * beta_k

    kg = torch.empty_like(k_hv)
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        gk_last = gk[:, :, end - 1 : end, :]
        kg[:, :, start:end] = k_hv[:, :, start:end] * torch.exp2(gk_last - gk[:, :, start:end])

    return {"g_corr": g_corr, "gk": gk, "qg": qg, "kbg": kbg, "vb": vb, "kg": kg}


def stage_c0(
    A: torch.Tensor,
    kbg: torch.Tensor,
    vb: torch.Tensor,
    *,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """C0: u = A @ vb, w = A @ kbg. A is sequence-major [B, T, HV, BT].

    kbg is [B, HV, T, K], vb is [B, HV, T, V]. Outputs are head-first.
    """
    batch, hv, tokens, k_dim = kbg.shape
    v_dim = vb.shape[-1]
    work_dtype = kbg.dtype
    A = A.to(work_dtype)
    w = torch.empty((batch, hv, tokens, k_dim), dtype=work_dtype, device=kbg.device)
    u = torch.empty((batch, hv, tokens, v_dim), dtype=work_dtype, device=vb.device)
    for batch_id in range(batch):
        for hv_id in range(hv):
            for start in range(0, tokens, chunk_size):
                end = min(start + chunk_size, tokens)
                length = end - start
                a = A[batch_id, start:end, hv_id, :length]
                w[batch_id, hv_id, start:end] = a @ kbg[batch_id, hv_id, start:end]
                u[batch_id, hv_id, start:end] = a @ vb[batch_id, hv_id, start:end]
    return {"w": w, "u": u}


def fused_cpu(inputs: dict[str, torch.Tensor], **v0_kwargs) -> dict[str, torch.Tensor]:
    v0 = stage_v0(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        inputs["A_log"], inputs.get("dt_bias"),
        **v0_kwargs,
    )
    c0 = stage_c0(inputs["A"], v0["kbg"], v0["vb"], chunk_size=v0_kwargs["chunk_size"])
    return {**v0, **c0}
