"""Shared input generation and layout helpers for KDA gate+WU fusion golden."""

from __future__ import annotations

import math
from typing import Optional

import torch

# FLA FP32 reciprocal of ln(2). Keep this for native/fp32 so CPU matches Triton.
RCP_LN2 = 1.4426950216
RCP_LN2_F64 = 1.0 / math.log(2.0)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_dtype(name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def uniform(shape, *, dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.rand(*shape, dtype=dtype, device=device) * 2 - 1


def make_akk(
    batch: int,
    tokens: int,
    hv: int,
    chunk_size: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sequence-major Akk-like matrix: [B, T, HV, BT], lower-triangular + I per chunk."""
    akk = uniform((batch, tokens, hv, chunk_size), dtype=dtype, device=device)
    lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=dtype, device=device))
    eye = torch.eye(chunk_size, dtype=dtype, device=device)
    for start in range(0, tokens, chunk_size):
        block = akk[:, start : start + chunk_size]
        akk[:, start : start + chunk_size] = (
            block * lower.view(1, chunk_size, 1, chunk_size)
            + eye.view(1, chunk_size, 1, chunk_size)
        )
    return akk


def make_inputs(
    *,
    batch: int,
    tokens: int,
    hk: int,
    hv: int,
    k_dim: int,
    v_dim: int,
    chunk_size: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Generate head-first tensors with a fixed RNG stream.

    q/k: [B, HK, T, K]
    v:   [B, HV, T, V]
    g:   [B, HV, T, K]  (key-wise)
    beta:[B, HV, T]
    A_log: [HV]
    dt_bias: [HV, K]
    A: [B, T, HV, BT] sequence-major, matching FLA recompute_w_u_fwd
    """
    if hv % hk != 0:
        raise ValueError("HV must be divisible by HK")
    if tokens % chunk_size != 0:
        raise ValueError("tokens must be divisible by chunk-size for this golden")
    torch.manual_seed(seed)
    return {
        "q": uniform((batch, hk, tokens, k_dim), dtype=dtype, device=device),
        "k": uniform((batch, hk, tokens, k_dim), dtype=dtype, device=device),
        "v": uniform((batch, hv, tokens, v_dim), dtype=dtype, device=device),
        "g": uniform((batch, hv, tokens, k_dim), dtype=torch.float32, device=device),
        "beta": uniform((batch, hv, tokens), dtype=dtype, device=device),
        "A_log": uniform((hv,), dtype=torch.float32, device=device),
        "dt_bias": uniform((hv, k_dim), dtype=torch.float32, device=device),
        "A": make_akk(batch, tokens, hv, chunk_size, dtype, device),
    }


def head_to_seq(tensor: torch.Tensor) -> torch.Tensor:
    """[B, H, T, ...] -> [B, T, H, ...]. Rank-3 beta: [B, H, T] -> [B, T, H]."""
    if tensor.ndim == 4:
        return tensor.permute(0, 2, 1, 3).contiguous()
    if tensor.ndim == 3:
        return tensor.permute(0, 2, 1).contiguous()
    return tensor


def seq_to_head(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.permute(0, 2, 1, 3).contiguous()
    if tensor.ndim == 3:
        return tensor.permute(0, 2, 1).contiguous()
    return tensor


def expand_hk_to_hv(tensor: torch.Tensor, hv: int) -> torch.Tensor:
    """Repeat HK heads along the head axis to HV. Accepts head-first [B, HK, T, K]."""
    hk = tensor.shape[1]
    group = hv // hk
    if group == 1:
        return tensor
    return tensor.repeat_interleave(group, dim=1)


def save_tensor(path, value: Optional[torch.Tensor]) -> None:
    if value is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.detach().cpu().contiguous(), path)
