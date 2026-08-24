"""Shared helpers for merge_fwd_bwd stage golden."""

from __future__ import annotations

from typing import Optional

import torch

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


def pack_ag_hm(he: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Pack FLA [He | M] layout: [R, HV, K, V+K]."""
    ranks, hv, k_dim, v_dim = he.shape
    packed = he.new_empty(ranks, hv, k_dim, v_dim + k_dim)
    packed[..., :v_dim] = he
    packed[..., v_dim:] = m
    return packed


def make_inputs(
    *,
    ranks: int,
    hv: int,
    k_dim: int,
    v_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device | str = "cpu",
    has_h0: bool,
    state_v_first: bool,
) -> dict[str, torch.Tensor | None]:
    """Generate rank-major affine pieces.

    He: [R, HV, K, V]
    M:  [R, HV, K, K]
    h0: [HV, K, V] or [HV, V, K] when state_v_first
    """
    torch.manual_seed(seed)
    he = uniform((ranks, hv, k_dim, v_dim), dtype=dtype, device=device)
    m = uniform((ranks, hv, k_dim, k_dim), dtype=dtype, device=device)
    h0 = None
    if has_h0:
        if state_v_first:
            h0 = uniform((hv, v_dim, k_dim), dtype=dtype, device=device)
        else:
            h0 = uniform((hv, k_dim, v_dim), dtype=dtype, device=device)
    return {"he": he, "m": m, "h0": h0}


def save_tensor(path, value: Optional[torch.Tensor]) -> None:
    if value is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.detach().cpu().contiguous(), path)
