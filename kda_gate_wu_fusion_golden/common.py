"""Shared input generation helpers for KDA gate+WU fusion golden."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

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


def normal(
    shape,
    *,
    mean: float,
    std: float,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device=device) * std + mean


def iter_chunks(
    tokens: int,
    chunk_size: int,
    cu_seqlens: Optional[Iterable[int]] = None,
) -> Iterable[tuple[int, int]]:
    """Yield (start, end) for each chunk, including a leftover tail and varlen spans."""
    if cu_seqlens is None:
        for start in range(0, tokens, chunk_size):
            yield start, min(start + chunk_size, tokens)
        return
    cu = [int(x) for x in cu_seqlens]
    if not cu or cu[0] != 0 or cu[-1] != tokens:
        raise ValueError(f"cu_seqlens must start at 0 and end at T={tokens}, got {cu[:1]}...{cu[-1:]}")
    for start, end in zip(cu[:-1], cu[1:]):
        if end < start:
            raise ValueError(f"cu_seqlens is not nondecreasing around {start}->{end}")
        for chunk_start in range(start, end, chunk_size):
            yield chunk_start, min(chunk_start + chunk_size, end)


def parse_normal_spec(spec: str) -> tuple[float, float]:
    match = re.fullmatch(r"normal\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)(?:,.*)?", spec.strip())
    if match is None:
        raise ValueError(f"unsupported input range spec: {spec}")
    return float(match.group(1)), float(match.group(2))


def load_case(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = data["config"]
    ctx = data.get("comparison_context") or {}
    return {
        "case_id": cfg.get("case_id", Path(path).stem),
        "batch": int(cfg["B"]),
        "tokens": int(cfg["T"]),
        "hk": int(cfg["HK"]),
        "hv": int(cfg["HV"]),
        "k_dim": int(cfg["K"]),
        "v_dim": int(cfg["V"]),
        "chunk_size": int(cfg["chunk_size"]),
        "dtype": str(cfg["dtype"]),
        "seed": int(cfg["seed"]),
        "safe_gate": bool(cfg["safe_gate"]),
        "use_gate": bool(cfg["use_gate_in_kernel"]),
        "lower_bound": float(cfg["lower_bound"]),
        "cu_seqlens": ctx.get("cu_seqlens"),
        "input_ranges": cfg["input_ranges"],
    }


def make_akk(
    batch: int,
    tokens: int,
    hv: int,
    chunk_size: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    cu_seqlens: Optional[Iterable[int]] = None,
    fill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Head-first Akk: dense BNSD `[B, HV, T, BT]`. Lower-triangular + I per chunk.

    Leftover tails shorter than `chunk_size` only fill `A[..., :length]`. Extra columns
    stay 0 so C0's BT×BT dot (padded time rows = 0) matches the length×length matmul.
    """
    akk = torch.zeros((batch, hv, tokens, chunk_size), dtype=dtype, device=device)
    if fill is None:
        fill = uniform((batch, hv, tokens, chunk_size), dtype=dtype, device=device)
    for start, end in iter_chunks(tokens, chunk_size, cu_seqlens):
        length = end - start
        lower = torch.tril(torch.ones(length, length, dtype=dtype, device=device))
        eye = torch.eye(length, dtype=dtype, device=device)
        block = fill[:, :, start:end, :length]
        akk[:, :, start:end, :length] = (
            block * lower.view(1, 1, length, length) + eye.view(1, 1, length, length)
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
    cu_seqlens: Optional[Iterable[int]] = None,
    input_ranges: Optional[dict[str, str]] = None,
) -> dict[str, torch.Tensor]:
    """Generate head-first BNSD tensors with a fixed RNG stream.

    q/k: [B, HK, T, K]
    v:   [B, HV, T, V]
    g:   [B, HV, T, K]  (key-wise)
    beta:[B, HV, T]
    A:   [B, HV, T, BT]
    A_log: [HV]
    dt_bias: [HV, K]

    Default fill is uniform(-1, 1). Case JSON uses `input_ranges` with normal().
    Leftover T and varlen `cu_seqlens` are allowed; T need not be a multiple of chunk_size.
    """
    if hv % hk != 0:
        raise ValueError("HV must be divisible by HK")
    if cu_seqlens is not None and batch != 1:
        raise ValueError("varlen golden only supports B=1")
    torch.manual_seed(seed)

    def fill(shape, spec_key: str, fill_dtype: torch.dtype) -> torch.Tensor:
        if input_ranges is None:
            return uniform(shape, dtype=fill_dtype, device=device)
        mean, std = parse_normal_spec(input_ranges[spec_key])
        return normal(shape, mean=mean, std=std, dtype=fill_dtype, device=device)

    tensors = {
        "q": fill((batch, hk, tokens, k_dim), "q_k_v", dtype),
        "k": fill((batch, hk, tokens, k_dim), "q_k_v", dtype),
        "v": fill((batch, hv, tokens, v_dim), "q_k_v", dtype),
        "g": fill((batch, hv, tokens, k_dim), "g_raw", torch.float32),
        "beta": fill((batch, hv, tokens), "beta_raw", dtype),
        "A_log": fill((hv,), "A_log", torch.float32),
        "dt_bias": fill((hv, k_dim), "dt_bias", torch.float32),
    }
    tensors["A"] = make_akk(
        batch, tokens, hv, chunk_size, dtype, device,
        cu_seqlens=cu_seqlens,
        fill=fill((batch, hv, tokens, chunk_size), "q_k_v", dtype),
    )
    return tensors


def head_to_seq(tensor: torch.Tensor) -> torch.Tensor:
    """BNSD `[B, H, T, ...]` -> BSND `[B, T, H, ...]`. Rank-3 beta: `[B, H, T]` -> `[B, T, H]`."""
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
