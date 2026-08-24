"""Print CPU vs Triton max/mean abs, rel, and cosine for saved .pt files."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def stats(real: torch.Tensor, expect: torch.Tensor) -> dict[str, float]:
    real = real.detach().float().reshape(-1)
    expect = expect.detach().float().reshape(-1)
    if real.numel() != expect.numel():
        raise ValueError(f"numel mismatch: real={real.numel()} expect={expect.numel()}")
    diff = (expect - real).abs()
    denom = real.abs().clamp_min(1e-6)
    rel = diff / denom
    cosine = torch.nn.functional.cosine_similarity(real[None], expect[None]).item()
    return {
        "abs_mean": diff.mean().item(),
        "abs_max": diff.max().item(),
        "rel_mean": rel.mean().item(),
        "rel_p99": rel.kthvalue(max(int(rel.numel() * 0.99), 1)).values.item(),
        "cosine": cosine,
        "finite_real": torch.isfinite(real).all().item(),
        "finite_expect": torch.isfinite(expect).all().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--expect-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="*", default=["g_corr", "gk", "qg", "kbg", "vb", "kg", "w", "u"])
    args = parser.parse_args()
    print(f"{'name':<8} {'abs_mean':>12} {'abs_max':>12} {'rel_mean':>12} {'rel_p99':>12} {'cosine':>12} finite")
    for name in args.names:
        real_path = args.real_dir / f"{name}.pt"
        expect_path = args.expect_dir / f"{name}.pt"
        if not real_path.exists() or not expect_path.exists():
            print(f"{name:<8} SKIP")
            continue
        result = stats(torch.load(real_path, map_location="cpu", weights_only=True),
                       torch.load(expect_path, map_location="cpu", weights_only=True))
        print(
            f"{name:<8} {result['abs_mean']:12.4e} {result['abs_max']:12.4e} "
            f"{result['rel_mean']:12.4e} {result['rel_p99']:12.4e} {result['cosine']:12.8f} "
            f"{result['finite_real']}/{result['finite_expect']}"
        )


if __name__ == "__main__":
    main()
