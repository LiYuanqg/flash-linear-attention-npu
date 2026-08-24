"""Run staged Triton (V0 then C0) and optionally the official FLA fused path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import head_to_seq, make_inputs, parse_dtype, save_tensor
from triton_kernels import run_fla, run_staged

STAGED_NAMES = ("g_corr", "gk", "qg", "kbg", "vb", "kg", "w", "u")
FLA_NAMES = ("gk", "qg", "kg", "w", "u")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/triton_staged"))
    parser.add_argument("--backend", choices=("staged", "fla"), default="staged")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hk", type=int, default=2)
    parser.add_argument("--hv", type=int, default=4)
    parser.add_argument("--k-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--use-gate-in-kernel", dest="use_gate", action="store_true", default=True)
    parser.add_argument("--no-use-gate-in-kernel", dest="use_gate", action="store_false")
    parser.add_argument("--safe-gate", dest="safe_gate", action="store_true", default=True)
    parser.add_argument("--no-safe-gate", dest="safe_gate", action="store_false")
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    dtype = parse_dtype(args.dtype)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cpu_inputs = make_inputs(
        batch=args.batch,
        tokens=args.tokens,
        hk=args.hk,
        hv=args.hv,
        k_dim=args.k_dim,
        v_dim=args.v_dim,
        chunk_size=args.chunk_size,
        dtype=dtype,
        seed=args.seed,
    )
    seq = {name: (value if name in {"A", "A_log", "dt_bias"} else head_to_seq(value)).to(device)
           for name, value in cpu_inputs.items()}

    kwargs = dict(
        chunk_size=args.chunk_size,
        use_gate=args.use_gate,
        safe_gate=args.safe_gate,
        lower_bound=args.lower_bound,
    )
    if args.backend == "staged":
        outputs = run_staged(
            seq["q"], seq["k"], seq["v"], seq["g"], seq["beta"], seq["A"],
            seq["A_log"], seq["dt_bias"], **kwargs,
        )
        names = STAGED_NAMES
    else:
        outputs = run_fla(
            seq["q"], seq["k"], seq["v"], seq["g"], seq["beta"], seq["A"],
            seq["A_log"], seq["dt_bias"], **kwargs,
        )
        names = FLA_NAMES

    if args.save_inputs:
        input_dir = args.out_dir / "inputs"
        for name, value in cpu_inputs.items():
            save_tensor(input_dir / f"{name}.pt", value if name in {"A", "A_log", "dt_bias"} else head_to_seq(value))

    for name in names:
        save_tensor(args.out_dir / f"{name}.pt", outputs[name])

    print(
        f"triton backend={args.backend} dtype={args.dtype} "
        f"use_gate={args.use_gate} safe_gate={args.safe_gate}; "
        f"saved {len(names)} tensors to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
