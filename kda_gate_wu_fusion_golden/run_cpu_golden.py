"""Run CPU V0 -> C0 golden and save sequence-major tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import head_to_seq, make_inputs, parse_dtype, save_tensor
from kda_gate_wu_golden import fused_cpu

OUTPUT_NAMES = ("g_corr", "gk", "qg", "kbg", "vb", "kg", "w", "u")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cpu_fp32"))
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
    parser.add_argument(
        "--compute-precision",
        choices=("fp32", "fp64", "native"),
        default="fp32",
        help="fp32/fp64 compute then save that dtype; native keeps generated dtype",
    )
    parser.add_argument("--use-gate-in-kernel", dest="use_gate", action="store_true", default=True)
    parser.add_argument("--no-use-gate-in-kernel", dest="use_gate", action="store_false")
    parser.add_argument("--safe-gate", dest="safe_gate", action="store_true", default=True)
    parser.add_argument("--no-safe-gate", dest="safe_gate", action="store_false")
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.dtype)
    inputs = make_inputs(
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
    if args.compute_precision == "native":
        work_inputs = dict(inputs)
    else:
        target = torch.float32 if args.compute_precision == "fp32" else torch.float64
        work_inputs = {name: value.to(target) for name, value in inputs.items()}

    outputs = fused_cpu(
        work_inputs,
        chunk_size=args.chunk_size,
        use_gate=args.use_gate,
        safe_gate=args.safe_gate,
        lower_bound=args.lower_bound,
    )

    if args.save_inputs:
        input_dir = args.out_dir / "inputs"
        for name, value in inputs.items():
            save_tensor(input_dir / f"{name}.pt", head_to_seq(value) if name != "A" else value)

    for name in OUTPUT_NAMES:
        save_tensor(args.out_dir / f"{name}.pt", head_to_seq(outputs[name]))

    print(
        f"cpu golden compute_precision={args.compute_precision} "
        f"use_gate={args.use_gate} safe_gate={args.safe_gate}; "
        f"saved {len(OUTPUT_NAMES)} tensors to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
