"""Run CPU merge_fwd_bwd stages and save snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import make_inputs, parse_dtype, save_tensor
from merge_golden import merge_cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cpu_fp32"))
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--hv", type=int, default=4)
    parser.add_argument("--k-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compute-precision", choices=("fp32", "fp64", "native"), default="fp32")
    parser.add_argument("--direction", choices=("fwd", "bwd"), default="fwd")
    parser.add_argument("--has-h0", action="store_true")
    parser.add_argument("--state-v-first", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.dtype)
    inputs = make_inputs(
        ranks=args.ranks,
        hv=args.hv,
        k_dim=args.k_dim,
        v_dim=args.v_dim,
        dtype=dtype,
        seed=args.seed,
        has_h0=args.has_h0,
        state_v_first=args.state_v_first,
    )
    if args.compute_precision == "native":
        work = dict(inputs)
    else:
        target = torch.float32 if args.compute_precision == "fp32" else torch.float64
        work = {name: None if value is None else value.to(target) for name, value in inputs.items()}

    outputs = merge_cpu(
        work["he"], work["m"], work["h0"],
        reverse=args.direction == "bwd",
        state_v_first=args.state_v_first,
    )
    if args.save_inputs:
        input_dir = args.out_dir / "inputs"
        for name, value in inputs.items():
            save_tensor(input_dir / f"{name}.pt", value)
    for name, value in outputs.items():
        save_tensor(args.out_dir / f"{name}.pt", value)
    print(
        f"cpu merge {args.direction} has_h0={args.has_h0} "
        f"state_v_first={args.state_v_first} compute={args.compute_precision}; "
        f"saved {len(outputs)} tensors to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
