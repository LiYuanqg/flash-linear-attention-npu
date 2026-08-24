"""Run staged Triton merge_fwd_bwd, or official FLA CP merge."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import make_inputs, parse_dtype, save_tensor
from triton_kernels import run_fla, run_staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/triton_staged"))
    parser.add_argument("--backend", choices=("staged", "fla"), default="staged")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--hv", type=int, default=4)
    parser.add_argument("--k-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--direction", choices=("fwd", "bwd"), default="fwd")
    parser.add_argument("--has-h0", action="store_true")
    parser.add_argument("--state-v-first", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    dtype = parse_dtype(args.dtype)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cpu_inputs = make_inputs(
        ranks=args.ranks,
        hv=args.hv,
        k_dim=args.k_dim,
        v_dim=args.v_dim,
        dtype=dtype,
        seed=args.seed,
        has_h0=args.has_h0,
        state_v_first=args.state_v_first,
    )
    gpu = {name: None if value is None else value.to(device) for name, value in cpu_inputs.items()}
    kwargs = dict(reverse=args.direction == "bwd", state_v_first=args.state_v_first)
    if args.backend == "staged":
        outputs = run_staged(gpu["he"], gpu["m"], gpu["h0"], **kwargs)
    else:
        outputs = run_fla(gpu["he"], gpu["m"], gpu["h0"], **kwargs)

    if args.save_inputs:
        input_dir = args.out_dir / "inputs"
        for name, value in cpu_inputs.items():
            save_tensor(input_dir / f"{name}.pt", value)
    for name, value in outputs.items():
        save_tensor(args.out_dir / f"{name}.pt", value)
    print(
        f"triton merge backend={args.backend} {args.direction} "
        f"has_h0={args.has_h0} state_v_first={args.state_v_first}; "
        f"saved {len(outputs)} tensors to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
