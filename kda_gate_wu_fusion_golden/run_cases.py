"""Run kda_gate_chunk_cumsum + recompute_w_u_fwd golden on cases/*.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from common import head_to_seq, load_case, make_inputs, parse_dtype
from compare_stats import stats
from kda_gate_wu_golden import fused_cpu

STAGED_NAMES = ("g_corr", "gk", "qg", "kbg", "vb", "kg", "w", "u")
FLA_NAMES = ("gk", "qg", "kg", "w", "u")


def _to_seq(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value if name in {"A", "A_log", "dt_bias"} else head_to_seq(value)
        for name, value in inputs.items()
    }


def _move(seq: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in seq.items()}


def _print_table(title: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    print(title)
    print(f"{'name':<8} {'abs_mean':>12} {'abs_max':>12} {'rel_p99':>12} {'cosine':>12} finite")
    for name, result in rows:
        print(
            f"{name:<8} {result['abs_mean']:12.4e} {result['abs_max']:12.4e} "
            f"{result['rel_p99']:12.4e} {result['cosine']:12.8f} "
            f"{result['finite_real']}/{result['finite_expect']}"
        )


def _slice_seq_inputs(seq: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    out = {}
    for name, value in seq.items():
        if name in {"A_log", "dt_bias"}:
            out[name] = value
        else:
            out[name] = value[:, start:end]
    return out


def _merge_rows(acc: dict[str, dict], part: list[tuple[str, dict[str, float]]], numel: int) -> None:
    for name, result in part:
        slot = acc.setdefault(
            name,
            {"abs_sum": 0.0, "n": 0, "abs_max": 0.0, "rel_p99": 0.0, "cosine": 1.0,
             "finite_real": True, "finite_expect": True},
        )
        slot["abs_sum"] += result["abs_mean"] * numel
        slot["n"] += numel
        slot["abs_max"] = max(slot["abs_max"], result["abs_max"])
        slot["rel_p99"] = max(slot["rel_p99"], result["rel_p99"])
        slot["cosine"] = min(slot["cosine"], result["cosine"])
        slot["finite_real"] = slot["finite_real"] and result["finite_real"]
        slot["finite_expect"] = slot["finite_expect"] and result["finite_expect"]


def _finalize_rows(acc: dict[str, dict]) -> list[tuple[str, dict[str, float]]]:
    rows = []
    for name, slot in acc.items():
        rows.append((name, {
            "abs_mean": slot["abs_sum"] / max(slot["n"], 1),
            "abs_max": slot["abs_max"],
            "rel_p99": slot["rel_p99"],
            "cosine": slot["cosine"],
            "finite_real": slot["finite_real"],
            "finite_expect": slot["finite_expect"],
        }))
    return rows


def _compare(real: dict[str, torch.Tensor], expect: dict[str, torch.Tensor], names: tuple[str, ...]):
    rows = []
    for name in names:
        if name not in real or name not in expect:
            continue
        rows.append((name, stats(real[name].cpu(), expect[name].cpu())))
    return rows


def _append_rows(lines: list[str], title: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    _print_table(title, rows)
    lines.append(title)
    for name, result in rows:
        lines.append(
            f"{name} abs_mean={result['abs_mean']:.4e} abs_max={result['abs_max']:.4e} "
            f"rel_p99={result['rel_p99']:.4e} cosine={result['cosine']:.8f}"
        )


def _run_pair(runner, seq, kwargs, cpu_seq, names, cu, device):
    if not cu:
        got = runner(
            seq["q"], seq["k"], seq["v"], seq["g"], seq["beta"], seq["A"],
            seq["A_log"], seq["dt_bias"], **kwargs,
        )
        return _compare(cpu_seq, got, names), got

    acc: dict[str, dict] = {}
    dense_kwargs = dict(kwargs)
    dense_kwargs["cu_seqlens"] = None
    for start, end in zip(cu[:-1], cu[1:]):
        if end <= start:
            continue
        part_in = _move(_slice_seq_inputs(seq, start, end), device)
        got = runner(
            part_in["q"], part_in["k"], part_in["v"], part_in["g"], part_in["beta"],
            part_in["A"], part_in["A_log"], part_in["dt_bias"], **dense_kwargs,
        )
        cpu_part = {name: cpu_seq[name][:, start:end] for name in names if name in cpu_seq}
        _merge_rows(acc, _compare(cpu_part, got, names), (end - start))
        del got, part_in
        torch.cuda.empty_cache()
    return _finalize_rows(acc), None


def _run_one_dense(inputs, case, device, skip_fla, lines):
    work = {name: value.to(torch.float32) for name, value in inputs.items()}
    cpu = fused_cpu(
        work,
        chunk_size=case["chunk_size"],
        use_gate=case["use_gate"],
        safe_gate=case["safe_gate"],
        lower_bound=case["lower_bound"],
        cu_seqlens=None,
    )
    cpu_seq = {name: head_to_seq(value) for name, value in cpu.items()}
    del work, cpu
    if device.type != "cuda":
        print("skip GPU: CUDA not available")
        lines.append("GPU skipped")
        return "\n".join(lines) + "\n"

    from triton_kernels import run_fla, run_staged

    seq = _move(_to_seq(inputs), device)
    del inputs
    kwargs = dict(
        chunk_size=case["chunk_size"],
        use_gate=case["use_gate"],
        safe_gate=case["safe_gate"],
        lower_bound=case["lower_bound"],
        cu_seqlens=None,
    )
    staged_rows, staged = _run_pair(run_staged, seq, kwargs, cpu_seq, STAGED_NAMES, None, device)
    _append_rows(lines, "CPU FP32 vs staged Triton", staged_rows)
    if skip_fla:
        return "\n".join(lines) + "\n"
    fla_rows, fla = _run_pair(run_fla, seq, kwargs, cpu_seq, FLA_NAMES, None, device)
    _append_rows(lines, "CPU FP32 vs FLA kda_gate_chunk_cumsum+recompute_w_u_fwd", fla_rows)
    if staged is not None and fla is not None:
        _append_rows(lines, "staged Triton vs FLA", _compare(staged, fla, FLA_NAMES))
    del staged, fla, seq, cpu_seq
    torch.cuda.empty_cache()
    return "\n".join(lines) + "\n"


def _run_one_varlen(case, device, skip_fla, lines):
    """Per-sequence to keep peak memory at max(seq_len), not packed T."""
    cu = case["cu_seqlens"]
    if device.type != "cuda":
        print("skip GPU: CUDA not available")
        lines.append("GPU skipped")
        return "\n".join(lines) + "\n"

    from triton_kernels import run_fla, run_staged

    staged_acc: dict[str, dict] = {}
    fla_acc: dict[str, dict] = {}
    vs_acc: dict[str, dict] = {}
    kwargs = dict(
        chunk_size=case["chunk_size"],
        use_gate=case["use_gate"],
        safe_gate=case["safe_gate"],
        lower_bound=case["lower_bound"],
        cu_seqlens=None,
    )
    n_seq = 0
    for seq_id, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        length = end - start
        if length <= 0:
            continue
        n_seq += 1
        inputs = make_inputs(
            batch=1,
            tokens=length,
            hk=case["hk"],
            hv=case["hv"],
            k_dim=case["k_dim"],
            v_dim=case["v_dim"],
            chunk_size=case["chunk_size"],
            dtype=parse_dtype(case["dtype"]),
            seed=case["seed"] + seq_id,
            cu_seqlens=None,
            input_ranges=case["input_ranges"],
        )
        work = {name: value.to(torch.float32) for name, value in inputs.items()}
        cpu = fused_cpu(
            work,
            chunk_size=case["chunk_size"],
            use_gate=case["use_gate"],
            safe_gate=case["safe_gate"],
            lower_bound=case["lower_bound"],
        )
        cpu_seq = {name: head_to_seq(value) for name, value in cpu.items()}
        del work, cpu
        seq = _move(_to_seq(inputs), device)
        del inputs
        staged = run_staged(
            seq["q"], seq["k"], seq["v"], seq["g"], seq["beta"], seq["A"],
            seq["A_log"], seq["dt_bias"], **kwargs,
        )
        _merge_rows(staged_acc, _compare(cpu_seq, staged, STAGED_NAMES), length)
        if not skip_fla:
            fla = run_fla(
                seq["q"], seq["k"], seq["v"], seq["g"], seq["beta"], seq["A"],
                seq["A_log"], seq["dt_bias"], **kwargs,
            )
            _merge_rows(fla_acc, _compare(cpu_seq, fla, FLA_NAMES), length)
            _merge_rows(vs_acc, _compare(staged, fla, FLA_NAMES), length)
            del fla
        del staged, seq, cpu_seq
        torch.cuda.empty_cache()
        if seq_id % 8 == 0:
            print(f"  varlen seq {seq_id}/{len(cu)-1} len={length}", flush=True)

    lines.append(f"varlen sequences={n_seq} (CPU/GPU per sequence)")
    _append_rows(lines, "CPU FP32 vs staged Triton", _finalize_rows(staged_acc))
    if not skip_fla:
        _append_rows(lines, "CPU FP32 vs FLA kda_gate_chunk_cumsum+recompute_w_u_fwd", _finalize_rows(fla_acc))
        _append_rows(lines, "staged Triton vs FLA", _finalize_rows(vs_acc))
    return "\n".join(lines) + "\n"


def run_one(case_path: Path, *, device: torch.device, skip_fla: bool) -> str:
    case = load_case(case_path)
    cu = case["cu_seqlens"]
    print(
        f"\n=== {case['case_id']}  B={case['batch']} HK={case['hk']} HV={case['hv']} "
        f"T={case['tokens']} chunk={case['chunk_size']} "
        f"{'varlen N=' + str(len(cu) - 1) if cu else 'dense'} ===",
        flush=True,
    )
    lines = [f"# {case['case_id']}"]
    if cu:
        return _run_one_varlen(case, device, skip_fla, lines)

    inputs = make_inputs(
        batch=case["batch"],
        tokens=case["tokens"],
        hk=case["hk"],
        hv=case["hv"],
        k_dim=case["k_dim"],
        v_dim=case["v_dim"],
        chunk_size=case["chunk_size"],
        dtype=parse_dtype(case["dtype"]),
        seed=case["seed"],
        cu_seqlens=None,
        input_ranges=case["input_ranges"],
    )
    return _run_one_dense(inputs, case, device, skip_fla, lines)


def _default_cases_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "cases", here / "cases"):
        if candidate.is_dir():
            return candidate
    return here.parent / "cases"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=_default_cases_dir())
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cases"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-fla", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="case stems, e.g. case0 case3")
    args = parser.parse_args()

    paths = sorted(args.cases_dir.glob("case*.json"))
    if args.only:
        wanted = set(args.only)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise SystemExit(f"no cases in {args.cases_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; CPU golden only", file=sys.stderr)
        device = torch.device("cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in paths:
        try:
            text = run_one(path, device=device, skip_fla=args.skip_fla)
        except Exception as exc:
            text = f"# {path.stem}\nFAILED: {type(exc).__name__}: {exc}\n"
            print(text, flush=True)
        summaries.append(text)
        (args.out_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
    (args.out_dir / "summary.txt").write_text("\n".join(summaries), encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
