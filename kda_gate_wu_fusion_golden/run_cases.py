"""Run kda_gate_chunk_cumsum + recompute_w_u_fwd golden on cases/*.json."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import torch

from common import head_to_seq, load_case, make_inputs, parse_dtype, save_tensor
from compare_stats import stats
from kda_gate_wu_golden import fused_cpu

STAGED_NAMES = ("g_corr", "gk", "qg", "kbg", "vb", "kg", "w", "u")
FLA_NAMES = ("gk", "qg", "kg", "w", "u")
VIZ_ELEMS = 100_000


def _find_ct() -> str | None:
    env = os.environ.get("CT")
    if env and Path(env).exists():
        return env
    for candidate in (Path.home() / ".venvs/fla/bin/ct", Path.home() / ".local/bin/ct"):
        if candidate.exists():
            return str(candidate)
    return None


def _subsample_pair(real: torch.Tensor, expect: torch.Tensor, *, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    real_1d = real.detach().reshape(-1).cpu()
    expect_1d = expect.detach().reshape(-1).cpu()
    if real_1d.numel() != expect_1d.numel():
        raise ValueError(f"numel mismatch: {real_1d.numel()} vs {expect_1d.numel()}")
    count = min(n, real_1d.numel())
    if real_1d.numel() == count:
        return real_1d.contiguous(), expect_1d.contiguous()
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(real_1d.numel(), generator=generator)[:count]
    return real_1d[index].contiguous(), expect_1d[index].contiguous()


def _dump_pair(
    real: dict[str, torch.Tensor],
    expect: dict[str, torch.Tensor],
    names: tuple[str, ...],
    real_dir: Path,
    expect_dir: Path,
    *,
    seed: int,
) -> None:
    for offset, name in enumerate(names):
        if name not in real or name not in expect:
            continue
        real_s, expect_s = _subsample_pair(real[name], expect[name], n=VIZ_ELEMS, seed=seed + offset)
        save_tensor(real_dir / f"{name}.pt", real_s)
        save_tensor(expect_dir / f"{name}.pt", expect_s)


def _ct_viz(cpu_dir: Path, tri_dir: Path, out_dir: Path, names: tuple[str, ...]) -> None:
    ct = _find_ct()
    if ct is None:
        print("skip ct viz: ct binary not found", flush=True)
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        cpu = cpu_dir / f"{name}.pt"
        tri = tri_dir / f"{name}.pt"
        if not cpu.exists() or not tri.exists():
            print(f"skip ct viz {name}: missing dump", flush=True)
            continue
        print(f"ct viz {name} -> {out_dir}", flush=True)
        subprocess.run(
            [ct, "viz", str(cpu), str(tri), "--out_dir", str(out_dir), "--name", name, "-sc", str(VIZ_ELEMS), "-wl", "1"],
            check=True,
        )


def _viz_dirs(dump_dir: Path) -> dict[str, Path]:
    return {
        "cpu": dump_dir / "cpu_fp32",
        "staged": dump_dir / "triton_staged",
        "fla": dump_dir / "triton_fla",
        "ct_staged": dump_dir / "ct_staged",
        "ct_fla": dump_dir / "ct_fla",
    }


def _to_seq(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value if name in {"A_log", "dt_bias"} else head_to_seq(value)
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
            out[name] = value[:, :, start:end] if value.ndim >= 3 else value[:, start:end]
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
        cpu_part = {name: cpu_seq[name][:, :, start:end] if cpu_seq[name].ndim >= 3 else cpu_seq[name][:, start:end]
                    for name in names if name in cpu_seq}
        _merge_rows(acc, _compare(cpu_part, got, names), (end - start))
        del got, part_in
        torch.cuda.empty_cache()
    return _finalize_rows(acc), None


def _run_one_dense(inputs, case, device, skip_fla, lines, dump_dir: Path | None):
    work = {name: value.to(torch.float32) for name, value in inputs.items()}
    cpu = fused_cpu(
        work,
        chunk_size=case["chunk_size"],
        use_gate=case["use_gate"],
        safe_gate=case["safe_gate"],
        lower_bound=case["lower_bound"],
        cu_seqlens=None,
    )
    del work
    if device.type != "cuda":
        print("skip GPU: CUDA not available")
        lines.append("GPU skipped")
        return "\n".join(lines) + "\n"

    from triton_kernels import run_fla, run_staged

    dirs = _viz_dirs(dump_dir) if dump_dir is not None else None
    kwargs = dict(
        chunk_size=case["chunk_size"],
        use_gate=case["use_gate"],
        safe_gate=case["safe_gate"],
        lower_bound=case["lower_bound"],
        cu_seqlens=None,
    )
    bnsd = _move(inputs, device)
    staged_rows, staged = _run_pair(run_staged, bnsd, kwargs, cpu, STAGED_NAMES, None, device)
    _append_rows(lines, "CPU FP32 vs staged Triton", staged_rows)
    if dirs is not None and staged is not None:
        _dump_pair(cpu, staged, STAGED_NAMES, dirs["cpu"], dirs["staged"], seed=case["seed"])
    del staged, bnsd
    torch.cuda.empty_cache()
    gc.collect()

    if not skip_fla:
        cpu_seq = {name: head_to_seq(cpu[name]) for name in FLA_NAMES}
        del cpu
        gc.collect()
        seq = _move(_to_seq(inputs), device)
        fla_rows, fla = _run_pair(run_fla, seq, kwargs, cpu_seq, FLA_NAMES, None, device)
        _append_rows(lines, "CPU FP32 vs FLA kda_gate_chunk_cumsum+recompute_w_u_fwd", fla_rows)
        if dirs is not None and fla is not None:
            _dump_pair(cpu_seq, fla, FLA_NAMES, dirs["cpu"] / "fla_view", dirs["fla"], seed=case["seed"] + 17)
        del fla, seq, cpu_seq
        torch.cuda.empty_cache()
        gc.collect()
    else:
        del cpu

    del inputs
    gc.collect()
    if dirs is not None:
        _ct_viz(dirs["cpu"], dirs["staged"], dirs["ct_staged"], STAGED_NAMES)
        if not skip_fla:
            _ct_viz(dirs["cpu"] / "fla_view", dirs["fla"], dirs["ct_fla"], FLA_NAMES)
    return "\n".join(lines) + "\n"


def _run_one_varlen(case, device, skip_fla, lines, dump_dir: Path | None):
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
    staged_real_parts = {name: [] for name in STAGED_NAMES}
    staged_expect_parts = {name: [] for name in STAGED_NAMES}
    fla_real_parts = {name: [] for name in FLA_NAMES}
    fla_expect_parts = {name: [] for name in FLA_NAMES}
    kwargs = dict(
        chunk_size=case["chunk_size"],
        use_gate=case["use_gate"],
        safe_gate=case["safe_gate"],
        lower_bound=case["lower_bound"],
        cu_seqlens=None,
    )
    n_seq = 0
    tokens = case["tokens"]
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
        del work
        bnsd = _move(inputs, device)
        seq = _move(_to_seq(inputs), device)
        del inputs
        staged = run_staged(
            bnsd["q"], bnsd["k"], bnsd["v"], bnsd["g"], bnsd["beta"], bnsd["A"],
            bnsd["A_log"], bnsd["dt_bias"], **kwargs,
        )
        _merge_rows(staged_acc, _compare(cpu, staged, STAGED_NAMES), length)
        n_sample = max(256, int(VIZ_ELEMS * length / max(tokens, 1)))
        if dump_dir is not None:
            for offset, name in enumerate(STAGED_NAMES):
                real_s, expect_s = _subsample_pair(
                    cpu[name], staged[name], n=n_sample, seed=case["seed"] + seq_id * 32 + offset,
                )
                staged_real_parts[name].append(real_s)
                staged_expect_parts[name].append(expect_s)
        if not skip_fla:
            fla = run_fla(
                seq["q"], seq["k"], seq["v"], seq["g"], seq["beta"], seq["A"],
                seq["A_log"], seq["dt_bias"], **kwargs,
            )
            _merge_rows(fla_acc, _compare(cpu_seq, fla, FLA_NAMES), length)
            staged_seq = {name: head_to_seq(staged[name]) for name in FLA_NAMES}
            _merge_rows(vs_acc, _compare(staged_seq, fla, FLA_NAMES), length)
            if dump_dir is not None:
                for offset, name in enumerate(FLA_NAMES):
                    real_s, expect_s = _subsample_pair(
                        cpu_seq[name], fla[name], n=n_sample, seed=case["seed"] + 17 + seq_id * 32 + offset,
                    )
                    fla_real_parts[name].append(real_s)
                    fla_expect_parts[name].append(expect_s)
            del fla
        del staged, bnsd, seq, cpu, cpu_seq
        torch.cuda.empty_cache()
        if seq_id % 8 == 0:
            print(f"  varlen seq {seq_id}/{len(cu)-1} len={length}", flush=True)

    lines.append(f"varlen sequences={n_seq} (CPU/GPU per sequence)")
    _append_rows(lines, "CPU FP32 vs staged Triton", _finalize_rows(staged_acc))
    if not skip_fla:
        _append_rows(lines, "CPU FP32 vs FLA kda_gate_chunk_cumsum+recompute_w_u_fwd", _finalize_rows(fla_acc))
        _append_rows(lines, "staged Triton vs FLA", _finalize_rows(vs_acc))
    if dump_dir is not None:
        dirs = _viz_dirs(dump_dir)

        def _cat_trim(parts: list[torch.Tensor]) -> torch.Tensor:
            cat = torch.cat(parts)
            return cat[:VIZ_ELEMS].contiguous()

        for name in STAGED_NAMES:
            if staged_real_parts[name]:
                save_tensor(dirs["cpu"] / f"{name}.pt", _cat_trim(staged_real_parts[name]))
                save_tensor(dirs["staged"] / f"{name}.pt", _cat_trim(staged_expect_parts[name]))
        _ct_viz(dirs["cpu"], dirs["staged"], dirs["ct_staged"], STAGED_NAMES)
        if not skip_fla:
            for name in FLA_NAMES:
                if fla_real_parts[name]:
                    save_tensor(dirs["cpu"] / "fla_view" / f"{name}.pt", _cat_trim(fla_real_parts[name]))
                    save_tensor(dirs["fla"] / f"{name}.pt", _cat_trim(fla_expect_parts[name]))
            _ct_viz(dirs["cpu"] / "fla_view", dirs["fla"], dirs["ct_fla"], FLA_NAMES)
    return "\n".join(lines) + "\n"


def run_one(case_path: Path, *, device: torch.device, skip_fla: bool, dump_dir: Path | None) -> str:
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
        return _run_one_varlen(case, device, skip_fla, lines, dump_dir)

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
    return _run_one_dense(inputs, case, device, skip_fla, lines, dump_dir)


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
    parser.add_argument("--no-viz", action="store_true", help="stats only, skip ct viz dumps")
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
            dump_dir = None if args.no_viz else (args.out_dir / path.stem)
            text = run_one(path, device=device, skip_fla=args.skip_fla, dump_dir=dump_dir)
        except Exception as exc:
            text = f"# {path.stem}\nFAILED: {type(exc).__name__}: {exc}\n"
            print(text, flush=True)
        summaries.append(text)
        (args.out_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
    (args.out_dir / "summary.txt").write_text("\n".join(summaries), encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
