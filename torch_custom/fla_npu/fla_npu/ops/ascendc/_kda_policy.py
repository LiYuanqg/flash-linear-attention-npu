"""Shared KDA host policy for forward outputs and optimized backward inputs."""

from __future__ import annotations

from typing import Tuple


FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT = "0f0f0c97af39343855b43bbbaddcedfda5cb9d77"
FLA_ORG_KDA_FWD_ALIGNMENT_SOURCE = (
    "https://github.com/fla-org/flash-linear-attention/blob/"
    f"{FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT}/fla/ops/kda/chunk_fwd.py"
)


def kda_fwd_optional_output_mask(
    *,
    output_final_state: bool,
    use_gate_in_kernel: bool,
    disable_recompute: bool,
    return_intermediate_states: bool,
) -> Tuple[bool, ...]:
    """Return the visibility mask for the low-level 12-value FLA interface."""

    return (
        True,
        output_final_state,
        not use_gate_in_kernel or disable_recompute,
        True,
        True,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute,
        disable_recompute or return_intermediate_states,
        True,
    )


def _select_kda_bwd_optimized(implementation, q_rstd, k_rstd, disable_recompute):
    if implementation not in ("auto", "legacy", "optimized"):
        raise ValueError("implementation must be auto, legacy or optimized")
    if (q_rstd is None) != (k_rstd is None):
        raise ValueError("q_rstd and k_rstd must be supplied together")
    if implementation == "legacy":
        if q_rstd is not None:
            raise ValueError("legacy does not support L2Norm backward")
        return False
    return implementation == "optimized" or q_rstd is not None or not disable_recompute


def _canonical_kda_bwd_metadata(cu, indices, total_tokens, chunk_size=64):
    """Remove empty sequences without changing token or global chunk storage."""
    import operator

    def host_list(value):
        if hasattr(value, "device") and value.device.type != "cpu":
            raise ValueError("optimized metadata must be Host-resident; device readback is not implicit")
        if hasattr(value,"detach"):
            value=value.detach().flatten().tolist()
        return tuple(operator.index(x) for x in value)

    def chunk_pairs(boundaries):
        return tuple(value for seq,(a,b) in enumerate(zip(boundaries,boundaries[1:]))
                     for chunk in range((b-a+chunk_size-1)//chunk_size) for value in (seq,chunk))

    if cu is None:
        if indices is not None:
            raise ValueError("chunk_indices requires cu_seqlens")
        return None, None, (total_tokens + chunk_size - 1) // chunk_size
    cu = host_list(cu)
    if len(cu) < 2 or cu[0] != 0 or cu[-1] != total_tokens or any(a > b for a,b in zip(cu,cu[1:])):
        raise ValueError("invalid cu_seqlens")
    original = chunk_pairs(cu)
    if indices is not None and host_list(indices) != original:
        raise ValueError("chunk_indices must use canonical sequence-major order")
    compact = (cu[0],) + tuple(b for a,b in zip(cu,cu[1:]) if b > a)
    indices = chunk_pairs(compact)
    return compact, indices, len(indices) // 2


def _prepare_kda_bwd_optimized(args):
    import torch
    import torch_npu
    import math
    from ._runtime import (
        optional_bool as _optional_bool, optional_float as _optional_float,
        ACL_FORMAT_ND, ACL_FORMAT_NCHW, ACL_FORMAT_NCDHW, ACL_FORMAT_NCL,
    )

    args = dict(args)
    for name, default in (("disable_recompute",True),("safe_gate",True),
                          ("use_gate_in_kernel",False),("use_exp2",True),("state_v_first",False)):
        args[name] = _optional_bool(args[name],default)
    args["lower_bound"] = _optional_float(args["lower_bound"],-5.0)
    q = args["q"]
    if q.device.type != "npu" or "Ascend950" not in torch.npu.get_device_name(q.device):
        raise ValueError("optimized KDA backward requires Ascend950/A5")
    packed = args["cu_seqlens"] is not None
    shape = tuple(q.shape)
    if len(shape) != (3 if packed else 4):
        raise ValueError("optimized expects dense [B,H,T,D] or packed [H,T,D]")
    b,h,t,d = (1,*shape) if packed else shape
    if min(b,h,t) <= 0 or d != 128:
        raise ValueError("optimized requires positive B/H/T and K=V=128; T=0 is not supported")
    if not math.isfinite(float(args["scale"])):
        raise ValueError("scale must be finite")
    if (int(args["chunk_size"]) != 64 or not args["safe_gate"] or
            not args["use_gate_in_kernel"] or not args["use_exp2"] or args["state_v_first"]):
        raise ValueError("optimized requires C=64, safe gate, gate-in-kernel, exp2 and K-first states")
    if args["initial_state"] is not None or args["dht"] is not None:
        raise ValueError("optimized initial_state/dht are not supported")
    if not args["disable_recompute"] and (h > 256 or h % 8):
        raise ValueError("optimized recompute currently requires H<=256 and H divisible by 8")
    cu, indices, nc = _canonical_kda_bwd_metadata(args["cu_seqlens"], args["chunk_indices"], t)
    token = shape
    scalar = shape[:-1]
    state = (nc,h,128,128) if packed else (b,nc,h,128,128)

    def check(name, expected, dtypes, optional=False):
        x = args[name]
        if optional and x is None:
            return
        if x is None or tuple(x.shape) != expected or x.dtype not in dtypes:
            raise ValueError(f"optimized {name}: expected shape {expected}, dtype {dtypes}")
        if x.device != q.device or not x.is_contiguous():
            raise ValueError(f"optimized {name}: expected contiguous tensor on q.device")
        standard_formats = {ACL_FORMAT_ND, ACL_FORMAT_NCHW,
                            ACL_FORMAT_NCDHW, ACL_FORMAT_NCL}
        if int(torch_npu.get_npu_format(x)) not in standard_formats:
            raise ValueError(f"optimized {name}: private NPU storage formats are not supported")

    bf16 = (torch.bfloat16,)
    fp32 = (torch.float32,)
    mixed = (torch.bfloat16,torch.float32)
    for name in ("q","k","v","d_o"):
        check(name,token,bf16)
    check("beta",scalar,mixed)
    for name in ("Aqk","Akk"):
        check(name,(*scalar,64),bf16)
    check("raw_g",token,mixed)
    check("A_log",(h,),mixed)
    for name in ("q_rstd","k_rstd"):
        check(name,scalar,fp32,optional=True)
    saved = ("gk","w","qg","kg","v_new","h")
    if args["disable_recompute"]:
        for name in saved:
            check(name,state if name == "h" else token,fp32 if name == "gk" else bf16)
    elif any(args[name] is not None for name in saved):
        raise ValueError("recompute mode requires gk/w/qg/kg/v_new/h to be None")
    bias = args["dt_bias"]
    if bias is not None:
        if bias.dtype != torch.float32 or bias.numel() != h*128 or not bias.is_contiguous() or bias.device != q.device:
            raise ValueError("dt_bias must be contiguous FP32 with H*128 elements on q.device")
        bias = bias.view(h,128)

    args.update(dt_bias=bias, cu_seqlens=cu, chunk_indices=indices)
    return args


_KDA_BWD_TOKEN_TENSORS = (
    "q", "k", "v", "beta", "gk", "Aqk", "Akk", "w", "qg", "kg", "v_new",
    "d_o", "raw_g", "q_rstd", "k_rstd",
)
_KDA_TAIL_GUARD_LOGS = 0


def _log_kda_tail_guard(message):
    global _KDA_TAIL_GUARD_LOGS
    _KDA_TAIL_GUARD_LOGS += 1
    if _KDA_TAIL_GUARD_LOGS <= 8:
        import warnings
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _kda_host_cu(cu):
    if cu is None:
        return None
    if hasattr(cu, "detach"):
        cu = cu.detach().cpu().flatten().tolist()
    return tuple(int(x) for x in cu)


def _kda_chunk_pairs(cu, chunk_size):
    """Canonical sequence-major chunk_indices for a host cu_seqlens."""
    pairs = []
    for seq, (begin, end) in enumerate(zip(cu, cu[1:])):
        n_chunks = (end - begin + chunk_size - 1) // chunk_size
        for chunk in range(n_chunks):
            pairs.extend((seq, chunk))
    return tuple(pairs)


def _sync_npu(tensor):
    """Wait until queued NPU work that produced ``tensor`` has stored it.

    Stable launches go through the torch_npu task queue and return before the
    kernel store runs. A later in-place write on that same storage is then
    overwritten when the kernel is submitted, so the caller still sees the
    unrepaired values.
    """
    device = getattr(tensor, "device", None)
    if device is None or getattr(device, "type", None) != "npu":
        return
    import torch
    torch.npu.synchronize(device)


def _repair_padded_chunk_kg(k, gk, kg, valid_tokens, chunk_size):
    """Rebuild ``kg`` on the last partial chunk after a zero-padded relaunch.

    Padding makes that chunk 64 rows so the fused VF does not spill into
    rows 0..16. The extra ``g`` rows are zeros, but safe-gate maps
    ``g=0`` to ``lower_bound * sigmoid(exp(A_log) * dt_bias)``, which is
    not a zero cumsum step. The kernel then takes ``gk_last`` from the
    padded row and writes ``kg = k * exp2(gk_last - gk)`` for every valid
    token in the chunk. ``gk`` itself is causal, so the last real token is
    the ``gk_last`` those rows should have used.
    """
    import torch

    if gk is None or kg is None or valid_tokens <= 0:
        return kg
    start = (int(valid_tokens) // int(chunk_size)) * int(chunk_size)
    if start >= int(valid_tokens):
        return kg
    hk = int(k.shape[1])
    hv = int(gk.shape[1])
    if hk <= 0 or hv % hk != 0:
        return kg
    k_hv = k if hv == hk else k.repeat_interleave(hv // hk, dim=1)
    gk_tail = gk.narrow(2, start, int(valid_tokens) - start)
    gk_last = gk.narrow(2, int(valid_tokens) - 1, 1)
    n_tail = int(valid_tokens) - start
    delta = (gk_last - gk_tail).clamp(-80.0, 80.0)
    fixed = (k_hv.narrow(2, start, n_tail).float() * torch.exp2(delta)).to(dtype=kg.dtype)
    # New storage: the queued kernel still owns the buffer it was given.
    out = kg.clone()
    out.narrow(2, start, n_tail).copy_(fixed)
    if _KDA_TAIL_GUARD_LOGS < 8:
        max_abs = float((fixed.float() - kg.narrow(2, start, n_tail).float()).abs().max())
        _log_kda_tail_guard(
            "KDA recompute kg repair seqlen=%d max_abs=%s" % (int(valid_tokens), max_abs)
        )
    return out


def _kda_pad_token_tensor(tensor, token_dim, seqlen, pad_rows, repeat_last=False):
    import torch

    if tensor is None:
        return None
    pad_shape = list(tensor.shape)
    pad_shape[token_dim] = pad_rows
    if repeat_last:
        tail = tensor.narrow(token_dim, seqlen - 1, 1).expand(*pad_shape).clone()
    else:
        tail = tensor.new_zeros(pad_shape)
    return torch.cat((tensor, tail), dim=token_dim).contiguous()


def _kda_slice_padded_bwd(outputs, original_seqlen, token_dim):
    restored = []
    for index, value in enumerate(outputs):
        if value is None:
            restored.append(None)
            continue
        if index < 5:
            value = value.narrow(token_dim, 0, original_seqlen)
        restored.append(value.contiguous())
    return tuple(restored)


def _kda_combine_split_bwd(results):
    import torch

    restored = []
    for output_index in range(8):
        values = [result[output_index] for result in results]
        if values[0] is None:
            restored.append(None)
        elif output_index < 5:
            restored.append(torch.cat(
                [value.squeeze(0) for value in values], dim=1).contiguous())
        else:
            total = values[0]
            for value in values[1:]:
                total = total + value
            restored.append(total)
    return tuple(restored)


def run_kda_recompute_with_tail_guard(
    q, k, v, g, beta, a, launch, *, cu_seqlens=None, chunk_indices=None,
    chunk_size=64, repair_kg=False,
):
    """Zero-pad leftover T%64 before ChunkKdaBwdRecompute.

    Fused VF extra stores can overwrite leftover rows 0..16. Kernel repair
    only covers leftover < 16, so training tails of 16-63 stay wrong unless
    the last chunk is a full 64. When the kernel applies safe-gate itself,
    ``repair_kg`` rebuilds the last chunk's ``kg`` from the real last token.
    ``launch`` returns ``(gk, w, u, qg, kg)``.
    """
    import torch

    cu = _kda_host_cu(cu_seqlens)
    token_dim = 2
    seqlen = int(q.shape[token_dim])
    tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "a": a}

    def _launch(tok, cu_arg, indices_arg):
        return launch(
            tok["q"], tok["k"], tok["v"], tok["g"], tok["beta"], tok["a"],
            cu_arg, indices_arg,
        )

    def _slice_outputs(outputs, original_seqlen):
        sliced = []
        for value in outputs:
            if value is None:
                sliced.append(None)
            else:
                sliced.append(value.narrow(token_dim, 0, original_seqlen).contiguous())
        return tuple(sliced)

    has_varlen_tail = cu is not None and any(
        (end - begin) % chunk_size != 0 for begin, end in zip(cu, cu[1:]))
    if has_varlen_tail and len(cu) > 2:
        _log_kda_tail_guard(
            f"KDA recompute tail-guard split packed leftover cu={cu}"
        )
        parts = []
        for start, end in zip(cu, cu[1:]):
            seq_len = end - start
            if seq_len <= 0:
                continue
            sub = {
                name: tensor.narrow(token_dim, start, seq_len).contiguous()
                for name, tensor in tensors.items()
            }
            parts.append(run_kda_recompute_with_tail_guard(
                sub["q"], sub["k"], sub["v"], sub["g"], sub["beta"], sub["a"],
                launch, cu_seqlens=None, chunk_indices=None,
                chunk_size=chunk_size, repair_kg=repair_kg,
            ))
        combined = []
        for output_index in range(5):
            values = [part[output_index] for part in parts]
            if values[0] is None:
                combined.append(None)
            else:
                combined.append(torch.cat(values, dim=token_dim).contiguous())
        return tuple(combined)

    if seqlen % chunk_size != 0 and (cu is None or len(cu) == 2):
        pad_rows = (
            (seqlen + chunk_size - 1) // chunk_size
        ) * chunk_size - seqlen
        _log_kda_tail_guard(
            f"KDA recompute tail-guard pad seqlen={seqlen} pad={pad_rows}"
        )
        padded = {
            name: _kda_pad_token_tensor(
                tensor, token_dim, seqlen, pad_rows, repeat_last=False,
            )
            for name, tensor in tensors.items()
        }
        # Padding stays inside the existing last chunk, so the chunk count
        # does not grow. Varlen tiling rejects cu_seqlens without
        # chunk_indices (ACLNN_ERR_INNER_NULLPTR / 561103).
        padded_cu = None if cu is None else (0, seqlen + pad_rows)
        padded_indices = (
            None if padded_cu is None else _kda_chunk_pairs(padded_cu, chunk_size)
        )
        outputs = _launch(padded, padded_cu, padded_indices)
        if repair_kg and outputs[0] is not None and outputs[4] is not None:
            _sync_npu(outputs[0])
            gk, w, u, qg, kg = outputs
            kg = _repair_padded_chunk_kg(k, gk, kg, seqlen, chunk_size)
            outputs = (gk, w, u, qg, kg)
        return _slice_outputs(outputs, seqlen)

    return _launch(tensors, cu, chunk_indices)


def run_kda_bwd_optimized_with_tail_guard(args, launch):
    """Keep V2 C-Intra off leftover rows.

    Packed leftover sequences are split into dense calls; a single leftover
    chunk is padded to 64 inside the existing last state, then token grads
    are sliced back.  Full 64-token chunks stay on one launch.
    """
    import torch

    args = dict(args)
    q = args["q"]
    if q is None:
        return launch(args)
    chunk_size = int(args.get("chunk_size") or 64)
    cu = _kda_host_cu(args.get("cu_seqlens"))
    packed = cu is not None
    seqlen = int(q.shape[1] if packed else q.shape[2])
    token_dim = 1 if packed else 2
    has_varlen_tail = packed and any(
        (end - begin) % chunk_size != 0 for begin, end in zip(cu, cu[1:]))

    if has_varlen_tail and len(cu) > 2:
        _log_kda_tail_guard(f"KDA V2 tail-guard split packed leftover cu={cu}")
        h_state = args["h"]
        chunk_begin = 0
        results = []
        for start, end in zip(cu, cu[1:]):
            seq_len = end - start
            if seq_len <= 0:
                continue
            n_chunks = (seq_len + chunk_size - 1) // chunk_size
            sub = dict(args)

            def dense_slice(tensor):
                if tensor is None:
                    return None
                return tensor.narrow(1, start, seq_len).unsqueeze(0).contiguous()

            for name in _KDA_BWD_TOKEN_TENSORS:
                sub[name] = dense_slice(args.get(name))
            sub["h"] = h_state.narrow(0, chunk_begin, n_chunks).unsqueeze(0).contiguous()
            sub["cu_seqlens"] = None
            sub["chunk_indices"] = None
            results.append(run_kda_bwd_optimized_with_tail_guard(sub, launch))
            chunk_begin += n_chunks
        return _kda_combine_split_bwd(results)

    if seqlen % chunk_size != 0 and (cu is None or len(cu) == 2):
        _log_kda_tail_guard(
            f"KDA V2 tail-guard pad seqlen={seqlen} packed={packed}"
        )
        pad_rows = ((seqlen + chunk_size - 1) // chunk_size) * chunk_size - seqlen
        for name in _KDA_BWD_TOKEN_TENSORS:
            args[name] = _kda_pad_token_tensor(
                args.get(name), token_dim, seqlen, pad_rows,
                repeat_last=(name == "gk"))
        if cu is not None:
            args["cu_seqlens"] = (0, seqlen + pad_rows)
            args["chunk_indices"] = _kda_chunk_pairs(args["cu_seqlens"], chunk_size)
        return _kda_slice_padded_bwd(launch(args), seqlen, token_dim)

    return launch(args)
