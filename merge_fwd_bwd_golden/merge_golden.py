"""CPU formulas for merge_fwd_bwd stages.

Without h0 (CP default, start h=0). Rank 0 cube `M_0 @ 0` is skipped:

    V0:     h = He_0
    C_r:    h = M_r @ h            r = 1 .. R-1
    V_r:    h = h + He_r           r = 1 .. R-1

With h0 (intracard), every rank is the same C then V pair:

    C_r:    h = M_r @ h
    V_r:    h = h + He_r           r = 0 .. R-1

BWD uses the same updates with ranks reversed.
When state_v_first, h is [V, K]: MM is h @ M^T, add is h + He^T.
"""

from __future__ import annotations


def apply_mm(m, h, *, state_v_first: bool):
    if state_v_first:
        return h @ m.transpose(-1, -2)
    return m @ h


def apply_add(h, he, *, state_v_first: bool):
    if state_v_first:
        return h + he.transpose(-1, -2)
    return h + he


def merge_cpu(
    he,
    m,
    h0=None,
    *,
    reverse: bool = False,
    state_v_first: bool = False,
) -> dict:
    """he [R,HV,K,V], m [R,HV,K,K]. Returns stage snapshots plus final h."""
    ranks = he.shape[0]
    order = list(range(ranks - 1, -1, -1) if reverse else range(ranks))
    stages: dict = {}
    if h0 is None:
        if state_v_first:
            h = he[order[0]].transpose(-1, -2).contiguous()
        else:
            h = he[order[0]].contiguous()
        stages["h_v0"] = h.clone()
        start = 1
        pair_id = 1
    else:
        h = h0
        start = 0
        pair_id = 0
    for rank in order[start:]:
        h = apply_mm(m[rank], h, state_v_first=state_v_first)
        stages[f"h_c{pair_id}"] = h.clone()
        h = apply_add(h, he[rank], state_v_first=state_v_first)
        stages[f"h_v{pair_id}"] = h.clone()
        pair_id += 1
    stages["h"] = h
    return stages
