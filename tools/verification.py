from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class VerifyResult:
    commit_depth: torch.LongTensor             # [B]
    commit_beam:  torch.LongTensor             # [B]
    commit_tokens: List[List[int]]             # [t0..t_d*]
    commit_real_positions: List[List[int]]     # real-row indices along the path
    commit_preds: List[List[int]]              # base preds at each path node (depth 0 included)
    accepted_masks_by_depth: List[torch.BoolTensor]  # per-depth [B,K]

def _filter_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature and temperature > 0.0:
        logits = logits / float(temperature)
    V = logits.size(-1)
    if top_k and 0 < top_k < V:
        vals, _ = torch.topk(logits, k=top_k, dim=-1)
        thr = vals[..., -1].unsqueeze(-1)
        logits = torch.where(logits >= thr, logits, torch.full_like(logits, float("-inf")))
    if top_p and 0.0 < top_p < 1.0:
        s, idx = torch.sort(logits, dim=-1, descending=True)
        p = torch.softmax(s, dim=-1)
        c = p.cumsum(dim=-1)
        cut = c > top_p
        cut[..., 0] = False
        s = torch.where(cut, torch.full_like(s, float("-inf")), s)
        inv = torch.empty_like(idx)
        inv.scatter_(-1, idx, torch.arange(V, device=logits.device).expand_as(idx))
        logits = s.gather(-1, inv)
    return logits

@torch.no_grad()
def verify_routes(
    logits: torch.FloatTensor,    # [B, L, V]
    pack,                         # needs .real_pos, .alive_by_d
    tree,                         # BatchedBeamTree
    do_sample: bool,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> VerifyResult:
    device = logits.device
    B, L, V = logits.shape
    D = tree.cur_depth
    K = tree.K

    real_pos    = pack.real_pos              # [B, D+1, K] (−1 for dead)
    alive_by_d  = pack.alive_by_d            # list of [B, K] (bool)
    tokens_by_d = [tree.tokens_by_d[d] for d in range(D + 1)]          # [B, K]
    parent_by_d = [None] + [tree.parent_by_d[d].to(torch.int64) for d in range(1, D + 1)]
    cum_by_d    = [tree.cum_by_d[d] for d in range(D + 1)]

    # 1) Compute base-model prediction ONCE at every alive node’s REAL row
    #    pred_at_real[b, d, k] is used:
    #      - to validate children at depth d+1 (as the parent’s pred)
    #      - to report the path preds (current depth’s pred)
    pred_at_real = torch.full((B, D + 1, K), -1, dtype=torch.long, device=device)

    for d in range(D + 1):
        alive = alive_by_d[d]                              # [B, K]
        idx = torch.nonzero(alive, as_tuple=False)         # [N,2]
        if idx.numel() == 0:
            continue
        b_idx = idx[:, 0]; k_idx = idx[:, 1]
        rows  = real_pos[b_idx, d, k_idx]                  # [N]
        row_logits = logits[b_idx, rows, :]                # [N, V]
        if do_sample:
            fl = _filter_logits(row_logits, temperature, int(top_k or 0), float(top_p or 0.0))
            probs = torch.softmax(fl, dim=-1)
            bad = torch.isinf(fl).all(dim=-1)
            if bad.any():
                probs[bad] = torch.softmax(row_logits[bad], dim=-1)
            preds = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(1)  # [N]
        else:
            preds = torch.argmax(row_logits, dim=-1)       # [N]
        pred_at_real[b_idx, d, k_idx] = preds

    # 2) Verify children using their PARENT’S pred_at_real (cascading)
    accepted_masks_by_depth: List[torch.BoolTensor] = []
    acc_prev = alive_by_d[0].clone()                       # depth-0 is already valid
    accepted_masks_by_depth.append(acc_prev)

    for d in range(1, D + 1):
        alive_d = alive_by_d[d]                            # [B, K]
        parents = parent_by_d[d]                           # [B, K]
        drafted = tokens_by_d[d]                           # [B, K]
        acc_d = torch.zeros_like(alive_d, dtype=torch.bool)

        idx = torch.nonzero(alive_d, as_tuple=False)       # [Nc, 2]
        if idx.numel() > 0:
            b_c = idx[:, 0]; k_c = idx[:, 1]
            p_c = parents[b_c, k_c]                        # [Nc]
            # parent must be accepted and exist
            parent_ok = (p_c >= 0) & acc_prev[b_c, p_c]
            # parent’s pred equals this child’s drafted token?
            parent_pred = pred_at_real[b_c, d - 1, p_c]    # [Nc]
            child_tok   = drafted[b_c, k_c]
            ok = parent_ok & (parent_pred == child_tok)
            acc_d[b_c, k_c] = ok

        accepted_masks_by_depth.append(acc_d)
        acc_prev = acc_d

    # 3) Choose commit (deepest accepted, tie-break by highest cum_logp)
    commit_depth = torch.zeros(B, dtype=torch.long, device=device)
    commit_beam  = torch.zeros(B, dtype=torch.long, device=device)
    commit_tokens: List[List[int]] = [[] for _ in range(B)]
    commit_real_positions: List[List[int]] = [[] for _ in range(B)]
    commit_preds: List[List[int]] = [[] for _ in range(B)]

    for b in range(B):
        # deepest depth with any acceptance
        best_d = 0
        for d in range(1, D + 1):
            if bool(accepted_masks_by_depth[d][b].any()):
                best_d = d

        if best_d == 0:
            # root only (usually k=0)
            k0 = int(torch.nonzero(alive_by_d[0][b], as_tuple=False)[0].item()) if alive_by_d[0][b].any() else 0
            commit_depth[b] = 0
            commit_beam[b]  = k0
            toks, _ = tree.accept(b, depth=0, beam_idx=k0)
            r0 = int(real_pos[b, 0, k0].item())
            p0 = int(pred_at_real[b, 0, k0].item())
            commit_tokens[b] = toks
            commit_real_positions[b] = [r0]
            commit_preds[b] = [p0]
            continue

        # pick best beam among accepted at deepest depth
        mask = accepted_masks_by_depth[best_d][b]                      # [K]
        scores = torch.where(mask, cum_by_d[best_d][b],
                             torch.full_like(cum_by_d[best_d][b], float("-inf")))
        k_star = int(torch.argmax(scores).item())
        commit_depth[b] = best_d
        commit_beam[b]  = k_star

        # recover path beams k0..k_best_d
        beams = [k_star]
        for d in range(best_d, 0, -1):
            beams.append(int(parent_by_d[d][b, beams[-1]].item()))
        beams.reverse()  # [k0, k1, ..., k_best_d]

        toks, _ = tree.accept(b, depth=best_d, beam_idx=k_star)
        rpos = [int(real_pos[b, d, beams[d]].item()) for d in range(0, best_d + 1)]
        preds_path = [int(pred_at_real[b, d, beams[d]].item()) for d in range(0, best_d + 1)]

        commit_tokens[b] = toks
        commit_real_positions[b] = rpos
        commit_preds[b] = preds_path

    return VerifyResult(
        commit_depth=commit_depth,
        commit_beam=commit_beam,
        commit_tokens=commit_tokens,
        commit_real_positions=commit_real_positions,
        commit_preds=commit_preds,                      # <- “current depth” preds along path
        accepted_masks_by_depth=accepted_masks_by_depth,
    )


@torch.no_grad()
def update_kv_cache(
    past_key_values,                            # DynamicCache from the latest forward (len = past_seen_tokens + L)
    past_seen_tokens: int,                      # seq length BEFORE this forward
    commit_real_positions: list[list[int]],     # from verify: per-batch local rows (0..L-1) of accepted REAL tokens
    pad_with: str = "zeros",                    # "zeros" | "repeat_last"
) -> int:
    """
    In-place:
      - Keep cache[ : past_seen_tokens ] as-is.
      - For each batch b, copy accepted rows (local -> absolute) into slots
        [past_seen_tokens : past_seen_tokens + count_b).
      - Fill the remainder up to the batch-wise max with padding.
      - Truncate sequence dim to new_len = past_seen_tokens + max(count_b).
    Returns the new seq length.
    """
    B = len(commit_real_positions)
    counts = [len(r) for r in commit_real_positions]
    Amax   = max(counts, default=0)
    new_len = past_seen_tokens + Amax

    def _dims(t: torch.Tensor):
        # Accept both [B, H, S, D] and [H, B, S, D]; sequence dim is index 2 in both
        if t.size(0) == B:   return 0, 2
        if t.size(1) == B:   return 1, 2
        raise RuntimeError(f"Unexpected cache shape {tuple(t.shape)}; cannot locate batch dim {B}")

    for layer in past_key_values.layers:
        K = layer.keys
        V = layer.values
        bdim, sdim = _dims(K)
        cur_len = K.size(sdim)
        # Source snapshot (avoid overwriting reads)
        Ksrc = K.clone()
        Vsrc = V.clone()

        for b in range(B):
            rows_local = commit_real_positions[b]
            cb = len(rows_local)

            if cb > 0:
                src_abs = torch.as_tensor(rows_local, device=K.device, dtype=torch.long) + past_seen_tokens
                tgt_abs = torch.arange(past_seen_tokens, past_seen_tokens + cb, device=K.device)

                if bdim == 0:
                    # K[b, :, tgt, :] = Ksrc[b, :, src, :]
                    K[b, :, tgt_abs, :] = Ksrc[b, :, src_abs, :]
                    V[b, :, tgt_abs, :] = Vsrc[b, :, src_abs, :]
                else:
                    # K[:, b, tgt, :] = Ksrc[:, b, src, :]
                    K[:, b, tgt_abs, :] = Ksrc[:, b, src_abs, :]
                    V[:, b, tgt_abs, :] = Vsrc[:, b, src_abs, :]

            # pad the remainder [past_seen_tokens+cb : new_len) for this batch
            if new_len > past_seen_tokens + cb:
                pad_idx = torch.arange(past_seen_tokens + cb, new_len, device=K.device)
                if pad_with == "repeat_last" and cb > 0:
                    last_src = past_seen_tokens + rows_local[-1]
                    if bdim == 0:
                        K[b, :, pad_idx, :] = Ksrc[b, :, last_src, :].unsqueeze(-2).expand(-1, pad_idx.numel(), -1)
                        V[b, :, pad_idx, :] = Vsrc[b, :, last_src, :].unsqueeze(-2).expand(-1, pad_idx.numel(), -1)
                    else:
                        K[:, b, pad_idx, :] = Ksrc[:, b, last_src, :].unsqueeze(-2).expand(-1, pad_idx.numel(), -1)
                        V[:, b, pad_idx, :] = Vsrc[:, b, last_src, :].unsqueeze(-2).expand(-1, pad_idx.numel(), -1)
                else:
                    # zeros padding (safe; those slots won’t be attended)
                    if bdim == 0:
                        K[b, :, pad_idx, :].zero_()
                        V[b, :, pad_idx, :].zero_()
                    else:
                        K[:, b, pad_idx, :].zero_()
                        V[:, b, pad_idx, :].zero_()

        # finally truncate sequence dim to new_len
        if K.size(sdim) != new_len:
            slicer = [slice(None)] * K.dim()
            slicer[sdim] = slice(0, new_len)
            layer.keys   = K[tuple(slicer)].contiguous()
            layer.values = V[tuple(slicer)].contiguous()

    if hasattr(past_key_values, "set_seq_length"):
        past_key_values.set_seq_length(new_len)

    return new_len
