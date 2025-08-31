from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch

@dataclass
class InputPack:
    # Core model inputs
    input_ids: torch.LongTensor          # [B, L]
    cache_position: torch.LongTensor     # [B, L]
    loss_mask: torch.BoolTensor          # [B, L]
    attn_mask: torch.FloatTensor         # [B, L, L]  (0 / -inf)

    # Indices for JP & adapter fusion
    jacobi_indices: torch.LongTensor     # [2, N_j]   (row0=batch, row1=seq_idx)
    cat_indices: torch.LongTensor        # [B, Nj_max, S]
    cat_mask: torch.BoolTensor           # [B, Nj_max, S]

    # Tree-to-sequence mapping (critical for fast verify)
    # real_pos[b, d, k] = start index of (real token) for node at depth d, beam k; -1 if dead
    real_pos: torch.LongTensor           # [B, D+1, K]
    # jp1_pos[b, d, k] = parent's first JP row used to verify depth-d token; -1 if invalid (d=0 or parent dead)
    jp1_pos: torch.LongTensor            # [B, D+1, K]
    # alive masks copied for convenience
    alive_by_d: List[torch.BoolTensor]   # len=D+1, each [B, K]
    parent_by_d: List[Optional[torch.Tensor]]  # len=D+1, parent_by_d[d] is [B, K] (None at d=0)


class InputProcessor:
    """
    Builds linear inputs from BatchedBeamTree and records direct logits row
    indices for verification (no reconstruction later).
    """

    def __init__(self, J: int, S: int, JP: int, pad_to_multiple: Optional[int] = None, dense_attn: bool = True):
        self.J = int(J)
        self.S = int(S)
        self.JP = int(JP)
        self.pad_to_multiple = pad_to_multiple
        self.dense_attn = dense_attn  # keep True if your model expects [B,L,L] bias

    @torch.no_grad()
    def build_data(self, tree) -> InputPack:
        dev = tree.tokens_by_d[0].device
        B, K = tree.B, tree.K
        D = tree.cur_depth  # 0..J

        tokens_by_d  = [tree.tokens_by_d[d] for d in range(D + 1)]         # each [B,K]
        alive_by_d   = [tree.alive_by_d[d]  for d in range(D + 1)]         # each [B,K] bool
        parent_by_d  = [None] + [tree.parent_by_d[d].to(torch.int64) for d in range(1, D + 1)]

        # ---- 1) Build per-batch linearisation + real_pos ----
        real_pos = torch.full((B, D + 1, K), -1, dtype=torch.long, device=dev)
        seq_tokens: List[torch.Tensor] = []
        seq_cache:  List[torch.Tensor] = []
        seq_loss:   List[torch.Tensor] = []

        L_per_batch = torch.zeros(B, dtype=torch.long, device=dev)
        for b in range(B):
            toks_out: List[int] = []
            cache_out: List[int] = []
            loss_out: List[bool] = []

            p = 0
            for d in range(D + 1):
                alive_row = alive_by_d[d][b]  # [K]
                toks_row  = tokens_by_d[d][b] # [K]
                for k in range(K):
                    if not bool(alive_row[k]):
                        continue
                    real_pos[b, d, k] = p
                    # real token
                    toks_out.append(int(toks_row[k].item()))
                    cache_out.append(d)
                    loss_out.append(False)
                    # J placeholders
                    for j in range(1, self.J + 1):
                        toks_out.append(self.JP)
                        cache_out.append(d + j)
                        loss_out.append(True)
                    p += (self.J + 1)

            seq_tokens.append(torch.tensor(toks_out, dtype=torch.long, device=dev))
            seq_cache.append(torch.tensor(cache_out, dtype=torch.long, device=dev))
            seq_loss.append(torch.tensor(loss_out, dtype=torch.bool, device=dev))
            L_per_batch[b] = p

        # ---- 2) Padding / stacking to [B,L] ----
        L = int(L_per_batch.max().item()) if B > 0 else 0
        if self.pad_to_multiple and L > 0:
            m = int(self.pad_to_multiple)
            L = ((L + m - 1) // m) * m

        def pad1(x, val, L):
            if x.numel() == L: return x
            return torch.cat([x, torch.full((L - x.numel(),), val, dtype=x.dtype, device=x.device)], dim=0)

        input_ids     = torch.stack([pad1(seq_tokens[b], self.JP, L) for b in range(B)], dim=0) if L > 0 else torch.empty((B,0), dtype=torch.long, device=dev)
        cache_position= torch.stack([pad1(seq_cache[b], -1,   L) for b in range(B)], dim=0) if L > 0 else torch.empty((B,0), dtype=torch.long, device=dev)
        loss_mask     = torch.stack([pad1(seq_loss[b],  0,    L) for b in range(B)], dim=0) if L > 0 else torch.empty((B,0), dtype=torch.bool, device=dev)

        # ---- 3) Fast jacobi_indices + cat_indices from real_pos ----
        # JP positions for every alive node: real_pos + [1..J]
        # Build packed list per batch, then pad to Nj_max.
        batch_j_idx: List[torch.Tensor] = []
        batch_cat_idx: List[torch.Tensor] = []
        batch_cat_msk: List[torch.Tensor] = []

        for b in range(B):
            rp = real_pos[b]  # [D+1,K]
            alive_mask = rp >= 0
            if alive_mask.any():
                rp_alive = rp[alive_mask]          # [N_nodes_b]
                # all JP positions for these nodes:
                jp_all = (rp_alive.unsqueeze(1) + torch.arange(1, self.J + 1, device=dev)).reshape(-1)  # [N_nodes_b * J]
                b_ids = torch.full_like(jp_all, b, dtype=torch.long, device=dev)
                batch_j_idx.append(torch.stack([b_ids, jp_all], dim=0))

                # cat indices (S previous within block)
                S = self.S
                offsets = torch.arange(1, S + 1, dtype=torch.long, device=dev)  # [S]
                starts  = rp_alive.repeat_interleave(self.J)                     # [N_j_b]
                preds   = jp_all.unsqueeze(1) - offsets.unsqueeze(0)             # [N_j_b, S]
                valid   = preds >= starts.unsqueeze(1)
                preds   = torch.maximum(preds, starts.unsqueeze(1))
                batch_cat_idx.append(preds)
                batch_cat_msk.append(valid)
            else:
                batch_j_idx.append(torch.empty((2, 0), dtype=torch.long, device=dev))
                batch_cat_idx.append(torch.empty((0, self.S), dtype=torch.long, device=dev))
                batch_cat_msk.append(torch.empty((0, self.S), dtype=torch.bool, device=dev))

        jacobi_indices = torch.cat(batch_j_idx, dim=1) if B > 0 else torch.empty((2, 0), dtype=torch.long, device=dev)

        Nj_max = max((x.size(0) for x in batch_cat_idx), default=0)
        if Nj_max > 0:
            def pad_cat(xi, xm):
                n = xi.size(0)
                if n == Nj_max: return xi, xm
                pad_i = torch.zeros((Nj_max - n, self.S), dtype=xi.dtype, device=dev)
                pad_m = torch.zeros((Nj_max - n, self.S), dtype=xm.dtype, device=dev)
                return torch.cat([xi, pad_i], dim=0), torch.cat([xm, pad_m], dim=0)
            cat_indices = torch.stack([pad_cat(batch_cat_idx[b], batch_cat_msk[b])[0] for b in range(B)], dim=0)
            cat_mask    = torch.stack([pad_cat(batch_cat_idx[b], batch_cat_msk[b])[1] for b in range(B)], dim=0)
        else:
            cat_indices = torch.empty((B, 0, self.S), dtype=torch.long, device=dev)
            cat_mask    = torch.empty((B, 0, self.S), dtype=torch.bool, device=dev)

        # ---- 4) Build jp1_pos for fast verify: position of parent's first JP for each depth-d token ----
        # For depth 0: invalid (-1). For d>=1: jp1 = real_pos at (d-1, parent_k) + 1
        jp1_pos = torch.full_like(real_pos, -1)
        for d in range(1, D + 1):
            par = parent_by_d[d]         # [B,K]
            rp_parent = real_pos[:, d - 1, :]  # [B,K] positions of parents
            # gather the parent real_pos at the parent beam indices
            # Build index: for each (b,k): parent_beam = par[b,k]
            b_idx = torch.arange(B, device=dev).unsqueeze(1).expand(B, K)  # [B,K]
            parent_real = rp_parent[b_idx, par.clamp_min(0)]               # [B,K], garbage where parent dead
            jp1_pos[:, d, :] = torch.where((par >= 0) & (parent_real >= 0), parent_real + 1, torch.full_like(parent_real, -1))

        # ---- 5) Attention mask (optional dense)
        if L == 0 or not self.dense_attn:
            attn_mask = torch.empty((B, L, L), dtype=torch.float32, device=dev)
        else:
            attn_mask = torch.full((B, L, L), float("-inf"), dtype=torch.float32, device=dev)
            for b in range(B):
                p = 0
                # For each alive node (d,k) in BFS order, allow path + intra-block causal.
                # To recover ancestor path quickly: climb parents on the fly (D is small).
                # Build map from (d,k) -> its ancestors' real positions.
                for d in range(D + 1):
                    alive_row = alive_by_d[d][b]
                    for k in range(K):
                        if not bool(alive_row[k]): continue
                        r0 = int(real_pos[b, d, k].item())
                        # Collect path real positions
                        cols: List[int] = []
                        dd, kk = d, k
                        while dd >= 0:
                            if real_pos[b, dd, kk] < 0: break
                            cols.append(int(real_pos[b, dd, kk].item()))
                            if dd == 0: break
                            kk = int(parent_by_d[dd][b, kk].item())
                            dd -= 1
                        cols.sort()
                        # row for real token
                        attn_mask[b, r0, cols] = 0.0
                        # rows for JPs in this block
                        for j in range(1, self.J + 1):
                            rj = r0 + j
                            attn_mask[b, rj, cols] = 0.0
                            # intra-block previous JPs
                            if j > 0:
                                attn_mask[b, rj, slice(r0 + 1, rj + 1)] = 0.0

            # mask padded rows/cols (already -inf by construction beyond L_b, since we never wrote them)

        return InputPack(
            input_ids=input_ids,
            cache_position=cache_position,
            loss_mask=loss_mask,
            attn_mask=attn_mask,
            jacobi_indices=jacobi_indices,
            cat_indices=cat_indices,
            cat_mask=cat_mask,
            real_pos=real_pos,
            jp1_pos=jp1_pos,
            alive_by_d=alive_by_d,
            parent_by_d=parent_by_d,
        )
