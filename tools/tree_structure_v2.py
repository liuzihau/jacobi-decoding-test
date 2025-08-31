from __future__ import annotations
from typing import Optional, Tuple, List
import torch

class BatchedBeamTree:
    """
    Batched Top-K token dependency tree for Jacobi decoding.

    Shapes (per call):
      B  = batch size
      K  = beams kept per batch (Top-K)
      J  = total Jacobi depths
      Bc = per-parent branching (candidates per beam)

    At each depth d:
      - We keep up to K beams per batch (tokens, scores, parent pointers).
      - Expansion takes [B, K, Bc] children (tokens, step logps, valid mask).
      - Path retrieval works for ANY node at ANY depth (early accept).

    Storage (per depth d = 0..J):
      tokens[d]      : Long [B, K]
      step_logp[d]   : Float [B, K]   (root step_logp is the provided logp0, often 0)
      cum_logp[d]    : Float [B, K]
      parent_beam[d] : Int16 [B, K]   (index 0..K-1 into previous depth; -1 at d=0)
      alive[d]       : Bool [B, K]    (finite score marker)

    Determinism: ties are broken by adding a tiny epsilon times (-flat_index) before topk.
    """

    def __init__(self, batch_size: int, K: int, J: int, device: str | torch.device = "cpu"):
        self.B = int(batch_size)
        self.K = int(K)
        self.J = int(J)
        self.device = torch.device(device)

        # Per-depth storage (filled lazily)
        self.tokens_by_d:  list[torch.Tensor] = []  # [B, K] long
        self.step_by_d:    list[torch.Tensor] = []  # [B, K] float
        self.cum_by_d:     list[torch.Tensor] = []  # [B, K] float
        self.parent_by_d:  list[torch.Tensor] = []  # [B, K] int16
        self.alive_by_d:   list[torch.Tensor] = []  # [B, K] bool

        self.cur_depth: int = -1  # will become 0 after init_roots()

    # ---------------- Build ----------------

    @torch.no_grad()
    def init_roots(self, root_tokens: torch.LongTensor, root_logps: Optional[torch.FloatTensor] = None):
        """
        Initialise depth 0 from root tokens (next AR token for each batch element).

        Args:
          root_tokens : [B] long
          root_logps  : [B] float (optional, default 0)
        """
        assert root_tokens.shape == (self.B,), f"root_tokens should be [B]={self.B}"
        if root_logps is None:
            root_logps = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        else:
            assert root_logps.shape == (self.B,)

        # Allocate depth 0 tensors
        t0 = torch.full((self.B, self.K), 0, dtype=torch.long, device=self.device)
        s0 = torch.full((self.B, self.K), float("-inf"), dtype=torch.float32, device=self.device)
        c0 = torch.full((self.B, self.K), float("-inf"), dtype=torch.float32, device=self.device)
        p0 = torch.full((self.B, self.K), -1, dtype=torch.int16, device=self.device)
        a0 = torch.zeros((self.B, self.K), dtype=torch.bool, device=self.device)

        # Put the single real root at beam 0; others inactive
        t0[:, 0] = root_tokens.to(self.device)
        s0[:, 0] = root_logps.to(self.device)
        c0[:, 0] = root_logps.to(self.device)
        a0[:, 0] = True

        self.tokens_by_d  = [t0]
        self.step_by_d    = [s0]
        self.cum_by_d     = [c0]
        self.parent_by_d  = [p0]
        self.alive_by_d   = [a0]
        self.cur_depth = 0

    def pad_ragged_to_K(self, children_tok_A, children_step_A):
        """
        children_tok_A:     [B, A, Bc] long
        children_step_A:    [B, A, Bc] float
        returns: (children_tokens, children_step_logps, valid_mask) all [B, K, Bc]
        """
        B, A, Bc = children_tok_A.shape
        dev = children_tok_A.device

        children_tokens     = torch.empty((B, self.K, Bc), dtype=children_tok_A.dtype,  device=dev)
        children_step_logps = torch.empty((B, self.K, Bc), dtype=children_step_A.dtype, device=dev)
        valid_mask          = torch.zeros((B, self.K, Bc), dtype=torch.bool, device=dev)

        children_tokens[:, :A, :] = children_tok_A
        children_step_logps[:, :A, :] = children_step_A
        valid_mask[:, :A, :] = True
        return children_tokens, children_step_logps, valid_mask


    @torch.no_grad()
    def expand_depth(
        self,
        children_tokens: torch.LongTensor,       # [B, K, Bc]
        children_step_logps: torch.FloatTensor,  # [B, K, Bc]
        valid_mask: Optional[torch.BoolTensor] = None,  # [B, K, Bc]
        prune_to_topk: bool = True,
    ):
        """
        Expand from depth d to d+1 for all batches with vectorised Top-K per batch.

        Notes:
          - We respect previous 'alive' beams automatically: invalid children if parent is dead.
          - If fewer than K valid children exist, dead beams are filled (alive=False, scores=-inf).
        """
        assert self.cur_depth >= 0, "Call init_roots() first."
        B, A, Bc = children_tokens.shape
        assert B == self.B and children_step_logps.shape == children_tokens.shape, "children must be [B, K, Bc] with matching B,K"

        if A != self.K:
            children_tokens, children_step_logps, valid_mask = self.pad_ragged_to_K(children_tokens, children_step_logps)

        if valid_mask is None:
            valid_mask = torch.ones((B, self.K, Bc), dtype=torch.bool, device=self.device)
        else:
            assert valid_mask.shape == (B, self.K, Bc)

        # Parent stats
        prev_alive = self.alive_by_d[self.cur_depth]                      # [B, K]
        prev_cum   = self.cum_by_d[self.cur_depth]                        # [B, K]

        # Disable children of dead parents
        valid_mask = valid_mask & prev_alive.unsqueeze(-1)                # [B, K, Bc]

        # Candidate cumulative scores
        cand_step = torch.where(valid_mask, children_step_logps, torch.full_like(children_step_logps, float("-inf")))
        cand_cum  = prev_cum.unsqueeze(-1) + cand_step                    # [B, K, Bc]

        # Flatten per batch for a single topk: [B, K*Bc]
        BKB = self.K * Bc
        cand_flat = cand_cum.view(B, BKB)
        # Deterministic tie-break: favour lower flat indices
        eps = (torch.arange(BKB, device=self.device, dtype=torch.float32) + 1).unsqueeze(0)  # [1,BKB]
        # Add tiny negative epsilon proportional to index (bigger index => slightly worse)
        adj_scores = cand_flat + (-eps * 1e-12)

        keep_k = self.K if prune_to_topk else min(BKB, self.K)  # still emit K rows
        top_vals, top_idx = torch.topk(adj_scores, k=keep_k, dim=1, largest=True, sorted=True)  # [B, K]
        # Recover true (unadjusted) cum scores
        top_cum = cand_flat.gather(1, top_idx)                                             # [B, K]

        # Map back to (parent_beam, child_idx)
        parent_beam = torch.div(top_idx, Bc, rounding_mode='floor').to(torch.int16)        # [B, K]
        child_idx   = (top_idx % Bc).to(torch.int64)                                       # [B, K]

        # Gather kept tokens and step logps from flattened views
        flat_tokens = children_tokens.view(B, BKB)
        flat_steps  = children_step_logps.view(B, BKB)
        kept_tokens = flat_tokens.gather(1, top_idx)                                       # [B, K]
        kept_steps  = flat_steps.gather(1, top_idx)                                        # [B, K]

        # Alive mask for new depth: finite cum scores
        new_alive = torch.isfinite(top_cum)

        # Store depth d+1
        self.tokens_by_d.append(kept_tokens)
        self.step_by_d.append(kept_steps)
        self.cum_by_d.append(top_cum)
        self.parent_by_d.append(parent_beam)
        self.alive_by_d.append(new_alive)
        self.cur_depth += 1

    # ---------------- Query ----------------

    @torch.no_grad()
    def final_topk_handles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (beam_indices, scores) for current depth, sorted best-first per batch.

        Returns:
          beam_idx : [B, K] int64   (indices into current depth's K beams, already topk-sorted)
          scores   : [B, K] float32 (cum logp)
        """
        cur_scores = self.cum_by_d[self.cur_depth]            # [B, K]
        cur_alive  = self.alive_by_d[self.cur_depth]          # [B, K]
        scores = torch.where(cur_alive, cur_scores, torch.full_like(cur_scores, float("-inf")))
        # Already sorted from construction; but re-sort just in case
        vals, idx = torch.topk(scores, k=self.K, dim=1, largest=True, sorted=True)
        return idx.to(torch.int64), vals

    @torch.no_grad()
    def accept(self, batch_idx: int, depth: int, beam_idx: int) -> Tuple[List[int], float]:
        """
        Accept ANY node (early or leaf) and return (token_path, cum_logp).

        Args:
          batch_idx : 0..B-1
          depth     : 0..cur_depth
          beam_idx  : 0..K-1  (index within that depth’s beams)
        """
        assert 0 <= batch_idx < self.B
        assert 0 <= depth <= self.cur_depth
        assert 0 <= beam_idx < self.K

        toks: List[int] = []
        d = depth
        k = beam_idx
        while d >= 0:
            tok = int(self.tokens_by_d[d][batch_idx, k].item())
            toks.append(tok)
            if d == 0:
                break
            k = int(self.parent_by_d[d][batch_idx, k].item())
            d -= 1
        toks.reverse()
        cum = float(self.cum_by_d[depth][batch_idx, beam_idx].item())
        return toks, cum

    @torch.no_grad()
    def materialise_paths(self, depth: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Materialise all K paths per batch up to 'depth' (default = current depth).

        Returns:
          paths    : [B, K, depth+1] long   (token ids, with rubbish where beams are dead)
          steps    : [B, K, depth+1] float  (step logps, root step is logp0)
          cum_last : [B, K] float           (cumulative logp at 'depth')
        """
        if depth is None:
            depth = self.cur_depth
        assert 0 <= depth <= self.cur_depth

        B, K = self.B, self.K
        paths = torch.empty((B, K, depth + 1), dtype=torch.long, device=self.device)
        steps = torch.empty((B, K, depth + 1), dtype=torch.float32, device=self.device)

        # Start from (depth, all K), backtrack to 0
        # We do a small loop over depth (J is small), vectorised over [B,K].
        cur_beam = torch.arange(K, device=self.device).view(1, K).expand(B, K).to(torch.int64)

        for d in range(depth, -1, -1):
            tok_d   = self.tokens_by_d[d].gather(1, cur_beam)   # [B, K]
            step_d  = self.step_by_d[d].gather(1, cur_beam)     # [B, K]
            paths[:, :, d] = tok_d
            steps[:, :, d] = step_d
            if d > 0:
                parent_d = self.parent_by_d[d].to(torch.int64)  # [B, K]
                cur_beam = parent_d.gather(1, cur_beam)         # follow parents for next iteration

        cum_last = self.cum_by_d[depth]
        return paths, steps, cum_last
    
    @torch.no_grad()
    def reset(self):
        """
        In-place reset to an empty tree (keeps B, K, J, device unchanged).
        Next call should be init_roots(...).
        O(1) Python work; lets CUDA caching allocator reuse memory on next expand.
        """
        self.tokens_by_d.clear()
        self.step_by_d.clear()
        self.cum_by_d.clear()
        self.parent_by_d.clear()
        self.alive_by_d.clear()
        self.cur_depth = -1

    @torch.no_grad()
    def reinit(self, *, batch_size: int | None = None, K: int | None = None, J: int | None = None, device=None):
        """
        Change structural params and reset. If you pass nothing, it's equivalent to reset().
        """
        if batch_size is not None: self.B = int(batch_size)
        if K is not None:          self.K = int(K)
        if J is not None:          self.J = int(J)
        if device is not None:     self.device = torch.device(device)
        # wipe state
        self.reset()

    # ---------------- Introspection ----------------

    def depth(self) -> int:
        return self.cur_depth

    def alive_mask(self, depth: Optional[int] = None) -> torch.BoolTensor:
        if depth is None: depth = self.cur_depth
        return self.alive_by_d[depth]
    
    
# ---------- pretty dump helpers ----------
def _dump_depth(tree, d: int):
    B, K = tree.B, tree.K
    toks   = tree.tokens_by_d[d].detach().cpu()
    steps  = tree.step_by_d[d].detach().cpu()
    cum    = tree.cum_by_d[d].detach().cpu()
    alive  = tree.alive_by_d[d].detach().cpu()
    parent = tree.parent_by_d[d].detach().cpu() if d > 0 else torch.full((B, K), -1, dtype=torch.int16)

    print(f"\n=== Depth {d} ===")
    for b in range(B):
        print(f"[batch {b}]")
        print(" beam | parent | token | step_logp  step_prob |   cum_logp   alive")
        for k in range(K):
            sl = float(steps[b, k])
            sp = float(torch.exp(steps[b, k])) if torch.isfinite(steps[b, k]) else 0.0
            cl = float(cum[b, k])
            print(f"  {k:>3} |  {int(parent[b,k]):>6} | {int(toks[b,k]):>5} | {sl:>9.4f}  {sp:>8.4f} | {cl:>10.4f}   {bool(alive[b,k])}")
        print()

def _accept_best_at_depth(tree, depth: int, batch_idx: int):
    scores = tree.cum_by_d[depth]                                     # [B, K]
    vals, idx = torch.topk(scores, k=1, dim=1, largest=True, sorted=True)
    beam = int(idx[batch_idx, 0].item())
    path, cum = tree.accept(batch_idx=batch_idx, depth=depth, beam_idx=beam)
    print(f"[accept] batch={batch_idx}, depth={depth}, best_beam={beam} -> path={path}, cum_logp={cum:.4f}")
    return path, cum

# ---------- deterministic test ----------
def test_tree_deterministic():
    torch.manual_seed(0)

    # Small, easy-to-read sizes
    B, K, J, Bc = 2, 3, 3, 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Your BatchedBeamTree class should be in scope
    tree = BatchedBeamTree(batch_size=B, K=K, J=J, device=device)

    # Depth 0 (roots)
    t0  = torch.tensor([10, 20], dtype=torch.long, device=device)
    lp0 = torch.zeros(B, dtype=torch.float32, device=device)
    tree.init_roots(t0, lp0)
    _dump_depth(tree, 0)

    # Depth 1: only beam-0 is alive initially; others will be auto-masked
    children_tok_1     = torch.zeros((B, K, Bc), dtype=torch.long,  device=device)
    children_steplog_1 = torch.full((B, K, Bc), float("-inf"),      dtype=torch.float32, device=device)

    # Batch 0: root expands to {101:0.6, 102:0.4}
    children_tok_1[0, 0]     = torch.tensor([101, 102], device=device)
    children_steplog_1[0, 0] = torch.log(torch.tensor([0.6, 0.4], dtype=torch.float32, device=device))

    # Batch 1: root expands to {201:0.55, 202:0.45}
    children_tok_1[1, 0]     = torch.tensor([201, 202], device=device)
    children_steplog_1[1, 0] = torch.log(torch.tensor([0.55, 0.45], dtype=torch.float32, device=device))

    valid_mask_1 = torch.ones((B, K, Bc), dtype=torch.bool, device=device)
    tree.expand_depth(children_tok_1, children_steplog_1, valid_mask_1, prune_to_topk=True)
    _dump_depth(tree, 1)

    # Depth 2: provide candidates for *each* beam index; dead parents auto-mask
    children_tok_2     = torch.empty((B, K, Bc), dtype=torch.long,  device=device)
    children_steplog_2 = torch.empty((B, K, Bc), dtype=torch.float32, device=device)
    valid_mask_2       = torch.ones( (B, K, Bc), dtype=torch.bool,  device=device)

    # Make per-beam probabilities deterministic & distinct:
    # beam i has probs [p, 1-p] with p = clip(0.75 - 0.1*i, 0.55, 0.95)
    for b in range(B):
        for i in range(K):
            p = max(0.55, min(0.95, 0.75 - 0.10 * i))
            probs = torch.tensor([p, 1.0 - p], dtype=torch.float32, device=device)
            children_steplog_2[b, i] = torch.log(probs)
            base = 1000*b + 10*(i+1)
            children_tok_2[b, i] = torch.tensor([base+1, base+2], dtype=torch.long, device=device)

    tree.expand_depth(children_tok_2, children_steplog_2, valid_mask_2, prune_to_topk=True)
    _dump_depth(tree, 2)

    # Early acceptance (depth=2): accept best beam per batch
    _accept_best_at_depth(tree, depth=2, batch_idx=0)
    _accept_best_at_depth(tree, depth=2, batch_idx=1)

    # Depth 3 (final): similar deterministic scheme with a slightly different profile
    children_tok_3     = torch.empty((B, K, Bc), dtype=torch.long,  device=device)
    children_steplog_3 = torch.empty((B, K, Bc), dtype=torch.float32, device=device)
    valid_mask_3       = torch.ones( (B, K, Bc), dtype=torch.bool,  device=device)

    for b in range(B):
        for i in range(K):
            # beam i at depth 3: probs skewed toward second option now
            q = max(0.55, min(0.90, 0.60 + 0.08 * i))
            probs = torch.tensor([1.0 - q, q], dtype=torch.float32, device=device)
            children_steplog_3[b, i] = torch.log(probs)
            base = 2000*b + 100*(i+1)
            children_tok_3[b, i] = torch.tensor([base+7, base+9], dtype=torch.long, device=device)

    tree.expand_depth(children_tok_3, children_steplog_3, valid_mask_3, prune_to_topk=True)
    _dump_depth(tree, 3)

    # Final Top-K handles and full paths
    beam_idx, scores = tree.final_topk_handles()  # [B, K], [B, K]
    paths, steps, cum = tree.materialise_paths()  # [B, K, J+1], [B, K, J+1], [B, K]

    print("\n=== Final best paths (per batch) ===")
    for b in range(B):
        bestk = int(beam_idx[b, 0])
        toks, total = tree.accept(b, depth=tree.depth(), beam_idx=bestk)
        print(f"[batch {b}] best_beam={bestk} path={toks} total_logp={total:.4f}")

    # Optional: show all K paths compactly
    print("\nAll paths tokens [B,K,J+1]:")
    print(paths.cpu())
    print("\nAll paths cumulative logp [B,K]:")
    print(cum.cpu())

if __name__ == "__main__":
    test_tree_deterministic()