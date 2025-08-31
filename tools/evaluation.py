from typing import Optional, Tuple
import torch


class WordCounter:
    def __init__(self, J, V, K, dev):
        self.J = J
        self.V = V
        self.K = K
        self.dev = dev
        self.epoch_count = []
        self.current_count = None
        self.work_dtype = torch.float16
    
    def initialize_new_count(self):
        self.current_count = torch.zeros((self.J, self.V), dtype=torch.int32, device=self.dev)
    
    def update_count(self, logits):
        # --- Efficient top-K counting over jacobi logits (OOM-safe) ---
        with torch.no_grad():
            N, _ = logits.shape
            dev = logits.device
            if self.current_count.device != dev:
                self.current_count = self.current_count.to(dev)
            try:
                free_mem, _ = torch.cuda.mem_get_info(dev)
                bytes_per_row = self.V * (2 if self.work_dtype == torch.float16 else 4)
                rows_per_chunk = max(64, min(N, int(0.20 * free_mem // bytes_per_row)))
            except Exception:
                rows_per_chunk = 512

            row_ids = torch.arange(N, device=dev)

            for s in range(0, N, rows_per_chunk):
                e = min(N, s + rows_per_chunk)
                rids = row_ids[s:e]                 # [R]
                token_ids = (rids % self.J)              # [R] → which Jacobi token each row belongs to

                # top-K indices only (no need for values; no full sort)
                topk_idx = torch.topk(
                    logits[s:e].to(self.work_dtype), k=self.K, dim=-1, largest=True, sorted=False
                ).indices                           # [R, K]

                # In-place sparse accumulation: counts[token_ids, topk_idx] += 1
                self.current_count.index_put_(
                    (token_ids.repeat_interleave(self.K), topk_idx.reshape(-1)),
                    torch.ones(topk_idx.numel(), dtype=self.current_count.dtype, device=self.dev),
                    accumulate=True
                )
    def append_curr_count(self):
        self.epoch_count.append(self.current_count)
        self.current_count = None

    def save(self, path):
        torch.save(torch.stack(self.epoch_count, dim=0), path)



@torch.no_grad()
class Evaluator:
    """
    Tracks:
      - correct_counts[k, j]: #times the j-th Jacobi token's target was in Top-(k+1)
      - totals_per_j[j]:      #examples accumulated for position j
      - alpha_sum[j]:         sum of alpha(p,q) over examples at position j
      - alpha_count[j]:       #examples for alpha at position j (==totals_per_j)
    """
    def __init__(self, J: int, K: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32):
        self.J = int(J)
        self.K = int(K)
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self):
        dev = self.device if self.device is not None else torch.device("cpu")
        self.correct_counts = torch.zeros(self.K, self.J, device=dev, dtype=torch.long)  # [K, J]
        self.totals_per_j  = torch.zeros(self.J, device=dev, dtype=torch.long)           # [J]
        self.alpha_sum     = torch.zeros(self.J, device=dev, dtype=self.dtype)           # [J]
        self.alpha_count   = torch.zeros(self.J, device=dev, dtype=torch.long)           # [J]

    def to(self, device: torch.device):
        self.device = device
        self.correct_counts = self.correct_counts.to(device)
        self.totals_per_j  = self.totals_per_j.to(device)
        self.alpha_sum     = self.alpha_sum.to(device)
        self.alpha_count   = self.alpha_count.to(device)
        return self

    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure logits are [B, J, V]. If [B*J, V], reshape using self.J.
        """
        if x.dim() == 3:
            return x
        elif x.dim() == 2:
            BJ, V = x.shape
            assert BJ % self.J == 0, f"Cannot reshape logits of shape {x.shape} with J={self.J}"
            B = BJ // self.J
            return x.view(B, self.J, V)
        else:
            raise ValueError(f"Expected logits with dim 2 or 3, got {x.shape}")

    def update(self, jacobi_logits: torch.Tensor, target_logits: torch.Tensor):
        """
        Accumulate metrics for a batch.

        Args:
          jacobi_logits: [B, J, V] or [B*J, V]
          target_logits: [B, J, V] or [B*J, V]
        """
        # Move to evaluator device (without changing dtype for logits)
        dev = self.device if self.device is not None else jacobi_logits.device
        jl = self._ensure_3d(jacobi_logits).to(dev, non_blocking=True)   # [B, J, V]
        tl = self._ensure_3d(target_logits).to(dev, non_blocking=True)   # [B, J, V]
        B, J, V = jl.shape
        assert J == self.J, f"J mismatch: got {J}, expected {self.J}"

        # ---------- Top-k correctness per position ----------
        # target indices: [B, J]
        target_idx = tl.argmax(dim=-1)
        # top-k predictions: [B, J, K]
        top_vals, top_idx = jl.topk(self.K, dim=-1, largest=True, sorted=True)
        # matches per (b,j,k): [B, J, K] boolean
        matches = (top_idx == target_idx.unsqueeze(-1))
        # inclusive Top-k → cumulative OR along k (use cumsum then clamp to 1)
        incl_topk = matches.cumsum(dim=-1).clamp_max_(1)                 # [B, J, K]
        # sum over batch → [J, K], then transpose to [K, J]
        batch_counts = incl_topk.sum(dim=0).to(dtype=torch.long).transpose(0, 1)  # [K, J]
        self.correct_counts += batch_counts

        # totals per j (each j got B examples this update)
        self.totals_per_j += torch.full((self.J,), B, device=dev, dtype=torch.long)

        # ---------- Alpha overlap per position ----------
        # probs: [B, J, V]
        p = torch.softmax(jl, dim=-1)
        q = torch.softmax(tl, dim=-1)
        alpha = torch.minimum(p, q).sum(dim=-1)  # [B, J]
        # sum over batch for each j
        self.alpha_sum   += alpha.sum(dim=0).to(self.dtype)
        self.alpha_count += torch.full((self.J,), B, device=dev, dtype=torch.long)

    def compute_aplha(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          topk_counts: [K, J] long – raw correct counts
          alpha_mean:  [J]    float – mean alpha per Jacobi position
        """
        # Avoid divide-by-zero
        alpha_den = torch.clamp(self.alpha_count.to(self.alpha_sum.dtype), min=1)
        return self.alpha_sum / alpha_den

    def compute_K_accuracy(self) -> torch.Tensor:
        """
        Convenience: inclusive Top-k accuracy matrix [K, J] as floats.
        """
        den = torch.clamp(self.totals_per_j.to(self.correct_counts.dtype), min=1)  # [J]
        return (self.correct_counts.to(torch.float32) / den.unsqueeze(0))          # [K, J]
