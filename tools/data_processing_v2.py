"""
JacobiDataset v2 — clean, scalable dataset builder for self‑draft ("Jacobi") training

Features
- Reads sharded .pt files written by the refactored generator (list[dict] per shard)
  * Also works with single‑record .pt files (dict)
- Supports two scheduling modes:
  1) inline: [prompt, (J jacobi), g0, (J jacobi), g1, ...]
  2) tail:   [prompt, g0..gK-1, (J jacobi per g in order)]
- Emits: input_ids, labels (next tokens), attention_mask (additive mask), loss_mask,
         hidden_state_target (teacher hidden states for Jacobi tokens), filename
- Efficient attention construction via blockwise tensor ops (no Python loops over L^2)
- Pluggable mask strategies; easy to add variants

Conventions
- We treat PAD token as a real token id in input_ids; attention is controlled by an
  additive bias matrix where allowed = 0, masked = -inf
- Loss is only applied on Jacobi token positions (loss_mask=1); at non‑Jacobi positions loss_mask=0
- Labels:
  • normal positions (prompt, g): label = next token in the *normal* stream (i.e., g_k)
  • jacobi positions: follow the rules described by the user

Usage
    ds = JacobiDatasetV2(
        files=["/path/to/shard_00000.pt", "/path/to/shard_00001.pt"],
        max_len=2048,
        jacobi_J=2,
        schedule="inline",      # or "tail"
        pad_id=151643,
        vocab_size=151936,
    )
    collate = JacobiCollatorV2()
    batch = collate([ds[0], ds[1]])

Notes
- For speed/memory, we lazily load a shard on access; we only precompute shard sizes at __init__
- Hidden states are sliced to the subset used as Jacobi targets and concatenated in output
"""
from __future__ import annotations
import math
import os
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ---------------------------
# Helpers for sharded files
# ---------------------------

def _list_pt_files(paths_or_dirs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in paths_or_dirs:
        P = Path(p)
        if P.is_dir():
            out.extend(sorted([str(x) for x in P.glob("**/*.pt")]))
        elif P.is_file():
            out.append(str(P))
        else:
            # glob pattern
            out.extend(sorted([str(x) for x in Path().glob(p)]))
    return out


def _load_record(obj: Any, idx: int = 0) -> Dict[str, Any]:
    if isinstance(obj, list):
        return obj[idx]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported .pt object type: {type(obj)}")


class ShardIndex:
    """Map global sample index -> (shard_file, local_idx). Lazily loads shards when needed."""
    def __init__(self, files: Sequence[str]):
        self.files = list(files)
        self.starts: List[int] = []  # cumulative start index per shard
        self.lengths: List[int] = []
        total = 0
        for f in self.files:
            obj = torch.load(f, map_location="cpu")
            n = len(obj) if isinstance(obj, list) else 1
            self.starts.append(total)
            self.lengths.append(n)
            total += n
        self.total = total

    def __len__(self):
        return self.total

    def locate(self, idx: int) -> Tuple[str, int]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        # binary search over starts
        lo, hi = 0, len(self.starts)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.starts[mid] <= idx:
                lo = mid
            else:
                hi = mid
        local = idx - self.starts[lo]
        return self.files[lo], local


# ---------------------------
# Core builders (sequence, labels, masks)
# ---------------------------

def _safe_len(prompt_ids: torch.Tensor, g: torch.Tensor, J: int, max_len: int, schedule: str) -> int:
    """Compute how many g tokens we can use under max_len for given schedule and J."""
    P = int(prompt_ids.numel())
    K_all = int(g.numel())
    if schedule == "inline":
        # seq length = P + K * (J + 1)
        max_K = max(0, (max_len - P) // (J + 1))
        return min(K_all, max_K)
    elif schedule == "tail":
        # seq length = P + K + J*K = P + K*(J+1)
        max_K = max(0, (max_len - P) // (J + 1))
        return min(K_all, max_K)
    else:
        raise ValueError(f"Unknown schedule {schedule}")


def _build_inline(prompt: torch.Tensor, g: torch.Tensor, J: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (seq_ids, labels, loss_mask).
    Layout: [prompt, (J jacobi), g0, (J jacobi), g1, ...]
    Labels: for block k: J jacobi labels = g[k : k+J], g label = g[k]
    Loss only on jacobi positions.
    """
    P = prompt.numel()
    K = g.numel()
    EFF_K = effective_g_count(K, J)
    g = g.view(-1)

    # Build stacked jacobi placeholders (use PAD, true id doesn't matter since loss_mask gates)
    # We'll fill input_ids with PAD at jacobi positions.
    blocks = []
    jacobi_inputs = []
    label_blocks = []
    loss_blocks = []

    for k in range(EFF_K + 1):
        # jacobi labels g[k : k+J], pad with last available if short
        j_inputs = g[k : min(k + J, K)]
        j_labels = g[k + 1 : min(k + J + 1, K)]
        if j_labels.numel() < J:
            raise Exception(f"jacobi labels({j_labels}) doesn't match the jacobi token nums ({J})")
        # block tokens: J PADs then g_k
        block_ids = torch.empty(J + 1 if k != EFF_K else J, dtype=prompt.dtype)
        block_ids = 0  # will be replaced with PAD later by caller
        block_ids[-1] = g[k] if k != EFF_K else 0
        blocks.append(block_ids)

        jacobi_inputs.append(j_inputs)

        # labels: J jacobi (j_labels), then label for g_k is next token g[k] -> g[k+1]
        label_blocks.append(j_labels)
        if k != EFF_K:
            g_label = g[min(k + 1, K)]
            label_blocks.append(g_label)

        # loss mask: 1 for J jacobi, 0 for the g position
        lb = torch.zeros(J + 1 if k != EFF_K else J, dtype=torch.int64)
        lb[:-1] = 1
        lb[-1] = 0 if k != EFF_K else 1
        loss_blocks.append(lb)

    blocks = torch.cat(blocks) if blocks else torch.empty(0, dtype=prompt.dtype)
    jacobi_inputs = torch.cat(jacobi_inputs) if jacobi_inputs else torch.empty(0, dtype=g.dtype)
    labels = torch.cat(label_blocks) if label_blocks else torch.empty(0, dtype=g.dtype)
    loss_mask = torch.cat(loss_blocks) if loss_blocks else torch.empty(0, dtype=torch.int64)

    seq_ids = torch.cat([prompt, blocks])
    return seq_ids, labels, jacobi_inputs, loss_mask

def effective_g_count(K: int, J: int):
    """Max number of g tokens usable for Jacobi labeling without padding."""
    return K - J - 1

def _build_tail(prompt: torch.Tensor, g: torch.Tensor, J: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (seq_ids, labels, loss_mask).
    Layout: [prompt, g0..gK-1, (for k=0..K-1) J jacobi tokens tied to g_k]
    Labels: jacobi labels for group k are g[k : k+J] (padded at end); normal positions
            (prompt+g) are labeled with next normal token.
    """
    P = prompt.numel()
    K = g.numel()
    EFF_K = effective_g_count(K, J)
    # base sequence: prompt + all g
    g = g.view(-1)
    seq_ids = torch.cat([prompt, g[:EFF_K]])

    # labels for base tokens (causal next-token for the g stream). For prompt positions,
    # we set labels to the next normal token (g[0] for all prompt positions), but mask loss.
    base_labels = torch.empty(P + EFF_K, dtype=g.dtype)
    if K > 0:
        base_labels[:P] = 0
        base_labels[P - 1 : P + EFF_K] = g[:EFF_K + 1]
    else:
        base_labels.fill_(0)

    base_loss = torch.zeros(P + EFF_K, dtype=torch.int64)

    # append jacobi placeholders and labels per g_k
    jacobi_ids = []
    jacobi_inputs = []
    jacobi_labels = []
    jacobi_loss = []
    for k in range(EFF_K + 1):
        j_inputs = g[k : min(k + J, K)]
        j_labels = g[k + 1 : min(k + J + 1, K)]
        if j_labels.numel() < J:
            raise Exception(f"jacobi labels({j_labels}) doesn't match the jacobi token nums ({J})")
        jacobi_ids.append(torch.zeros(J, dtype=seq_ids.dtype))  # PAD placeholder
        jacobi_inputs.append(j_inputs)
        jacobi_labels.append(j_labels)
        jacobi_loss.append(torch.ones(J, dtype=torch.int64))

    if jacobi_ids:
        jacobi_ids = torch.cat(jacobi_ids)
        seq_ids = torch.cat([seq_ids, jacobi_ids])
        jacobi_inputs = torch.cat(jacobi_inputs)
        jacobi_labels = torch.cat(jacobi_labels)
        labels = torch.cat([base_labels, jacobi_labels])
        jacobi_loss = torch.cat(jacobi_loss)
        loss_mask = torch.cat([base_loss, jacobi_loss])
    else:
        labels = base_labels
        loss_mask = base_loss

    return seq_ids, labels, jacobi_inputs, loss_mask


def _mask_inline(Lp: int, K: int, J: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build additive attention bias for inline schedule.
    Rules (approximate the user examples):
      - Base: strictly causal lower‑triangular over the entire sequence
      - g_k cannot attend to jacobi tokens from its own block (only prompt + past g)
      - jacobi tokens in block k can attend to prompt + previous jacobi within the block + past g
      - jacobi tokens cannot see jacobi tokens from *earlier* blocks
    """
    L = Lp + K * (J + 1)
    neg_inf = torch.finfo(dtype).min
    bias = torch.full((L, L), fill_value=neg_inf, dtype=dtype, device=device)
    # causal allow
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    bias = bias.masked_fill(causal, 0)

    # mask g_k from its own block jacobi
    for k in range(K):
        start = Lp + k * (J + 1)
        j_slice = slice(start, start + J)
        g_pos = start + J
        bias[g_pos, start : g_pos] = neg_inf  # zeroed by default causal; re‑mask to neg_inf
        # allow g_k to attend prompt + previous g only (already allowed by causal)

        # prevent jacobi in block k from seeing jacobi in previous blocks
        if k > 0:
            prev_j_end = Lp + (k - 1) * (J + 1) + J
            bias[start : start + J, Lp : prev_j_end] = neg_inf
    return bias


def _mask_tail(Lp: int, K: int, J: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build additive attention bias for tail schedule.
    Rules (approximate examples):
      - Base segment [prompt + g] uses causal mask.
      - For each group k in the tail jacobi section, J jacobi tokens can attend to prompt + g[:k+1]
        and to previous jacobi *within the same group* (j0->none, j1->j0, ...).
      - Jacobi of group k cannot see jacobi of other groups.
    """
    EFF_K = effective_g_count(K, J)
    L_base = Lp + EFF_K
    L = L_base + (EFF_K + 1) * J
    neg_inf = torch.finfo(dtype).min
    bias = torch.full((L, L), fill_value=neg_inf, dtype=dtype, device=device)

    # causal for the base part
    causal_base = torch.tril(torch.ones(L_base, L_base, dtype=torch.bool, device=device))
    bias[:L_base, :L_base] = torch.where(causal_base, torch.tensor(0, dtype=dtype, device=device), torch.tensor(neg_inf, dtype=dtype, device=device))

    # tail groups
    for k in range(EFF_K + 1):
        # rows for jacobi tokens of group k
        row_start = L_base + k * J
        row_end = row_start + J
        # allow attending to prompt + g[:k+1]
        bias[row_start:row_end, : (Lp + k + 1)] = 0
        # within-group lower triangular (j can see earlier j in same group)
        tri = torch.tril(torch.ones(J, J, dtype=torch.bool, device=device))
        bias[row_start:row_end, row_start:row_end] = torch.where(tri, torch.tensor(0, dtype=dtype, device=device), torch.tensor(neg_inf, dtype=dtype, device=device))
        # everything else remains masked
    return bias


# ---------------------------
# Dataset
# ---------------------------
class JacobiDatasetV2(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        max_len: int = 2048,
        jacobi_J: int = 2,
        schedule: str = "inline",  # "inline" or "tail"
        pad_id: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.files = _list_pt_files([files])
        # [int(len(files) * portion_s):int(len(files) * portion_e) if portion_e < 1 else None]
        if not self.files:
            raise ValueError("No .pt files found.")
        self.index = ShardIndex(self.files)
        self.max_len = max_len
        self.J = jacobi_J
        assert schedule in {"inline", "tail"}
        self.schedule = schedule
        self.pad_id = pad_id
        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        elif dtype == "no":
            dtype = torch.float32
        self.bias_dtype = dtype

    def __len__(self):
        return len(self.index)

    def _fetch(self, idx: int) -> Dict[str, Any]:
        f, local = self.index.locate(idx)
        obj = torch.load(f, map_location="cpu")
        rec = _load_record(obj, local)
        # Support both new and legacy keys
        g_key = "generated_next" if "generated_next" in rec else "generated_tokens"
        h_key = "hidden" if "hidden" in rec else "hidden_states"
        return {
            "input_ids": rec["input_ids"],
            "g": rec[g_key],
            "hidden": rec[h_key],
            "filename": f"{f}#{local}-{idx}",
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._fetch(idx)
        prompt = r["input_ids"].to(torch.long)
        g_all = r["g"].to(torch.long)
        hidden_all = r["hidden"].to(self.bias_dtype)  # training usually wants fp32

        # Respect length budget by trimming g
        K = _safe_len(prompt, g_all, self.J, self.max_len, self.schedule)
        g = g_all[:K]
        hidden = hidden_all[:K]  # hidden aligned with generated steps

        if self.schedule == "inline":
            seq, labels, loss_mask = _build_inline(prompt, g, self.J)
            Lp = prompt.numel()
            bias = _mask_inline(Lp, K, self.J, self.bias_dtype, device=hidden.device)
            # select hidden targets: jacobi positions only, per block take J rows from hidden starting at k
            # For block k, jacobi labels map to hidden[k : k+J]
            H_list = []
            for k in range(K):
                end = min(k + self.J, K)
                h = hidden[k:end]
                if h.shape[0] < self.J:
                    h = torch.cat([h, h[-1:].expand(self.J - h.shape[0], -1)])
                H_list.append(h)
            H = (
                torch.cat(H_list, dim=0)
                if H_list
                else torch.empty(0, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
            )
        else:  # tail
            seq, labels, jacobi_inputs, loss_mask = _build_tail(prompt, g, self.J)
            Lp = prompt.numel()
            bias = _mask_tail(Lp, K, self.J, self.bias_dtype, device=hidden.device)
            # Hidden targets per group k: hidden[k : k+J] padded
            H_list = []
            EFF_K = effective_g_count(K, self.J)
            for k in range(1, EFF_K + 1 + 1):  # 94 + 2 = 96  --> 95, 96, 97, 98, 99
                end = min(k + self.J, K)
                h = hidden[k:end]
                if h.shape[0] < self.J:
                    h = torch.cat([h, h[-1:].expand(self.J - h.shape[0], -1)])
                H_list.append(h)
            H = (
                torch.cat(H_list, dim=0)
                if H_list
                else torch.empty(0, hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
            )

        # Replace jacobi placeholders in seq with pad_id
        jacobi_pos = (loss_mask == 1).nonzero(as_tuple=True)[0]
        seq = seq.clone()
        seq[jacobi_pos] = self.pad_id

        return {
            "input_ids": seq,                   # [L]
            "labels": labels,                   # [L]
            "jacobi_inputs": jacobi_inputs,     # [G * J]
            "attention_mask": bias,             # [L, L], 0 or -inf
            "loss_mask": loss_mask.to(self.bias_dtype),  # [L]
            "hidden_state_target": H,           # [J*K, H]
            "filename": r["filename"],
        }

class TestDataset(Dataset):
    def __init__(
            self,
            data,
            tokenizer,
            num_jacobi_tokens,
            pad_token_id,
            env_prompt="",
            user_prompt="",
            assistant_prompt=""
    ):
        self.data = data
        self.num_jacobi_tokens = num_jacobi_tokens
        self.pad_token_id = pad_token_id
        self.jacobi_tokens = [pad_token_id for _ in range(num_jacobi_tokens)]
        self.tokenizer = tokenizer
        self.env_prompt = env_prompt
        self.user_prompt = user_prompt
        self.assistant_prompt = assistant_prompt
    
    def __getitem__(self, idx):
        prompt = "<System>\n" + self.env_prompt + "\n"
        prompt += "<User>\n" + self.user_prompt + self.data[idx]['prompt'] + "\n"
        prompt += "<Assistant>\n" + self.assistant_prompt
        input_ids = self.tokenizer.encode(prompt) 
        loss_mask = [0 for _ in range(len(input_ids))] + [1 for _ in range(len(self.jacobi_tokens))]
        input_ids += self.jacobi_tokens
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "prompt": prompt
            }
    
    def __len__(self):
        return len(self.data)
                

class TestDatasetFromPT(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        tokenizer,
        jacobi_J: int = 2,
        pad_id: int = 0,
    ):
        self.files = _list_pt_files([files])
        # [int(len(files) * portion_s):int(len(files) * portion_e) if portion_e < 1 else None]
        if not self.files:
            raise ValueError("No .pt files found.")
        self.index = ShardIndex(self.files)
        self.tokenizer = tokenizer
        self.J = jacobi_J
        self.pad_id = pad_id

    def __len__(self):
        return len(self.index)

    def _fetch(self, idx: int) -> Dict[str, Any]:
        f, local = self.index.locate(idx)
        obj = torch.load(f, map_location="cpu")
        rec = _load_record(obj, local)
        return {
            "input_ids": rec["input_ids"],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self._fetch(idx)
        input_ids = r["input_ids"].tolist()
        loss_mask = [0 for _ in range(len(input_ids))] + [1 for _ in range(self.J)]
        input_ids += self.J * [self.pad_id]
        prompt = self.tokenizer.decode(input_ids)

        return {
            "input_ids": input_ids,                   # [L]
            "loss_mask": loss_mask,
            "prompt": prompt
        }


# ---------------------------
# Collator
# ---------------------------
class JacobiCollatorV2:
    def __init__(self, pad_id: int | None = None):
        self.pad_id = pad_id

    def __call__(self, feats: List[Dict[str, Any]]) -> Dict[str, Any]:
        # pad 1D sequences
        def pad1(key: str, pad_val: int | float) -> torch.Tensor:
            arr = [f[key] for f in feats]
            L = max(int(x.numel()) for x in arr)
            out = torch.full((len(arr), L), fill_value=pad_val, dtype=arr[0].dtype)
            for i, x in enumerate(arr):
                out[i, : x.numel()] = x
            return out

        # pad [L,L] attention bias
        def pad2_square(key: str, pad_fill: float) -> torch.Tensor:
            arr = [f[key] for f in feats]
            L = max(x.shape[0] for x in arr)
            out = torch.full((len(arr), L, L), fill_value=pad_fill, dtype=arr[0].dtype)
            for i, x in enumerate(arr):
                l = x.shape[0]
                out[i, :l, :l] = x
            return out

        input_ids = pad1("input_ids", self.pad_id if self.pad_id is not None else 0)
        labels = pad1("labels", -100)  # standard ignore_index
        jacobi_inputs = torch.cat([f["jacobi_inputs"] for f in feats], dim=0)  # standard ignore_index
        loss_mask = pad1("loss_mask", 0.0)

        # attention bias uses fill=-inf for masked regions
        neg_inf = torch.finfo(loss_mask.dtype).min
        attention_mask = pad2_square("attention_mask", pad_fill=neg_inf)

        # pack hidden_state_target (already [J*K_i, H]) -> stack along batch with offsets
        H_list = [f["hidden_state_target"] for f in feats]
        hidden_state_target = (
            torch.cat(H_list, dim=0) if H_list else torch.empty(0, dtype=loss_mask.dtype, device=loss_mask.device)
        ).to(loss_mask.dtype)

        return {
            "input_ids": input_ids.long(),
            "labels": labels.long(),
            "jacobi_inputs": jacobi_inputs.long(),
            "attention_mask": attention_mask,  # [B, L, L]
            "loss_mask": loss_mask,            # [B, L]
            "hidden_state_target": hidden_state_target,  # [sum_i J*K_i, H]
            "filenames": [f["filename"] for f in feats],
        }
    

def collate_data(feats: List[Dict[str, Any]], pad_id: int=None) -> Dict[str, Any]:
    # pad 1D sequences
    def pad1(key: str, pad_val: int | float) -> torch.Tensor:
        arr = [f[key] if isinstance(f[key], torch.Tensor) else torch.tensor(f[key]) for f in feats]
        L = max((int(x.numel()) for x in arr))
        out = torch.full((len(arr), L), fill_value=pad_val, dtype=arr[0].dtype)
        for i, x in enumerate(arr):
            out[i, : x.numel()] = x
        return out
    input_ids = pad1("input_ids", pad_id if pad_id is not None else 0)

    loss_mask = pad1("loss_mask", 0)

    prompts = [f["prompt"] for f in feats]
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "prompts": prompts
    }

def handle_openai_humaneval(dataset_name, tokenizer, J, pad_token_id):
    env_prompt = "You are a helpful AI assistant, your goal is trying to solve the question user asked.\n"
    user_prompt = "I'll give you a function name. You need to help me implement the function.\nA description will be provided below the function name. Also some examples will be given below.\n\n"
    assistant_prompt = ""
    data = load_dataset(dataset_name, split="test")
    return TestDataset(data, tokenizer, J, pad_token_id, env_prompt, user_prompt, assistant_prompt)

def handle_shareGPT(dataset_name, tokenizer, J, pad_token_id):
    return TestDatasetFromPT(dataset_name, tokenizer, J, pad_id=pad_token_id)

def load_all_test_datasets(path_list:List, tokenizer, J, pad_token_id):
    all_datasets = {}
    for path in path_list:
        if path == "openai/openai_humaneval":
            d = handle_openai_humaneval(path, tokenizer, J, pad_token_id)
        elif path == "./datasets/anon8231489123_ShareGPT_Vicuna_unfiltered_Qwen3-1.7B_GE2/test":
            d = handle_shareGPT(path, tokenizer, J, pad_token_id)
        all_datasets[path.split("/")[-1]] = d
    return all_datasets

# ---------------------------
# Test Helper
# ---------------------------
def show_attn_window(attn_bias: torch.Tensor, R: int = 2000, C: int = 2000, save_path: str = "./attn_mask.png"):
    L = attn_bias.shape[0]
    r0, c0 = max(0, L-R), max(0, L-C)
    window = attn_bias[r0:L, c0:L]
    img = (window == 0).to(torch.float32)  # 1 where allowed
    plt.figure(figsize=(6, 6), dpi=600)
    plt.imshow(img.numpy(), aspect="auto", origin="upper")
    plt.title(f"Allowed attention (last {R} rows × {C} cols)")
    plt.xlabel(f"cols {c0}..{L-1}")
    plt.ylabel(f"rows {r0}..{L-1}")
    plt.savefig(save_path, dpi=600)


def testJacobiDatasetV2(path):
    files = f"{path}/shard_00016.pt"
      
    ds = JacobiDatasetV2(
        files=files,
        max_len=2048,
        jacobi_J=3,            # how many Jacobi tokens per step
        schedule="tail",       # or "inline"
        pad_id=151643,         # tokenizer.pad_token_id
        vocab_size=151936,     # optional, not critical here
    )

    collate = JacobiCollatorV2(pad_id=151643)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    for b in tqdm(loader):
        if "#252" not in b["filenames"][0]:
            continue
        # pass
        # batch = next(iter(loader))
        
        print(f"input_ids")
        print(f"\tshape: {b['input_ids'].shape}")        # [B, L]
        print(f"labels")
        print(f"\tshape: {b['labels'].shape}")   # [B, L]     
        print(f"jacobi_inputs")
        print(f"\tshape: {b['jacobi_inputs'].shape}")   # [B, L]           
        print(f"attention mask")
        print(f"\tshape: {b['attention_mask'].shape}")   # [B, L, L]
        show_attn_window(b["attention_mask"][0], save_path="./attn_mask0.png")
        # show_attn_window(b["attention_mask"][1], save_path="./attn_mask1.png")
        # print(f"\tshape:{(batch["attention_mask"][0, -20:, -20:] / batch["attention_mask"].min()).to(torch.int32)}")   # [B, L, L]
        print(f"loss mask")
        print(f"\tshape: {b['loss_mask'].shape}")
        # print(f"\t{batch['loss_mask'][:, -300:]}")        # [B, L]
        print(f"hidden state target")
        print(f"\tshape: {b['hidden_state_target'].shape}")  # [sum_i J*K_i, H]

        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-1.7B", use_fast=False)

        jac_cols = torch.nonzero(b["loss_mask"][0] == 1, as_tuple=False).flatten()
        l = ""
        for i, tk in enumerate(b['labels'][0][jac_cols]):
            d = tokenizer.decode(tk)
            d = d.replace("\n", "\\n")
            l += f"[{i}|{d}] "
        print(l)
        print("=" * 80)
        l = ""
        for i, tk in enumerate(b['jacobi_inputs'][0]):
            d = tokenizer.decode(tk)
            d = d.replace("\n", "\\n")
            l += f"[{i}|{d}] "
        print(l)
        break
    
if __name__ == "__main__":
    testJacobiDatasetV2("/home/tliu0205/jacobi-decoding-test/datasets/anon8231489123_ShareGPT_Vicuna_unfiltered_Qwen3-1.7B_GE2/train")