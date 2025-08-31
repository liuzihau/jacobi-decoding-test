"""
Self‑Draft Data Generator

What it does
- downloads/loads a chat dataset
- converts to user/assistant format via tokenizer chat template
- runs a target LLM to generate a fixed number of tokens
- records per‑token: last hidden state, next token, and (configurable) log‑probs

Key features
- clean, modular pipeline; supports JSON or YAML configs
- dataset adapters for ShareGPT‑style schemas; easy to add more
- safety/length checks and deterministic shuffling
- optional top‑k compression of log‑probs to save space
- robust saving with small sharded .pt files + an index.jsonl

Usage
    python generate_selfdraft.py --config path/to/config.yaml

Config examples are at the bottom of this file for quick copy/paste.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import SaveLogProbsCfg, GenDataConfig, load_gen_data_config
try:
    import yaml  # type: ignore
except Exception:  # yaml is optional; JSON still works
    yaml = None


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


# ---------------------------
# Dataset adapters
# ---------------------------
@torch.no_grad()
def adapt_sharegpt(tokenizer, example: Dict[str, Any], max_len: int, jacobi_tokens: int) -> Optional[Dict[str, Any]]:
    """Adapt a single ShareGPT‑style record to one training prompt.
    Returns {input_ids, conversation} or None to skip.
    """
    roles = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "chatgpt": "assistant",
        "bard": "assistant",
        "bing": "assistant",
        "system": "system",
    }
    convroles = ["user", "assistant"]

    source = example.get("conversations", [])
    if not source:
        return None
    # Drop leading non‑user messages
    if roles.get(source[0].get("from"), "user") != "user":
        source = source[1:]

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # enforce alternating roles
    for j, s in enumerate(source):
        r = roles.get(s.get("from"), "user")
        if r != convroles[j % 2]:
            return None
        messages.append({"role": r, "content": s.get("value", "")})

    if tokenizer.pad_token_id is None:
        # fall back to EOS if PAD missing
        tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.unk_token_id

    # Build a prompt up to each user turn (excludes the final assistant)
    # We take only the last user turn before an assistant response to keep one prompt per record
    # This keeps things simple and avoids exploding the dataset.
    cut = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            cut = i
            break
    if cut is None:
        return None

    prompt = tokenizer.apply_chat_template(
        messages[: cut + 1], tokenize=False, add_generation_prompt=True
    )

    toks = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_len - jacobi_tokens,
    ).input_ids[0]

    if toks.shape[0] + jacobi_tokens > max_len:
        return None

    return {"input_ids": toks, "conversation": prompt}


def build_dataset(tokenizer, cfg: GenDataConfig) -> List[Dict[str, Any]]:
    ds = load_dataset(cfg.data.data_name, split=cfg.data.split, data_files=cfg.data.data_files)
    # deterministic order with seed, then slice
    ds = ds.shuffle(seed=cfg.gen.seed)
    ds = ds.select(range(cfg.data.start, cfg.data.end))

    out: List[Dict[str, Any]] = []
    adapter = cfg.data.adapter.lower()

    adapter_fn = {
        "sharegpt": adapt_sharegpt,
    }.get(adapter)
    if adapter_fn is None:
        raise ValueError(f"Unknown adapter: {cfg.data.adapter}")

    for ex in tqdm(ds, desc="preprocess", leave=False):
        got = adapter_fn(tokenizer, ex, cfg.gen.max_len, cfg.gen.jacobi_tokens)
        if got is not None:
            out.append(got)
    return out


# ---------------------------
# Generation + capture
# ---------------------------
@torch.no_grad()
def run_stream(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,  # [T]
    steps: int,
    do_sample: bool,
    temperature: float,
    save_probs: SaveLogProbsCfg,
    hidden_store_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Greedy/sampled streaming generation capturing last‑layer hidden states,
    next tokens, and (optionally) log‑probs each step.
    Returns CPU tensors for storage.
    """
    device = next(model.parameters()).device
    x = input_ids.unsqueeze(0).to(device)

    past = None
    current = x

    hidden_list: List[torch.Tensor] = []  # [steps, H]
    next_tokens: List[torch.Tensor] = []  # [steps]

    # optional prob storage
    store_mode = save_probs.mode.lower()
    store_topk = store_mode == "topk"
    store_full = store_mode == "full"
    probs_dtype = dtype_from_str(save_probs.dtype)

    if store_topk:
        topk_vals: List[torch.Tensor] = []   # [steps, K]
        topk_idx: List[torch.Tensor] = []    # [steps, K]
    elif store_full:
        logprobs_list: List[torch.Tensor] = []  # [steps, V]

    for _ in range(steps):
        out = model(
            input_ids=current,
            past_key_values=past,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        last_h = out.hidden_states[-1][:, -1, :]  # [1, H]
        logits = out.logits[:, -1, :]            # [1, V], float32 typically
        past = out.past_key_values

        # select next token
        if do_sample:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            ntok = torch.multinomial(probs, num_samples=1)  # [1,1]
        else:
            ntok = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]

        # capture
        hidden_list.append(last_h.squeeze(0).to(hidden_store_dtype))
        next_tokens.append(ntok.squeeze(0))

        if store_topk:
            vals, idx = torch.topk(F.log_softmax(logits, dim=-1), k=save_probs.top_k, dim=-1)
            topk_vals.append(vals.squeeze(0).to(probs_dtype))
            topk_idx.append(idx.squeeze(0).to(torch.int32))
        elif store_full:
            logp = F.log_softmax(logits, dim=-1)
            logprobs_list.append(logp.squeeze(0).to(probs_dtype))

        current = ntok  # next step prompt is just the new token [1,1]

    hidden = torch.stack(hidden_list, dim=0).cpu()        # [steps, H]
    next_t = torch.stack(next_tokens, dim=0).cpu()        # [steps]

    result: Dict[str, torch.Tensor] = {
        "generated_next": next_t,
        "hidden": hidden,
    }

    if store_topk:
        result["logprobs_topk_vals"] = torch.stack(topk_vals, dim=0).cpu()
        result["logprobs_topk_idx"] = torch.stack(topk_idx, dim=0).cpu()
    elif store_full:
        result["logprobs_full"] = torch.stack(logprobs_list, dim=0).cpu()

    result["input_ids"] = input_ids.cpu()
    return result


# ---------------------------
# Saving
# ---------------------------
class ShardedWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.shard_size = shard_size
        self.idx = 0
        self.in_shard = 0
        self.shard_path = self.out_dir / f"shard_{self.idx:05d}.pt"
        self.buf: List[Dict[str, Any]] = []
        self.index_f = open(self.out_dir / "index.jsonl", "a", encoding="utf-8")

    def write(self, item: Dict[str, Any]) -> None:
        self.buf.append(item)
        self.in_shard += 1
        if self.in_shard >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        torch.save(self.buf, self.shard_path.as_posix())
        # append index entries
        for i in range(len(self.buf)):
            self.index_f.write(json.dumps({"shard": self.shard_path.name, "offset": i}) + "\n")
        self.index_f.flush()
        # reset state
        self.buf.clear()
        self.in_shard = 0
        self.idx += 1
        self.shard_path = self.out_dir / f"shard_{self.idx:05d}.pt"

    def close(self) -> None:
        self._flush()
        self.index_f.close()


# ---------------------------
# Main
# ---------------------------
@torch.no_grad()
def main(cfg: GenDataConfig) -> None:
    set_seed(cfg.gen.seed)

    out_dir = Path(cfg.gen.output_folder) / f"{cfg.data.data_name.replace('/', '_')}_{cfg.model.model_name}_{cfg.gen.name}/{cfg.data.split}"
    ensure_dir(out_dir)

    model_id = f"{cfg.model.model_path}/{cfg.model.model_name}" if cfg.model.model_path else cfg.model.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    # load and adapt dataset
    dataset = build_dataset(tokenizer, cfg)
    if not dataset:
        print("No usable samples after preprocessing.")
        return

    # choose device/dtype
    weights_dtype = dtype_from_str(cfg.gen.dtype)
    if cfg.gen.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=weights_dtype,
        )
        device = next(model.parameters()).device
    else:
        device = torch.device(cfg.gen.device)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": device},
            torch_dtype=weights_dtype,
        )
    model.eval()

    # writer
    writer = ShardedWriter(out_dir, shard_size=max(1, cfg.gen.save_every))

    try:
        pbar = tqdm(range(len(dataset)), desc="generate")
        for i in pbar:
            # if i < 14000:
            #     continue
            rec = dataset[i]
            inp: torch.Tensor = rec["input_ids"]  # [T]

            # double‑check length
            if inp.shape[0] + cfg.gen.jacobi_tokens > cfg.gen.max_len:
                continue

            item = run_stream(
                model=model,
                input_ids=inp,
                steps=cfg.gen.jacobi_tokens,
                do_sample=cfg.gen.do_sample,
                temperature=cfg.gen.temperature,
                save_probs=cfg.save_probs,
                hidden_store_dtype=dtype_from_str(cfg.gen.save_hidden_dtype),
            )
            # basic metadata for traceability
            item["meta"] = {
                "prompt_len": int(inp.shape[0]),
                "steps": int(cfg.gen.jacobi_tokens),
                "model": cfg.model.model_name,
                "data_name": cfg.data.data_name,
            }
            writer.write(item)
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/generate_data_cfg.yaml")
    args = parser.parse_args()
    config = load_gen_data_config(args.config)
    main(config)



