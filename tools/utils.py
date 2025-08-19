import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from safetensors import safe_open


# -------------------------------------------------------------------
# Timing utilities
# -------------------------------------------------------------------

@dataclass
class Stat:
    total: float = 0.0
    count: int = 0

    def add(self, dt: float) -> None:
        self.total += dt
        self.count += 1

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class Timer:
    """
    Lightweight profiler with CPU or CUDA timing.
    Usage:
        with timer.profile("tag"):
            fn()
        # or
        out = timer.record_time("tag", fn, **kwargs)
        # or
        @timer.timeit("tag")
        def foo(...): ...

    Toggle with timer.enabled = True/False
    """
    def __init__(self, use_cuda_events: bool = False) -> None:
        self.enabled: bool = False
        self.use_cuda_events: bool = bool(use_cuda_events)
        self.report: Dict[str, Stat] = {}

    def _now(self) -> float:
        return time.perf_counter()

    def _cuda_time(self, fn: Callable, **kwargs):
        """Accurate CUDA timing using events."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        res = fn(**kwargs)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)  # milliseconds
        return res, ms / 1000.0

    def record_time(self, key: str, fn: Callable, **kwargs):
        if not self.enabled:
            return fn(**kwargs)
        if self.use_cuda_events and torch.cuda.is_available():
            res, dt = self._cuda_time(fn, **kwargs)
        else:
            t0 = self._now()
            res = fn(**kwargs)
            dt = self._now() - t0
        self.report.setdefault(key, Stat()).add(dt)
        return res

    # context manager form
    def profile(self, key: str):
        class _Ctx:
            def __init__(self, outer: "Timer", key: str) -> None:
                self.outer = outer
                self.key = key
                self.t0: Optional[float] = None
                self.start_event = None
                self.end_event = None

            def __enter__(self):
                if not self.outer.enabled:
                    return
                if self.outer.use_cuda_events and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    self.start_event = torch.cuda.Event(enable_timing=True)
                    self.end_event = torch.cuda.Event(enable_timing=True)
                    self.start_event.record()
                else:
                    self.t0 = self.outer._now()

            def __exit__(self, exc_type, exc, tb):
                if not self.outer.enabled:
                    return
                if self.outer.use_cuda_events and torch.cuda.is_available():
                    self.end_event.record()
                    torch.cuda.synchronize()
                    ms = self.start_event.elapsed_time(self.end_event)
                    dt = ms / 1000.0
                else:
                    dt = self.outer._now() - (self.t0 or self.outer._now())
                self.outer.report.setdefault(self.key, Stat()).add(dt)
        return _Ctx(self, key)

    # decorator form
    def timeit(self, key: Optional[str] = None):
        def deco(fn: Callable):
            tag = key or fn.__name__
            def wrapped(*args, **kwargs):
                return self.record_time(tag, fn, **kwargs) if not args else self.record_time(tag, lambda **kw: fn(*args, **kw), **kwargs)
            return wrapped
        return deco

    def reset(self) -> None:
        self.report.clear()

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: {"time": v.total, "count": v.count, "avg": v.avg} for k, v in self.report.items()}

    def pretty(self) -> str:
        lines = ["== Timer Report =="]
        for k, v in sorted(self.report.items(), key=lambda x: -x[1].total):
            lines.append(f"{k:32s}  total={v.total:8.4f}s  count={v.count:6d}  avg={v.avg:8.6f}s")
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2)


# global timer + helper for legacy call-sites
timer = Timer(use_cuda_events=True)  # set True for CUDA-accurate timings
timer.enabled = True

def _time(tag: str, fn: Callable, **kwargs):
    """Tiny wrapper to keep your existing call sites working."""
    return timer.record_time(tag, fn, **kwargs)

# -------------------------------------------------------------------
# Metrics / utilities
# -------------------------------------------------------------------

@torch.no_grad()
def top_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    jacobi_token_nums: int,
    topk: Tuple[int, ...] = (1,)
) -> List[torch.Tensor]:
    """
    Computes top-k accuracy per Jacobi position and sums across groups.
    output: [N, V] or [N, J, V]  (will be reshaped to [-1, J, V])
    target: [N] or [N, J]        (will be reshaped to [-1, J])
    Returns a list of tensors, one per k in topk, each shape [J] = per-position correct counts.
    """
    if output.dim() == 2:
        output = output.view(-1, jacobi_token_nums, output.shape[-1])
    if target.dim() == 1:
        target = target.view(-1, jacobi_token_nums)

    maxk = max(topk)
    # [B, J, maxk]
    _, pred = output.topk(maxk, dim=-1, largest=True, sorted=True)
    # broadcast target -> [B, J, maxk]
    tgt = target.unsqueeze(-1).expand_as(pred)
    correct = (pred == tgt).to(torch.float32)

    res = []
    for k in topk:
        # sum over batch, keep per-position across J
        res.append(correct[..., :k].sum(dim=0).sum(dim=-1))  # [J]
    return res


def output_abnormal_message(
    target_p: torch.Tensor,
    output_logp: torch.Tensor,
    jacobi_hidden_states: torch.Tensor,
    target_hidden_state: torch.Tensor,
    pshape0: int, pshape1: int, vshape0: int, vshape1: int,
) -> str:
    def _flags(t: torch.Tensor) -> str:
        return f"inf={bool(torch.isinf(t).any())}, nan={bool(torch.isnan(t).any())}"
    lines = [
        f"[target_p]        {_flags(target_p)}",
        f"[target_hidden]   {_flags(target_hidden_state)}",
        f"[output_logp]     {_flags(output_logp)}",
        f"[output_hidden]   {_flags(jacobi_hidden_states)}",
        f"[pshape]: {pshape0}, {pshape1}",
        f"[vshape]: {vshape0}, {vshape1}",
    ]
    return "\n".join(lines)


@torch.no_grad()
def load_jacobi_weight(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load jacobi-specific parameters from a .safetensors file.
    Skips backbone params (names starting with 'model.').
    """
    missing, mismatched = [], []
    loaded = 0
    with safe_open(ckpt_path, framework="pt") as f:
        keys = set(f.keys())
        for name, param in model.named_parameters():
            if name.startswith("model."):
                continue
            if name not in keys:
                missing.append(name)
                continue
            t = f.get_slice(name)[:]
            if t.shape != param.shape:
                mismatched.append((name, tuple(param.shape), tuple(t.shape)))
                continue
            param.data.copy_(t.to(device=param.device, dtype=param.dtype))
            loaded += 1

    if loaded:
        print(f"[load_jacobi_weight] loaded {loaded} tensors from {ckpt_path}")
    if missing:
        print(f"[load_jacobi_weight] missing keys: {len(missing)} (e.g., {missing[:5]})")
    if mismatched:
        ex = ", ".join([f"{n} exp{es} got{gs}" for n, es, gs in mismatched[:3]])
        print(f"[load_jacobi_weight] mismatched shapes: {len(mismatched)} ({ex})")
    if not missing and not mismatched:
        print("[load_jacobi_weight] all parameters loaded.")


def _exec_device(mod: nn.Module) -> torch.device:
    hk = getattr(mod, "_hf_hook", None)
    if hk is not None and getattr(hk, "execution_device", None) is not None:
        return torch.device(hk.execution_device)
    # fallback when no hook (e.g., CPU runs/tests)
    for p in mod.parameters(recurse=False):
        return p.device
    return torch.device("cpu")


def save_trainable_weights(model: torch.nn.Module, save_path: str | Path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Only params that require grad
    trainable_sd = {k: v.detach().cpu() 
                    for k, v in model.state_dict().items()
                    if k in dict(model.named_parameters()) and dict(model.named_parameters())[k].requires_grad}
    torch.save(trainable_sd, save_path)
    return save_path