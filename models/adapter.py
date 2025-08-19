import torch
import torch.nn as nn
import math
from tools.utils import  _exec_device

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2AdapterMLP(nn.Module):
    def __init__(self, input_size, output_size, intermediate_ratio=None, clamp=False, use_dora=False, dora_rank=2048, dora_alpha=4096):
        super().__init__()
        self.input_size = input_size
        self.clamp = clamp
        self.use_dora = use_dora
        self.dora_rank = dora_rank
        self.dora_alpha = dora_alpha  # Scaling factor for DoRA updates
        if self.clamp:
            print("adapter uses clamp")
        self.intermediate_size = int(input_size * intermediate_ratio) if intermediate_ratio is not None else input_size * 2
        self.hidden_size = output_size

        # Original linear layers (frozen if DoRA is enabled)
        self.gate_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        if self.use_dora:
            # Freeze original weights
            for layer in [self.gate_proj, self.up_proj, self.down_proj]:
                for param in layer.parameters():
                    param.requires_grad = False

            # Decompose and initialize DoRA components for each layer
            self._initialize_dora(self.gate_proj, self.input_size, self.intermediate_size)
            self._initialize_dora(self.up_proj, self.input_size, self.intermediate_size)
            self._initialize_dora(self.down_proj, self.intermediate_size, self.hidden_size)

    def _initialize_dora(self, layer, in_features, out_features):
        dev = _exec_device(layer)

        with torch.no_grad():
            w = layer.weight.detach()                           # safer than .data
            mag = torch.norm(w, dim=1, keepdim=True)           # [out, 1]
            dir = w / (mag + 1e-8)

        # Parameters (will move with the layer)
        layer.register_parameter("magnitude", nn.Parameter(mag.to(dev).clone()))
        layer.register_parameter("dora_A", nn.Parameter(torch.zeros(in_features, self.dora_rank, device=dev)))
        layer.register_parameter("dora_B", nn.Parameter(torch.zeros(self.dora_rank, out_features, device=dev)))
        nn.init.kaiming_uniform_(layer.dora_A, a=math.sqrt(5))
        nn.init.zeros_(layer.dora_B)

        # Buffer for frozen direction (moves with the layer; set persistent as you need)
        layer.register_buffer("direction", dir.to(dev), persistent=True)

    def _apply_dora(self, layer, x):
        """
        x: [..., in_features]
        layer.magnitude: [out_features, 1]   (trainable)
        layer.direction: [out_features, in_features] (buffer, frozen)
        layer.dora_A: [in_features, r]
        layer.dora_B: [r, out_features]
        """
        # devices/dtypes (assume you already co-located tensors to layer's device)
        W_dir = layer.direction.to(x.dtype)              # [out, in]
        mag   = layer.magnitude.to(x.dtype)              # [out, 1]
        A     = layer.dora_A.to(x.dtype)                 # [in, r]
        B     = layer.dora_B.to(x.dtype)                 # [r, out]

        # Base path: x @ (mag * direction)^T
        W_eff_T = (mag * W_dir).transpose(0, 1)          # [in, out]
        y = x @ W_eff_T                                  # [..., out]

        # Low-rank DoRA update in weight space, done in activation space efficiently:
        # Δy = α * (x @ A) @ (B ⊙ mag.squeeze(-1))   where ⊙ scales each column of B
        B_scaled = B * mag.squeeze(-1)                  # [r, out], broadcast on last dim
        delta = (x @ A) @ B_scaled                      # [..., out]

        return y + self.dora_alpha * delta

    def forward(self, hidden_state):
        # Gate projection with DoRA
        if self.use_dora:
            gate_proj = self._apply_dora(self.gate_proj, hidden_state)
        else:
            gate_proj = self.gate_proj(hidden_state)
        if self.clamp:
            gate_proj = torch.clamp(gate_proj, min=-1e2, max=1e2)
        gate_proj = self.act_fn(gate_proj)

        # Up projection with DoRA
        if self.use_dora:
            up_proj = self._apply_dora(self.up_proj, hidden_state)
        else:
            up_proj = self.up_proj(hidden_state)

        # Combine
        proj = gate_proj * up_proj
        if self.clamp:
            proj = torch.clamp(proj, min=-1e3, max=1e3)

        # Down projection with DoRA
        if self.use_dora:
            return self._apply_dora(self.down_proj, proj)
        else:
            return self.down_proj(proj)

class BasicLinear(nn.Module):
    def __init__(self, input_size, output_size, act=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = output_size
        self.proj = nn.Linear(self.input_size, output_size)
        self.act = act
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        x = self.proj(hidden_state)
        if self.act:
            x = self.act_fn(x)
        return x

class EnhancedQwen2MLP(nn.Module):
    def __init__(self, input_size, output_size, intermediate_ratio=2, clamp=False):
        super().__init__()
        self.layer1 = Qwen2AdapterMLP(input_size, input_size, intermediate_ratio, clamp)
        self.layer2 = Qwen2AdapterMLP(input_size, output_size, intermediate_ratio, clamp)
        self.layernorm = Qwen2RMSNorm(input_size)

    def forward(self, hidden_state):
        residual = hidden_state
        x = self.layer1(hidden_state)
        x = self.layernorm(x + residual)  # Residual connection
        return self.layer2(x)

class ProjectionQwen2AdapterMLP(nn.Module):
    def __init__(self, input_size, output_size, intermediate_ratio=None, clamp=False, layers=1):
        super().__init__()
        self.module_list = nn.ModuleList([Qwen2AdapterMLP(input_size, output_size, intermediate_ratio, clamp) for _ in range(layers)])
    
    def forward(self, hidden_state, idx):
        return self.module_list[idx](hidden_state) 
           
class ProjectionLinear(nn.Module):
    def __init__(self, input_size, output_size, layers):
        super().__init__()
        self.module_list = nn.ModuleList([BasicLinear(input_size, output_size) for _ in range(layers)])
    
    def forward(self, hidden_state, idx):
        return self.module_list[idx](hidden_state)

class ProjectionEnhancedQwen2MLP(nn.Module):
    def __init__(self, input_size, output_size, intermediate_ratio=None, clamp=False, layers=1):
        super().__init__()
        self.module_list = nn.ModuleList([EnhancedQwen2MLP(input_size, output_size, intermediate_ratio, clamp) for _ in range(layers)])
    
    def forward(self, hidden_state, idx):
        return self.module_list[idx](hidden_state)