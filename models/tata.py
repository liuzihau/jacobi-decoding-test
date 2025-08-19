from typing import Tuple, List, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3RMSNorm
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from models.datatype import JacobiCausalLMOutputWithPast
from models.adapter import Qwen2AdapterMLP, EnhancedQwen2MLP, BasicLinear

from tools.tree_structure import TreeStructure, InputProcessor
from tools.utils import timer, _time, _exec_device



class TaTa(nn.Module):
    """
    Jacobi-augmentedCausalLM.

    Key knobs
    - num_jacobi_tokens (J):   how many Jacobi tokens per group
    - num_prev_sequences (S):  how many previous hidden states to concatenate with the current one
    - adapter_insertion_freq (F):           insert an adapter every F decoder layers
    - shared_adapter:          one adapter shared across all slots vs. per-slot adapters
    - fuse_prev_hidden_states: concatenate hidden states from the previous F-1 layers as features
    - shared_jacobi_token:     use one shared Jacobi embedding vs. per-position embeddings (J x H)
    - use_pre_layer_norm:      apply a pre-adapter RMSNorm over the concatenated features
    - token_sets_inline:       dataset layout hint (inline vs tail); kept for compatibility
    - decoding_mode:           "jacobi" or baseline modes (left as-is for your jagenerate)
    """
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(
        self,
        *,
        pretrained_model_name_or_path,
        num_jacobi_tokens: int = 2,
        num_prev_sequences: int = 1,
        adapter_insertion_freq: int = 4,
        adapter_type: str = "Linear",         # "Linear" | "Qwen2MLP" | "EnhancedQwen2MLP"
        shared_adapter: bool = True,
        fuse_prev_hidden_states: bool = False,
        shared_jacobi_token: bool = True,
        jacobi_adapter_kwargs: dict | None = None,
        use_pre_layer_norm: bool = False,
        token_sets_inline: bool = True,
        decoding_mode: str = "jacobi",
        device_map="auto",
        precision: str = "fp16"
    ):
        # --- config & backbone ---
        super().__init__()
        
        # self.config = config
        if precision == "no":
            torch_dtype = torch.float32
        elif precision == "fp16":
            torch_dtype = torch.float16
        elif precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            raise NotImplementedError(f"Unknown precision type: {precision}")

        # --- basic backbone / lm head ---
        causal = self._load_model(pretrained_model_name_or_path, torch_dtype, device_map)
        self.model = causal.model # backbone
        self.lm_head = causal.lm_head # output head 
        self.vocab_size = causal.vocab_size
        self.device = self.model.device
        
        # --- jacobi knobs / layout flags ---
        self.num_jacobi_tokens = int(num_jacobi_tokens)
        self.num_prev_sequences = int(num_prev_sequences)
        self.adapter_insertion_freq = int(adapter_insertion_freq)
        self.shared_adapter = bool(shared_adapter)
        self.fuse_prev_hidden_states = bool(fuse_prev_hidden_states)
        self.shared_jacobi_token = bool(shared_jacobi_token)
        self.use_pre_layer_norm = bool(use_pre_layer_norm)
        self.token_sets_inline = bool(token_sets_inline)
        self.decoding_mode = decoding_mode

        if self.num_jacobi_tokens <= 0:
            raise ValueError("num_jacobi_tokens must be >= 1")
        if self.adapter_insertion_freq <= 0:
            raise ValueError("adapter_insertion_freq must be >= 1")
        
        # --- jacobi token dim / adapter input dimension ---
        hidden_size = int(self.model.embed_tokens.weight.shape[1]) # Hidden size per token from the backbone

        # --- jacobi token embedding(s) ---
        self.jacobi_weight = self._init_jacobi_token(hidden_size, torch_dtype)


        
        # --- adapter nums ---
        num_layers = len(self.model.layers)
        num_adapters = num_layers // self.adapter_insertion_freq if not self.shared_adapter else 1

        # Feature dimension sent into the adapter:
        #   base = (num_prev_sequences + 1) * H
        #   if fuse_prev_hidden_states: include previous (adapter_insertion_freq - 1) layers' features per token
        if self.fuse_prev_hidden_states:
            adapter_in = hidden_size * (self.adapter_insertion_freq) * (self.num_prev_sequences + 1)
        else:
            adapter_in = hidden_size * (self.num_prev_sequences + 1)
        adapter_out = hidden_size

        pre_adapter_layernorms = [
            Qwen2RMSNorm(adapter_in).to(dtype=torch_dtype) if self.use_pre_layer_norm else nn.Identity() for _ in range(num_adapters)
        ]

        adapters = [
            self._build_adapter(
                adapter_type=adapter_type,
                in_features=adapter_in,
                out_features=adapter_out,
                jacobi_adapter_kwargs=jacobi_adapter_kwargs or {},
                torch_dtype=torch_dtype
                ) for _ in range(num_adapters)
                ]
        
        slot = 0
        for i, layer in enumerate(self.model.layers):
            if (i + 1) % self.adapter_insertion_freq == 0:
                ln  = pre_adapter_layernorms[slot]
                adp = adapters[slot]
                layer.add_module("pre_adapter_layernorm", ln)
                layer.add_module("jacobi_adapter", adp)

                # NEW: co-locate the freshly attached children to this layer’s shard
                dev = _exec_device(layer)
                ln.to(dev)
                adp.to(dev)
                slot = slot + 1 if not self.shared_adapter else slot

        
    def _load_model(
            self, 
            pretrained_model_name_or_path: str,
            torch_dtype: torch.dtype,
            device_map: str
            ):

        if "Qwen2" in pretrained_model_name_or_path:
            return Qwen2ForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path, 
                torch_dtype=torch_dtype, 
                device_map=device_map
                )
        if "Qwen3" in pretrained_model_name_or_path:
            return Qwen3ForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path, 
                torch_dtype=torch_dtype,
                device_map=device_map
                )
        
    # -------------------------------
    # Factory helpers
    # -------------------------------
    def _build_adapter(
            self,
            *,
            adapter_type: str,
            in_features: int,
            out_features: int,
            jacobi_adapter_kwargs: dict,
            torch_dtype: torch.dtype
    ):
        """Create shared or per-slot adapters."""
        adapter_map = {
            "Linear": BasicLinear,
            "Qwen2MLP": Qwen2AdapterMLP,
            "EnhancedQwen2MLP": EnhancedQwen2MLP,
        }
        if adapter_type not in adapter_map:
            raise NotImplementedError(f"Unknown adapter_type: {adapter_type}")
        
        adapter_cls = adapter_map[adapter_type]
        return adapter_cls(in_features, out_features, **jacobi_adapter_kwargs).to(dtype=torch_dtype)
    
    # def _build_adapters(
    #     self,
    #     *,
    #     adapter_type: str,
    #     num_layers: int,
    #     in_features: int,
    #     out_features: int,
    #     jacobi_adapter_kwargs: dict,
    # ):
    #     """Create shared or per-slot adapters."""
    #     adapter_map = {
    #         "Linear": ProjectionLinear,
    #         "Qwen2MLP": ProjectionQwen2AdapterMLP,
    #         "EnhancedQwen2MLP": ProjectionEnhancedQwen2MLP,
    #     }
    #     if adapter_type not in adapter_map:
    #         raise NotImplementedError(f"Unknown adapter_type: {adapter_type}")

    #     adapter_cls = adapter_map[adapter_type]
    #     # Number of adapter slots: 1 when shared, else one per adapter_insertion_freq slot across layers.
    #     num_slots = 1 if self.shared_adapter else max(1, num_layers // self.adapter_insertion_freq)
    #     return adapter_cls(in_features, out_features, layers=num_slots, **jacobi_adapter_kwargs)

    def _init_jacobi_token(self, hidden_size: int, torch_dtype: torch.dtype) -> nn.Parameter:
        """
        Initialize trainable Jacobi token embedding(s).
        - shared: shape [H]
        - unshared: shape [J, H]
        Use the model/config dtype to avoid unnecessary casts.
        """
        # Device is CPU at init; safe — modules will be moved together later.
        if self.shared_jacobi_token:
            w = torch.full((hidden_size,), 1e-5, dtype=torch_dtype)
        else:
            w = torch.full((self.num_jacobi_tokens, hidden_size), 1e-5, dtype=torch_dtype)
        return nn.Parameter(w)

    def init_trainable_weights(self, name, param, method='kaiming'):
        std = self.model.config.initializer_range
        if 'proj.weight' in name:
            if method == 'xavier':
                nn.init.xavier_uniform_(param)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='relu')
            else:
                raise NotImplementedError
        elif 'bias' in name:
            nn.init.zeros_(param)  # Biases initialized to zero
        elif 'jacobi_weight' in name:
            nn.init.normal_(param, mean=0.0, std=std)  # Adjust bounds as necessary
        elif 'layernorm' in name:
            nn.init.ones_(param)  # layernorm initialized to all one in QwenRMSNorm

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def merge_jacobi_tokens(self, inputs_embeds: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        In-place replace embeddings at jacobi positions (loss_mask==1) with trainable jacobi weights.
        Works for shared and per-position (unshared) jacobi tokens.
        """
        # avoid copy if not needed; callers can clone() if they need the original
        emb = inputs_embeds
        B, L, H = emb.shape
        jacobi_pos = (loss_mask == 1)              # [B, L] bool
        if not jacobi_pos.any():
            return emb
        if self.shared_jacobi_token:
            # broadcast the single vector to all jacobi positions
            emb[jacobi_pos] = self.jacobi_weight.to(device=emb.device, dtype=emb.dtype, non_blocking=True)
        else:
            # each block of J jacobi tokens reuses the same J weights in order
            # shape sanity: (#positions,) -> (#positions, H)
            counts_per_batch = jacobi_pos.sum(dim=1) // self.num_jacobi_tokens
            # tile weights: [J,H] -> [groups*J, H] for this batch; do per-sample to preserve groups
            for b in range(B):
                if counts_per_batch[b] == 0:
                    continue
                tiled = self.jacobi_weight.to(device=emb.device, dtype=emb.dtype, non_blocking=True).repeat(int(counts_per_batch[b]), 1)
                emb[b, jacobi_pos[b]] = tiled
        return emb
    
    def _build_cache_position_from_loss_mask(
        self,
        *,
        loss_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.LongTensor, int]:
        """
        Create cache_position and position_ids for sequences that include Jacobi tokens.

        Rules:
        - Normal tokens (loss_mask==0) are positioned 0..N_norm-1 (per batch).
        - Each Jacobi group of size J placed after a normal token gets positions:
                prev_normal_pos + 1 .. prev_normal_pos + J
        - Adds past_seen_tokens from the KV cache as an absolute offset.

        Returns:
        cache_position: [B, L]
        position_ids:   [B, L]
        past_seen_tokens: int
        """
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        if cache_position is None:
            if loss_mask is None:
                # No Jacobi: contiguous positions per batch
                base = torch.arange(seq_len, device=device, dtype=torch.long)
                cache_position = base.unsqueeze(0).expand(-1, -1).clone()  # will be expanded later to [B,L]
            else:
                B, L = loss_mask.shape
                cache_position = torch.full((B, L), -1, device=device, dtype=torch.long)

                # Normal tokens -> strictly increasing per row (0..N_norm-1)
                is_normal = (loss_mask == 0).to(device=device)                                 # [B, L]
                norm_running = torch.cumsum(is_normal.to(torch.long), dim=1) - 1
                cache_position[is_normal] = norm_running[is_normal]

                # Jacobi tokens -> per-batch group stamping (ragged groups => minimal loop)
                J = int(self.num_jacobi_tokens)
                jacobi_mask = ~is_normal
                if jacobi_mask.any():
                    arange_J = torch.arange(1, J + 1, device=device, dtype=torch.long)  # [J]
                    for b in range(B):
                        pos_b = torch.nonzero(jacobi_mask[b], as_tuple=False).flatten()  # [G*J]
                        if pos_b.numel() == 0:
                            continue
                        # ensure complete groups
                        groups = pos_b.numel() // J
                        if groups == 0:
                            continue
                        pos_b = pos_b[: groups * J].view(groups, J)                     # [G, J]
                        prev_idx = pos_b[:, 0] - 1                                       # [G]
                        prev_norm_pos = norm_running[b, prev_idx]                         # [G]
                        offsets = prev_norm_pos.unsqueeze(1) + arange_J.unsqueeze(0)     # [G, J]
                        cache_position[b, pos_b] = offsets
        else:
            # user-supplied cache_position; ensure long dtype
            cache_position = cache_position.to(device=device, dtype=torch.long)

        # shift by past cache length and use the same for position_ids
        cache_position = cache_position + past_seen_tokens
        position_ids = cache_position
        return cache_position, position_ids, past_seen_tokens

    def cat_tokens_inline(self, hidden_states: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        Inline gather/concat for Jacobi adapters (no precomputed indices).

        Layout assumption (inline):
        [ ... , (J jacobi), g_k, (J jacobi), g_{k+1}, ... ]

        For each *jacobi group* (size J), we gather:
        - M = self.num_prev_sequences previous "anchor" positions spaced by (J+1),
            with a special case for the first group which uses immediate predecessors.
        - the group's J jacobi positions.

        Then we build per-jacobi-row features by concatenating windows:
        concat_{i=0..M}  target_states[:, -(J+i) : -i or None, :]
        So the output is [B*G, J, (M+1)*H], ready for the adapter.

        Args:
        hidden_states: [B, L, H]
        loss_mask:     [B, L]  (1 at jacobi positions, 0 otherwise)

        Returns:
        curr_states:   [B*G, J, (M+1)*H]
        """
        B, L, H = hidden_states.shape
        J = int(self.num_jacobi_tokens)
        M = int(self.num_prev_sequences)

        out_chunks = []
        device = hidden_states.device

        for b in range(B):
            # jacobi positions for this sample
            jac_pos = torch.nonzero(loss_mask[b] == 1, as_tuple=False).flatten()   # [G*J]
            if jac_pos.numel() == 0:
                continue

            # ensure complete groups
            G = jac_pos.numel() // J
            if G == 0:
                continue
            jac_pos = jac_pos[: G * J].view(G, J)                                  # [G, J]

            # ----- build previous-sequence anchors -----
            if M > 0:
                # first group: immediate predecessors of the first jacobi index
                first_start = jac_pos[0, 0]
                first_prev = (first_start - torch.arange(M, 0, -1, device=device)).view(1, M)   # [1, M]

                if G > 1:
                    # other groups: arithmetic progression with step (J+1)
                    # base start index for earliest prev seq of each group:
                    bases = jac_pos[1:, 0] - 1 - (J + 1) * (M - 1)                              # [G-1]
                    offsets = (torch.arange(M, device=device) * (J + 1)).view(1, M)             # [1, M]
                    normal_prev = bases.view(-1, 1) + offsets                                   # [G-1, M]
                    prev_idx = torch.cat([first_prev, normal_prev], dim=0)                      # [G, M]
                else:
                    prev_idx = first_prev                                                       # [1, M]
            else:
                prev_idx = jac_pos.new_empty((G, 0))                                            # [G, 0]

            # concatenate prev anchors + jacobi positions -> gather indices
            all_idx = torch.cat([prev_idx, jac_pos], dim=1)                                     # [G, M+J]

            # gather target states for this sample: [G, M+J, H]
            ts = hidden_states[b, all_idx]                                                      # [G, M+J, H]
            out_chunks.append(ts)

        if not out_chunks:
            # no jacobi rows at all
            return hidden_states.new_zeros((0, J, (M + 1) * H))

        # stack across batch samples: [sum_b G_b, M+J, H]
        target_states = torch.cat(out_chunks, dim=0)                                            # [\sum_B{G}, M+J, H]

        # ----- build per-jacobi-row features by concatenating windows -----
        # slices: last J, last J+1, ..., last J+M along the "M+J" axis
        feats = []
        for i in range(M + 1):
            start = -(J + i)
            end = -i if i > 0 else None
            feats.append(target_states[:, start:end, :])                                         # [*, J(+i), H] -> each ends with width J
        # each slice has trailing width exactly J; concat over feature dim
        curr_states = torch.cat(feats, dim=-1)                                                  # [*, J, (M+1)*H]

        # reshape to [B*G, J, (M+1)*H]
        return curr_states.contiguous()

    def cat_tokens_split(self, hidden_states: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """
        Tail (all-at-end) gather/concat for Jacobi adapters.

        Assumed layout per sample:
        [prompt, g0, g1, ..., g_{K-1},  (group0 J tokens), (group1 J tokens), ..., (groupK J tokens)]

        For each Jacobi *group* (size J), we gather:
        - M = self.num_prev_sequences indices that reference prior "anchors" on the base
            sequence (the arithmetic progression used in the original code),
        - the group's J Jacobi indices.
        Then we build per-jacobi-row features by concatenating windows of width J from the tail,
        exactly like in the inline variant.

        Args:
        hidden_states: [B, L, H]
        loss_mask:     [B, L]  (1 at jacobi positions, contiguous blocks of size J)

        Returns:
        curr_states:   [B*G, J, (M+1)*H]
                        where G = num_jacobi_groups per sample.
        """
        B, L, H = hidden_states.shape
        J = int(self.num_jacobi_tokens)
        M = int(self.num_prev_sequences)

        out_chunks = []
        device = hidden_states.device

        for b in range(B):
            # Locate jacobi positions and form groups [G, J]
            jac_pos = torch.nonzero(loss_mask[b] == 1, as_tuple=False).flatten()
            jac_pos = jac_pos.to(device)
            if jac_pos.numel() == 0:
                continue

            G = jac_pos.numel() // J
            if G == 0:
                continue
            jac_pos = jac_pos[: G * J].view(G, J)  # [G, J], contiguous groups

            # ----- build previous-sequence anchors (same arithmetic as original code) -----
            if M > 0:
                # reference_array ~ arange(first_jac, first_jac + G) as a column [G,1]
                first = jac_pos[0, 0]
                reference = (first + torch.arange(G, device=device)).view(G, 1)  # [G,1]

                # mix_array ~ arange(G-1+M, G-1, step=-1) as a row [1,M]
                # (length M): [G-1+M, G-2+M, ..., G]
                mix = torch.arange(G - 1 + M, G - 1, step=-1, device=device).view(1, M)  # [1, M]

                # prev_idx[g, m] = reference[g,0] - mix[0,m]
                prev_idx = reference - mix  # [G, M]
            else:
                prev_idx = jac_pos.new_empty((G, 0))  # [G, 0]

            # Gather [G, M+J, H]
            all_idx = torch.cat([prev_idx, jac_pos], dim=1)  # [G, M+J]
            ts = hidden_states[b, all_idx]                   # [G, M+J, H]
            out_chunks.append(ts)

        if not out_chunks:
            return hidden_states.new_zeros((0, J, (M + 1) * H))

        # Stack across batch: [sum_b G_b, M+J, H]
        target_states = torch.cat(out_chunks, dim=0)

        # ----- concatenate tail windows (width J) over feature dim -----
        # windows: last J, last J+1, ..., last J+M
        feats = []
        for i in range(M + 1):
            start = -(J + i)
            end = -i if i > 0 else None
            feats.append(target_states[:, start:end, :])  # [..., J, H]
        curr_states = torch.cat(feats, dim=-1)           # [..., J, (M+1)*H]

        # Final shape: [B*G, J, (M+1)*H]
        return curr_states.contiguous()



    def cat_tokens_with_index(self, hidden_states: torch.Tensor, cat_indices: torch.Tensor) -> torch.Tensor:
        """ [Inference]
        Gather & concatenate prior states for Jacobi adapters using precomputed indices.

        Args:
        hidden_states: [B, L, H]
        cat_indices:   [B, S, M]
            - S = G * J, where:
                J = self.num_jacobi_tokens (jacobi tokens per group)
                G = number of jacobi groups per sequence
            - M = mix window size per jacobi row (e.g., mix_sequences+1)

        Returns:
        curr_states: [B*G, J, M*H]
            (stacked over batches; groups are contiguous; ready for adapter forward)
        """
        B, L, H = hidden_states.shape
        B2, S, M = cat_indices.shape
        assert B == B2, f"Batch mismatch: hidden_states={B}, cat_indices={B2}"
        J = int(self.num_jacobi_tokens)
        assert S % J == 0, f"S ({S}) must be divisible by J ({J})"
        G = S // J

        # Handle optional negative indices (padding) by clamping then zeroing those slices
        has_neg = (cat_indices < 0).any()
        safe_idx = cat_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, H)  # [B, S, M, H]

        # Gather in one go: [B, S, M, H]
        gathered = hidden_states.gather(dim=1, index=safe_idx)

        if has_neg:
            # zero out positions that were padded (idx < 0)
            valid = (cat_indices >= 0).unsqueeze(-1)                              # [B, S, M, 1]
            gathered = torch.where(valid, gathered, torch.zeros_like(gathered))

        # Collapse mix dimension into features: [B, S, M*H]
        gathered = gathered.view(B, S, M * H)

        # Group S rows into (G, J): [B, G, J, M*H] -> [B*G, J, M*H]
        curr_states = gathered.view(B, G, J, M * H).reshape(B * G, J, M * H)

        return curr_states.contiguous()
    
    def handle_attention_mask(
        self,
        bs: int,
        target_length: int,
        attention_mask: Optional[torch.Tensor],
        past_seen_tokens: int,
        loss_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build an additive attention mask.

        If `attention_mask` is already 4D ([B,1,Q,K_now]), we left-pad the key axis with zeros
        for `past_seen_tokens` and return it (cast to `dtype`).

        Else we produce a [B, L, L] additive mask with:
        - Base causal deny (upper triangle).
        - Jacobi rule (works for INLINE and TAIL):
            For every Jacobi *column* j (loss_mask[:, j] == 1), mask all rows at column j,
            EXCEPT within its own Jacobi group block of size J×J, which is unmasked.
        """
        min_dtype = torch.finfo(dtype).min

        # -------- A) 4D mask provided: just pad keys for past length --------
        if attention_mask is not None and attention_mask.dim() > 2:  # [BLL] [B1,Q,K_now]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            # attention_mask: [B, 1, Q, K_now]
            B, H, Q, K_now = attention_mask.shape
            if past_seen_tokens > 0:
                pad = torch.zeros((B, H, Q, past_seen_tokens), dtype=attention_mask.dtype, device=device)
                out = torch.cat([pad, attention_mask], dim=-1)
            else:
                out = attention_mask
            return out.to(dtype)

        # -------- B) Build [B, L, L] additive mask --------
        L = target_length
        ar = torch.arange(L, device=device)
        # base causal deny: True where j > i
        deny = (ar > ar.view(-1, 1)).unsqueeze(0).expand(bs, -1, -1)  # [B, L, L] bool

        if loss_mask is None or not torch.any(loss_mask == 1):
            # no jacobi columns → just causal
            return torch.where(deny,
                            torch.tensor(min_dtype, dtype=dtype, device=device),
                            torch.tensor(0, dtype=dtype, device=device))

        J = int(self.num_jacobi_tokens)
        if J <= 0:
            # safety
            return torch.where(deny,
                            torch.tensor(min_dtype, dtype=dtype, device=device),
                            torch.tensor(0, dtype=dtype, device=device))

        if not self.token_sets_inline:
            raise NotImplementedError("Currently only can handle jacobi tokens insert inline situation")
        
        # column-wise jacobi mask C: mask all jacobi columns for all rows
        # C[b, i, j] = (loss_mask[b, j] == 1)
        C = loss_mask.to(torch.bool).unsqueeze(1).expand(-1, L, -1).clone()  # [B, L, L] bool

        # for each batch, locate jacobi positions and clear their own group blocks
        # groups must be contiguous blocks of length J
        for b in range(bs):
            jac_cols = torch.nonzero(loss_mask[b] == 1, as_tuple=False).flatten()  # [G*J]
            if jac_cols.numel() == 0:
                continue
            # ensure complete groups
            num_groups = jac_cols.numel() // J
            if num_groups == 0:
                continue
            jac_cols = jac_cols[: num_groups * J].view(num_groups, J)  # [G, J]
            # zero out C inside each J×J block
            for g in range(num_groups):
                s = jac_cols[g, 0].item()
                e = s + J  # exclusive
                C[b, s:e, s:e] = False

        # final deny = causal upper OR jacobi column mask
        deny = deny | C
        return torch.where(
            deny,
            torch.tensor(min_dtype, dtype=dtype, device=device),
            torch.tensor(0, dtype=dtype, device=device),
        )

    def update_hidden_states(
        self,
        hidden_states: torch.Tensor,   # [B, L, H]
        new_states: torch.Tensor,      # [ΣG, J, H]
        loss_mask: torch.Tensor,       # [B, L]
        jacobi_indices: Optional[torch.Tensor] = None,
    ) -> None:
        # Anchor device to the tensor owned by this layer shard
        device = hidden_states.device

        # Move all inputs that will be used in indexing/assignment
        new_states   = new_states.to(device, non_blocking=True)
        loss_mask    = loss_mask.to(device, non_blocking=True)
        if jacobi_indices is not None:
            jacobi_indices = jacobi_indices.to(device, non_blocking=True)

        # (Optional) ensure contiguous for safety/perf
        hidden_states = hidden_states.contiguous()
        new_states    = new_states.contiguous()

        B, L, H = hidden_states.shape
        J = int(self.num_jacobi_tokens)

        ptr = 0
        for b in range(B):
            if jacobi_indices is not None:
                idx = jacobi_indices[b]
            else:
                idx = torch.nonzero(loss_mask[b] == 1, as_tuple=False).flatten()

            if idx.numel() == 0:
                continue

            G = idx.numel() // J
            if G == 0:
                continue

            chunk = new_states[ptr: ptr + G]  # [G, J, H]
            ptr += G

            # Make sure idx is on the same device and do the write
            idx = idx.to(device, non_blocking=True)
            hidden_states[b, idx] = chunk.reshape(G * J, H)
            # or: hidden_states[b].index_copy_(0, idx, chunk.reshape(G*J, H))

            
    def forward_backbone_decoder_layer(
        self,
        decoder_layer,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values,
        # output_attentions: bool,
        use_cache: bool,
        cache_position: torch.LongTensor,
        position_embeddings: torch.Tensor,
        jacobi_indices: Optional[torch.Tensor],
        cat_indices: Optional[torch.Tensor],
        all_hidden_states,   # list or None; used only when fuse_prev_hidden_states=True
    ):
        """
        Single decoder layer with optional Jacobi adapter insertion.
        Returns a tuple matching HF convention:
        (hidden_states, [self_attn_weights], [present_key_value])
        """
        # --- Co-locate inputs to this layer's shard ---
        layer_device = _exec_device(decoder_layer)   # <- hook source of truth
        h_dtype = hidden_states.dtype
        def _to_layer(x, *, dtype=None):
            if x is None: 
                return None
            return x.to(device=layer_device, dtype=(dtype or x.dtype), non_blocking=True)
        hidden_states       = _to_layer(hidden_states, dtype=h_dtype)
        causal_mask         = _to_layer(causal_mask,   dtype=h_dtype)   # additive mask uses model compute dtype
        position_ids        = _to_layer(position_ids)                   # long
        cache_position      = _to_layer(cache_position)                 # long
        position_embeddings = _to_layer(position_embeddings[0], dtype=h_dtype), _to_layer(position_embeddings[1], dtype=h_dtype)

        # ---- Self-attention block ----
        residual = hidden_states

        normed = _time("layernorm", decoder_layer.input_layernorm, hidden_states=hidden_states)
        attn_out = _time(
            "attn",
            decoder_layer.self_attn,
            hidden_states=normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            # output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            jacobi_tokens=self.num_jacobi_tokens,
        )
        # unpack attention outputs

        # if output_attentions:
        attn_hidden, present_key_value = attn_out
        # else:
        #     attn_hidden, present_key_value = attn_out
        #     self_attn_weights = None

        hidden_states = residual.to(attn_hidden.device, non_blocking=True) + attn_hidden

        # ---- MLP block ----
        residual = hidden_states
        mlp_in = _time("layernorm", decoder_layer.post_attention_layernorm, hidden_states=hidden_states)
        mlp_out = _time("mlp", decoder_layer.mlp, x=mlp_in)
        hidden_states = residual.to(mlp_out.device, non_blocking=True) + mlp_out

        # ---- Jacobi adapter block (every adapter_insertion_freq layers) ----
        layer_idx = decoder_layer.self_attn.layer_idx
        if (layer_idx + 1) % self.adapter_insertion_freq == 0:
            adapter_idx = 0 if self.shared_adapter else (layer_idx // self.adapter_insertion_freq)

            # features to feed adapter
            if self.fuse_prev_hidden_states:
                # use the previous (F-1) inputs plus current hidden; all_hidden_states is a list we appended to upstream
                # example behavior preserved: concat last (F-1) entries + current hidden along the feature dim

                prev = all_hidden_states[(-self.adapter_insertion_freq + 1):] if all_hidden_states else []
                prev = [_to_layer(p) for p in prev]
                used_states = torch.cat(tuple(prev) + (hidden_states,), dim=-1) if prev else torch.cat((hidden_states,), dim=-1)
            else:
                used_states = hidden_states

            # gather/concat jacobi token rows
            if cat_indices is not None:
                # fast path: dataset precomputed gather indices
                curr_states = _time("cat_tokens", self.cat_tokens_with_index, hidden_states=used_states, cat_indices=cat_indices)
            else:
                # fallback paths (keep behavior)
                if self.token_sets_inline:
                    if self.num_prev_sequences != 1:
                        raise NotImplementedError(
                            f"inline concat currently supports num_prev_sequences==1, got {self.num_prev_sequences}"
                        )
                    curr_states = _time("cat_tokens", self.cat_tokens_inline, hidden_states=used_states, loss_mask=loss_mask)
                else:
                    curr_states = self.cat_tokens_split(used_states, loss_mask)

            # optional pre-adapter LN
            curr_states = decoder_layer.pre_adapter_layernorm(curr_states) if self.use_pre_layer_norm else curr_states

            # run adapter(s) and write back into jacobi rows
            new_states = _time("adapters", decoder_layer.jacobi_adapter, hidden_state=curr_states)#, idx=adapter_idx)
            _time(
                "update_hidden_states",
                self.update_hidden_states,
                hidden_states=hidden_states,
                new_states=new_states,
                loss_mask=loss_mask,
                jacobi_indices=jacobi_indices,
            )

        # ---- pack outputs ----
        outputs = (hidden_states,)
        # if output_attentions:
        #     outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


    def forward_backbone_decoder_layers(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values,
        # output_attentions: bool,
        use_cache: bool,
        cache_position: torch.LongTensor,
        position_embeddings: torch.Tensor,
        output_hidden_states: bool,
        jacobi_indices: Optional[torch.Tensor],
        cat_indices: Optional[torch.Tensor],
    ):
        """
        Runs all decoder layers with optional Jacobi adapters.

        Args match caller. Returns:
        hidden_states: Tensor [B, L, H] after final norm
        all_hidden_states: tuple(Tensor) or None
        next_decoder_cache: cache object or None
        all_self_attns: tuple(Tensor) or None
        """
        collect_h = bool(output_hidden_states)
        # collect_a = bool(output_attentions)

        hidden_col = [] if collect_h else None
        # attn_col = [] if collect_a else None
        next_decoder_cache = None

        for decoder_layer in self.model.layers:
            if collect_h:
                hidden_col.append(hidden_states)

            layer_outputs = self.forward_backbone_decoder_layer(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                loss_mask=loss_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                # output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                jacobi_indices=jacobi_indices,
                cat_indices=cat_indices,
                all_hidden_states=hidden_col,  # keeps your original signature; not used for data, just passthrough
            )

            # layer_outputs follows HF convention:
            # 0: hidden_states, 1: present_key_values (depends on output_attentions)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

            # if collect_a:
            #     attn_col.append(layer_outputs[1])

        # final norm
        hidden_states = self.model.norm(hidden_states)

        # append final hidden
        if collect_h:
            hidden_col.append(hidden_states)

        # pack outputs as tuples for HF compatibility
        all_hidden_states = tuple(hidden_col) if collect_h else None
        # all_self_attns = tuple(attn_col) if collect_a else None

        return hidden_states, all_hidden_states, next_decoder_cache#, all_self_attns


    def forward_backbone_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        jacobi_indices: Optional[torch.Tensor] = None,
        cat_indices: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # -------- 1) resolve flags from backbone --------
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # -------- 2) token embeddings (+Jacobi merge) --------
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # w = self.model.embed_tokens.weight
            # pad = self.model.embed_tokens.padding_idx

            # # force ids to the weight's device and proper dtype/contiguity
            # ids = input_ids.to(device=w.device, dtype=torch.long, non_blocking=True).contiguous()

            # inputs_embeds = F.embedding(ids, w, padding_idx=pad)


        if loss_mask is not None:
            inputs_embeds = _time(
                "merge_jacobi_tokens",
                self.merge_jacobi_tokens,
                inputs_embeds=inputs_embeds,
                loss_mask=loss_mask,
            )

        # -------- 3) cache_position / position_ids --------
        if position_ids is None or cache_position is None:
            seq_len = inputs_embeds.shape[1]
            cache_position, position_ids, past_seen_tokens = self._build_cache_position_from_loss_mask(
                loss_mask=loss_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                seq_len=seq_len,
                device=inputs_embeds.device,
            )
        else:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # -------- 4) rotary embeddings --------
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # -------- 5) attention mask (4D additive bias) --------
        causal_mask = _time(
            "handle_attention_mask",
            self.handle_attention_mask,
            bs=hidden_states.shape[0],
            target_length=hidden_states.shape[1],
            attention_mask=attention_mask,
            past_seen_tokens=past_seen_tokens,
            loss_mask=loss_mask,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # -------- 6) decoder stack --------
        hidden_states, all_hidden_states, next_decoder_cache = self.forward_backbone_decoder_layers(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
            jacobi_indices=jacobi_indices,
            cat_indices=cat_indices,
        )

        # -------- 7) package outputs --------
        next_cache = next_decoder_cache if use_cache else None
        if next_cache is not None:# and return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            # attentions=all_self_attns,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        jacobi_indices: Optional[torch.Tensor] = None,   # kept for interface; not required here
        cat_indices: Optional[torch.Tensor] = None,      # kept for interface; used in backbone
        num_logits_to_keep: int = 0,
        inference: bool = False,
        **loss_kwargs,
    ) -> Union[Tuple, JacobiCausalLMOutputWithPast]:
        """
        Forward unifies:
        - perf timing vs. normal
        - jacobi vs. naive decoding
        - training/inference behavior for jacobi side-outputs
        """

        # ------------- resolve defaults once -------------
        # output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = False if output_hidden_states is None else output_hidden_states

        # ------------- which backbone call? -------------
        if self.decoding_mode == "jacobi":
            # jacobi backbone
            outputs = _time(
                "backbone_model",
                self.forward_backbone_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                # output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                jacobi_indices=jacobi_indices,
                cat_indices=cat_indices,
            )

            hidden_states = outputs[0]                     # [B, L, H]
            logits = self.lm_head(hidden_states)          # [B, L, V]  (cheap)

            # ----- jacobi side-outputs only when training -----
            jacobi_logits = None
            jacobi_hidden_states = None
            jacobi_all_hidden_states = None

            if not inference and loss_mask is not None:
                # Build a single boolean mask over the flattened [B*L] dimension
                dev_ref = logits.device

                B, L, H = hidden_states.shape
                V = logits.size(-1)

                # 1) Build flat indices ON THE CURRENT DEVICE OF loss_mask, then move idx only
                flat_idx = loss_mask.reshape(-1).to(dtype=torch.bool).nonzero(as_tuple=True)[0]  # device follows loss_mask
                # Keep a CPU copy only if you’ll reuse across many devices/layers; optional:
                # flat_idx_cpu = flat_idx.detach().to('cpu')

                # 2) Gather on each source’s device, then move small results to dev_ref

                # hidden states -> [N, H]
                idx_h = flat_idx.to(hidden_states.device, non_blocking=True)
                jacobi_hidden_states = hidden_states.reshape(B * L, H).index_select(0, idx_h)
                if jacobi_hidden_states.device != dev_ref:
                    jacobi_hidden_states = jacobi_hidden_states.to(dev_ref, non_blocking=True)

                # logits -> [N, V]
                idx_l = flat_idx.to(logits.device, non_blocking=True)
                jacobi_logits = logits.reshape(B * L, V).index_select(0, idx_l)
                # already on dev_ref if you picked logits.device; otherwise:
                if jacobi_logits.device != dev_ref:
                    jacobi_logits = jacobi_logits.to(dev_ref, non_blocking=True)

                # 3) All-layer hidden states -> [LAYER, N, H] (each layer may be on a different GPU)
                if output_hidden_states:
                    j_layers = []
                    for t in outputs.hidden_states:  # each t: [B, L, H], possibly on cuda:k
                        idx_t = flat_idx.to(t.device, non_blocking=True)
                        j = t.reshape(B * L, H).index_select(0, idx_t)
                        # co-locate for stacking/consumption
                        if j.device != dev_ref:
                            j = j.to(dev_ref, non_blocking=True)
                        j_layers.append(j)
                    jacobi_all_hidden_states = torch.stack(j_layers, dim=0)  # [LAYER, N, H] on dev_ref

            return JacobiCausalLMOutputWithPast(
                logits=logits,
                jacobi_logits=jacobi_logits,
                past_key_values=outputs.past_key_values,
                jacobi_hidden_states=jacobi_hidden_states,
                jacobi_all_hidden_states=jacobi_all_hidden_states if output_hidden_states else None,
                attentions=outputs.attentions,
            )

        # ------------------------- naive decoding -------------------------
        elif self.decoding_mode == "naive":
            # vanilla Qwen2 forward
            outputs = _time(
                "model",
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                # output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]                                  # [B, L, H]
            # compute only tail logits if requested
            if num_logits_to_keep and num_logits_to_keep > 0:
                tail = hidden_states[:, -num_logits_to_keep:, :]
                logits = self.lm_head(tail)
            else:
                logits = self.lm_head(hidden_states)

            # debug trace (kept as-is)
            if hasattr(self, "wtf"):
                self.wtf = torch.cat([self.wtf, logits.argmax(dim=-1)], dim=-1)
            else:
                self.wtf = logits.argmax(dim=-1)

            loss = None
            if labels is not None:
                loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # ------------------------- unknown mode guard -------------------------
        else:
            raise ValueError(f"Unknown decoding_mode: {self.decoding_mode}")
    
    @torch.no_grad()
    def jagenerate(
            self,
            input_ids,
            max_new_tokens=512,
            max_length=2048,
            do_sample=False,
            top_p=0.0,
            top_k=0.0,
            repetition_penalty=1.0,
            temperature=0.0,
            force_autoregressive=False,
            tokenizer=None
            ):
        
        if not do_sample:
            temperature = 0.0

        input_processor = InputProcessor(input_ids.dtype, torch.float32, input_ids.dtype, input_ids.device, self.num_jacobi_tokens, self.num_prev_sequences)
        past_key_values = DynamicCache()
        tt = 0
        ct = {}

        # do the first inference
        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        input_ids = torch.cat([input_ids]+[padding]*self.num_jacobi_tokens, dim=-1)
        loss_mask = []
        prev_index, jacobi_index = [], []
        for i in range(input_ids.shape[0]):
            jacobi_indices = torch.nonzero(input_ids[i] == -1, as_tuple=True)
            jacobi_indices_groups = jacobi_indices[0].view(-1, self.num_jacobi_tokens)
            prev_index.append(jacobi_indices_groups[:, 0] - 1)
            jacobi_index.append(jacobi_indices[0])
            mask = torch.zeros_like(input_ids[i], device=input_ids.device)
            mask[jacobi_indices] = 1
            input_ids[i, jacobi_indices[0]] = 0
            loss_mask.append(mask)
        prev_index = torch.stack(prev_index, dim=0)
        jacobi_index = torch.stack(jacobi_index, dim=0)
        loss_mask = torch.stack(loss_mask, dim=0)

        output = self.forward(
            input_ids=input_ids,
            loss_mask=loss_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
            inference=True
            )

        # only support batch == 1
        route_indices = torch.nonzero(loss_mask[0] == 0, as_tuple=True)[0]
        for layer_idx in range(len(output["past_key_values"].key_cache)):
            output["past_key_values"].key_cache[layer_idx] = output["past_key_values"].key_cache[layer_idx][:, :, route_indices, :]
            output["past_key_values"].value_cache[layer_idx] = output["past_key_values"].value_cache[layer_idx][:, :, route_indices, :]
        past_seen_tokens = output["past_key_values"].get_seq_length()

        normal_token = decoding_normal_token(output["logits"][i, prev_index[i]], temperature, top_p, top_k)
        jacobi_token, jacobi_token_p, all_p = decoding_jacobi_token(output["logits"][i, jacobi_index[i]], temperature, top_p, top_k)
        current_decoded_tokens = normal_token.view(1, -1)

        # normal_token_dist = nn.Softmax(dim=-1)(output["logits"][0])
        # s = decoding_normal_token(normal_token_dist)
        # for i, (a, b) in enumerate(zip(input_ids[0], s)):
        #     a = tokenizer.decode([a.item()])
        #     b = tokenizer.decode([b.item()])
        #     a = a.replace("\n", "\\n")
        #     b = b.replace("\n", "\\n")
        #     print(f"[{i}th] input token: <{a}>, output token: <{b}>")

        # loop start
        while current_decoded_tokens.shape[-1] < max_new_tokens:
            trees = []
            input_ids, attention_mask, loss_mask, cache_position, jacobi_indices, cat_indices = [], [], [], [], [], []

            tree = TreeStructure(normal_token.detach().cpu().item())
            if PERFORMANCE_CHECK:
                kwargs = {
                    "jacobi_token": jacobi_token,
                    "jacobi_token_p": jacobi_token_p
                }
                timer.record_time("build_tree", tree.build_tree, **kwargs)
            else:
                tree.build_tree(jacobi_token, jacobi_token_p)

            if self.token_sets_inline:
                if PERFORMANCE_CHECK:
                    ith_input_ids, ith_attention_mask, ith_loss_mask, ith_cache_position, ith_jacobi_indices, ith_cat_indices = timer.record_time("build_input_inline", input_processor.build_inputs_inline_jacobi_token, **{"tree":tree})
                else:
                    ith_input_ids, ith_attention_mask, ith_loss_mask, ith_cache_position, ith_jacobi_indices, ith_cat_indices = input_processor.build_inputs_inline_jacobi_token(tree)
            
            
            # for k, layer in enumerate(tree.layers):
            #     for node in layer:
            #         parent = node.parent.val if node.parent is not None else None
            #         print(f"[{k}th layer] val: {node.val}, parent: {parent}, rouute: {node.route}")
            input_ids.append(ith_input_ids)
            attention_mask.append(ith_attention_mask)
            loss_mask.append(ith_loss_mask)
            cache_position.append(ith_cache_position)
            jacobi_indices.append(ith_jacobi_indices)
            cat_indices.append(ith_cat_indices)
            
            trees.append(tree)

            input_ids = torch.stack(input_ids, dim=0)
            attention_mask = torch.stack(attention_mask, dim=0)
            loss_mask = torch.stack(loss_mask, dim=0)
            cache_position = torch.stack(cache_position, dim=0)
            jacobi_indices = torch.stack(jacobi_indices, dim=0)
            cat_indices = torch.stack(cat_indices, dim=0)

            # print(f"="*60 + f" {current_decoded_tokens.shape[0]} " + f"="*60)
            # for i in range(attention_mask[0, 0].shape[0]):
            #     print((attention_mask[0, 0, i] / attention_mask.min()).detach().cpu().type(torch.int16).tolist())
            # # print()
            # print(loss_mask[0].detach().cpu().tolist())
            # print(cache_position[0].detach().cpu().tolist())

            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                past_key_values=output["past_key_values"],
                use_cache=True,
                cache_position=cache_position,
                output_hidden_states=False,
                return_dict=True,
                jacobi_indices = jacobi_indices,
                cat_indices = cat_indices,
                inference=True
                )
            
            i = 0  # only support batch == 1

            # sample
            token_sampled = decoding_normal_token(output["logits"][i], temperature, top_p, top_k)
            
            # 
            entropy = calculate_entropy(all_p)
            threshold = get_threshold(entropy)

            # verify (cheap)
            route_indices, ans_list = verify_final_route(input_ids[i], token_sampled, trees[i], force_autoregressive, do_sample, tokenizer)
            tt += 1
            for c in range(len(ans_list)):
                if tuple(ans_list[:c+1]) in ct:
                    ct[tuple(ans_list[:c+1])] += 1
                else:
                    ct[tuple(ans_list[:c+1])] = 1
            
            verified_tokens = token_sampled[route_indices]
            current_decoded_tokens = torch.cat([current_decoded_tokens, verified_tokens.view(1, -1)], dim=-1)
            if self.model.config.eos_token_id in current_decoded_tokens:
                break

            # print(current_decoded_tokens)
            # handle cache
            if PERFORMANCE_CHECK:
                timer.record_time("update_kv_cache", update_kv_cache, **{"output":output, "route_indices":route_indices, "past_seen_tokens":past_seen_tokens})
            else:
                update_kv_cache(output, route_indices, past_seen_tokens)
            past_seen_tokens = output["past_key_values"].get_seq_length()
            
            # prepare next input
            normal_token = token_sampled[route_indices][-1]
            jacobi_index_start = route_indices[-1]+1
            jacobi_index_end = jacobi_index_start + self.num_jacobi_tokens
            selected_jacobi_indices = torch.arange(jacobi_index_start, jacobi_index_end)
            jacobi_token_logits = output["logits"][i][selected_jacobi_indices]
            jacobi_token, jacobi_token_p, all_p = decoding_jacobi_token(jacobi_token_logits, temperature, top_p, top_k)
        return current_decoded_tokens[:max_new_tokens], tt, ct

def update_kv_cache(output, route_indices, past_seen_tokens):
    device = output["past_key_values"].key_cache[0].device
    route_indices_tensor = torch.tensor(route_indices, device=device)

    # Efficiently create cache indices
    cache_indices = torch.cat([torch.arange(0, past_seen_tokens, device=device), route_indices_tensor + past_seen_tokens], dim=-1)

    for layer_idx in range(len(output["past_key_values"].key_cache)):
            key_cache = output["past_key_values"].key_cache[layer_idx]
            value_cache = output["past_key_values"].value_cache[layer_idx]

            # Use `index_select` for faster tensor slicing
            output["past_key_values"].key_cache[layer_idx] = key_cache.index_select(2, cache_indices)
            output["past_key_values"].value_cache[layer_idx] = value_cache.index_select(2, cache_indices)

def decoding_normal_token(logits, temperature=0.0, top_p=0.0, top_k=0.0):
    """
    Decode token from logits with support for greedy decoding, temperature adjustment,
    top-k sampling, and top-p (nucleus) sampling, aligned with Hugging Face's approach.
    
    Args:
        logits (torch.Tensor): Logits of shape (batch_size, vocab_size).
        temperature (float): Temperature for scaling logits. If 0.0, use greedy decoding.
        top_p (float): Top-p threshold for nucleus sampling. If 0.0, no top-p filtering.
        top_k (float): Number of top tokens for top-k sampling. If 0.0, no top-k filtering.
    
    Returns:
        torch.Tensor: Sampled token indices of shape (batch_size,).
    """
    # Step 1: Greedy decoding if temperature is 0.0
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    
    # Step 2: Apply temperature scaling to logits
    logits = logits / temperature
    
    # Step 3: Apply top-k filtering if top_k > 0
    if top_k > 0:
        # Ensure top_k doesn't exceed vocabulary size
        top_k = min(int(top_k), logits.size(-1))
        # Get top-k logits; values is used for thresholding
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        # Threshold: smallest value in top-k
        min_values = values[..., -1].unsqueeze(-1)
        # Set logits below threshold to -inf
        logits = torch.where(logits >= min_values, logits, torch.full_like(logits, -float('inf')))
    
    # Step 4: Apply top-p filtering if 0 < top_p < 1.0
    if top_p > 0 and top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Tokens to remove: where cumulative probability exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask to include the token that makes cumulative prob exceed top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # Always keep the top token
        sorted_indices_to_remove[..., 0] = 0
        # Get indices to remove in original order
        batch_idx, sorted_idx = torch.where(sorted_indices_to_remove)
        if batch_idx.size(0) > 0:  # Only proceed if there are tokens to remove
            original_idx = sorted_indices[batch_idx, sorted_idx]
            logits[batch_idx, original_idx] = -float('inf')
    
    # Step 5: Compute probabilities and sample
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def decoding_jacobi_token(logits, temperature=0.0, top_p=0.0, top_k=0, expand=3):
    """
    Decode Jacobi tokens from logits, supporting greedy selection of top candidates or
    sampling, always returning probabilities and full distribution.
    
    Args:
        logits (torch.Tensor): Logits of shape (batch_size, seq_len, vocab_size), where
                               seq_len includes Jacobi tokens predicting future tokens.
        temperature (float): Temperature for scaling logits. If 0.0, use greedy decoding.
        top_p (float): Top-p threshold for nucleus sampling. If 0.0, no top-p filtering.
        top_k (float): Number of top tokens for top-k sampling. If 0, no top-k filtering.
        expand (int): Number of top candidates to return (default=3).
    
    Returns:
        tuple:
            - topk_index (torch.Tensor): Top `expand` token indices of shape (batch_size, seq_len, expand).
            - topk_p (torch.Tensor): Corresponding probabilities of shape (batch_size, seq_len, expand).
            - probs (torch.Tensor): Full probability distribution of shape (batch_size, seq_len, vocab_size).
    """
    # Step 1: Greedy decoding case (your suggestion, returning probs)
    if temperature == 0.0 and top_p == 0.0 and top_k == 0:
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, expand, dim=-1)
        topk_index, topk_probs = top.indices, top.values
        return topk_index, topk_probs, probs  # Return full probs in greedy mode
    
    # Step 2: Sampling mode (temperature > 0 or top_k/top_p specified)
    # Apply temperature scaling to logits
    if temperature > 0:
        logits = logits / temperature
    else:
        logits = logits.clone()  # Avoid modifying input if temperature is invalid
    
    # Step 3: Apply top-k filtering if top_k > 0
    if top_k > 0:
        top_k = min(int(top_k), logits.size(-1))
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        min_values = values[..., -1].unsqueeze(-1)
        logits = torch.where(logits >= min_values, logits, torch.full_like(logits, -float('inf')))
    
    # Step 4: Apply top-p filtering if 0 < top_p < 1.0
    if top_p > 0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        batch_idx, seq_idx, sorted_idx = torch.where(sorted_indices_to_remove)
        if batch_idx.size(0) > 0:
            original_idx = sorted_indices[batch_idx, seq_idx, sorted_idx]
            logits[batch_idx, seq_idx, original_idx] = -float('inf')
    
    # Step 5: Compute full probability distribution
    probs = torch.softmax(logits, dim=-1)
    
    # Step 6: Get top `expand` candidates from probabilities
    top = torch.topk(probs, expand, dim=-1)
    topk_index, topk_p = top.indices, top.values
    
    # Return indices, their probabilities, and full distribution
    return topk_index, topk_p, probs

def verify_final_route(inputs, outputs, tree, force_autoregressive=False, do_sample=False, tokenizer=None):
    # print("="*100)
    curr_node = tree.root
    index = tree.index_dict[curr_node]
    curr_ans, final_route, ans_list = outputs[index], curr_node.route, []

    
    if not force_autoregressive:
        found_ans = True
        while len(curr_node.children) > 0 and found_ans:
            found_ans = False
            for i, node in enumerate(curr_node.children):
                index = tree.index_dict[node]
                curr_pred, next_ans = inputs[index], outputs[index]

                # a, b, c = tokenizer.decode([curr_ans]), tokenizer.decode([curr_pred]), tokenizer.decode([next_ans])
                # a = a.replace("\n", "\\n")
                # b = b.replace("\n", "\\n")
                # c = c.replace("\n", "\\n")
                # print(f"ans: <{a}>, pred: <{b}>, next_ans: <{c}>")
                if do_sample:
                    pass
                else:
                    if curr_pred == curr_ans:
                        final_route = node.route
                        curr_ans = next_ans
                        curr_node = node
                        found_ans = True
                        ans_list.append(i)
                        break

    return final_route, ans_list

def calculate_entropy(posterior_prob):
    return -torch.sum(
            posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
        )

def get_threshold(posterior_entropy, posterior_threshold=0.3, posterior_alpha = 0.09):
    return torch.minimum(
        torch.ones_like(posterior_entropy) * posterior_threshold,
        torch.exp(-posterior_entropy) * posterior_alpha,
    )

def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list
