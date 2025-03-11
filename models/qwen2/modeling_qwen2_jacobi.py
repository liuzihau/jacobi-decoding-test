import time
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.utils import ModelOutput

from models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2Model, Qwen2RMSNorm
from tools.tree_structure import TreeStructure, InputProcessor
from tools.utils import PERFORMANCE_CHECK, timer

@dataclass
class JacobiCausalLMOutputWithPast(ModelOutput):
    
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    jacobi_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    jacobi_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    jacobi_all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Qwen2MLP(nn.Module):
    def __init__(self, input_size, output_size, intermediate_ratio=None, clamp=False):
        super().__init__()
        self.input_size = input_size
        self.clamp = clamp
        if self.clamp:
            print("adapter uses clamp")
        self.intermediate_size = int(input_size * intermediate_ratio) if intermediate_ratio is not None else input_size * 2
        self.hidden_size = output_size
        self.gate_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        gate_proj = self.gate_proj(hidden_state)
        if self.clamp:
            gate_proj = torch.clamp(gate_proj, min=-1e2, max=1e2)
        gate_proj = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_state)
        proj = gate_proj * up_proj
        if self.clamp:
            proj = torch.clamp(proj, min=-1e3, max=1e3)
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
        self.layer1 = Qwen2MLP(input_size, input_size, intermediate_ratio, clamp)
        self.layer2 = Qwen2MLP(input_size, output_size, intermediate_ratio, clamp)
        self.layernorm = Qwen2RMSNorm(input_size)

    def forward(self, hidden_state):
        residual = hidden_state
        x = self.layer1(hidden_state)
        x = self.layernorm(x + residual)  # Residual connection
        return self.layer2(x)

class ProjectionQwen2MLP(nn.Module):
    def __init__(self, input_size, output_size, intermediate_ratio=None, clamp=False, layers=1):
        super().__init__()
        self.module_list = nn.ModuleList([Qwen2MLP(input_size, output_size, intermediate_ratio, clamp) for _ in range(layers)])
    
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
    
class Qwen2JacobiForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    
    def __init__(self, config, jacobi_token_nums=2, mix_sequences=1, proj_freq=4, adapter_type='Linear', shared_adapter=True, fuse_previous_hidden_states=False, shared_jacobi_token=True, jacobi_adapter_kwargs=None, layer_norm=False, token_sets_inline=True, decoding_mode="jacobi"):
        super().__init__(config)
        self.confg = config
        self.token_sets_inline = token_sets_inline
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.mix_sequences = mix_sequences
        self.proj_freq = proj_freq
        self.jacobi_token_nums = jacobi_token_nums
        self.shared_adapter = shared_adapter
        self.fuse_previous_hidden_states = fuse_previous_hidden_states
        self.shared_jacobi_token = shared_jacobi_token
        self.layer_norm = layer_norm
        self.decoding_mode = decoding_mode

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        attn_layers, attn_hidden_size = config.num_hidden_layers, config.hidden_size
        self.pre_adapter_layernorm = Qwen2RMSNorm(attn_hidden_size*(self.mix_sequences+1))
        if self.fuse_previous_hidden_states:
            input_size = attn_hidden_size * self.proj_freq
        else:
            input_size = attn_hidden_size
        self.adapters = self.init_adapter(adapter_type, attn_layers, input_size, attn_hidden_size, jacobi_adapter_kwargs)
        self.jacobi_weight = self.init_jacobi_token(attn_hidden_size)
        
        self.post_init()

    def init_adapter(self, adapter_type, attn_layers, input_size, output_size, jacobi_adapter_kwargs):
        if adapter_type == "Linear":
            adapter_module = ProjectionLinear
        elif adapter_type == 'Qwen2MLP':
            adapter_module = ProjectionQwen2MLP
        elif adapter_type == 'EnhancedQwen2MLP':
            adapter_module = ProjectionEnhancedQwen2MLP
        else:
            raise NotImplementedError(f"{adapter_type} hasn't been implemented")
        if self.shared_adapter:
            adapter_layers = 1
        else:
            adapter_layers = attn_layers // self.proj_freq
        jacobi_adapter_kwargs = {} if jacobi_adapter_kwargs is None else jacobi_adapter_kwargs        
        return adapter_module(input_size, output_size, layers=adapter_layers, **jacobi_adapter_kwargs)

    def init_jacobi_token(self, attn_hidden_size):
        temp_weight = torch.ones((attn_hidden_size,), device=self.model.device, dtype=torch.float32) * 1e-5
        temp_weight = temp_weight.to(dtype=torch.bfloat16)
        if self.shared_jacobi_token:
            return nn.Parameter(temp_weight)
        else:
            stacked_weight = torch.stack([temp_weight] * self.jacobi_token_nums, dim=0)
            return nn.Parameter(stacked_weight)

    def init_trainable_weights(self, name, param, method='kaiming'):
        std = self.config.initializer_range
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
    
    def merge_jacobi_tokens(self, inputs_embeds, loss_mask):
        modified_embeds = inputs_embeds.clone()

        for i in range(inputs_embeds.shape[0]):
            replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
            if self.shared_jacobi_token:
                modified_embeds[i, replace_indices] = self.jacobi_weight
            else:
                expand_weight = self.jacobi_weight.repeat(replace_indices.shape[0] // self.jacobi_token_nums, 1)
                modified_embeds[i].scatter_(0, replace_indices.unsqueeze(-1).expand(-1, self.jacobi_weight.shape[-1]), expand_weight)

            return modified_embeds
        
    def cat_tokens_inline_lagacy(self, hidden_states, loss_mask, adapter_idx):
        for i in range(hidden_states.shape[0]):
            replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]  #[6, 7, 9, 10, 12, 13]
            replace_indices_groups = replace_indices.view(-1, self.jacobi_token_nums)  # [[6, 7], [9, 10], [12, 13]]
            prev_seq_indices_groups = (replace_indices_groups[:, 0] - self.mix_sequences).reshape(-1, 1)  #[[5], [8], [11]]
            all_indices = torch.cat([prev_seq_indices_groups, replace_indices_groups], dim=-1)  #[[5, 6, 7], [8, 9, 10], [11, 12, 13]]
            target_states.append(hidden_states[i, all_indices])  # [groups, jacobi_token_nums + mix_seq, hidden_dim]
        target_states = torch.cat(target_states, dim=0)  # [mix groups from all bs, jacobi_token_nums + mix_seq, hidden_dim]
        
        curr_states = target_states[:, -self.jacobi_token_nums:, :]  # [mix groups from all bs, jacobi_token_nums, hidden_dim]
        new_states = None
        for i in range(self.mix_sequences):
            prev_states = target_states[:, -(self.jacobi_token_nums+i+1):-(i+1), :]
            curr_states = torch.cat([curr_states, prev_states], dim=-1)
            if self.layer_norm:
                curr_states = self.pre_adapter_layernorm(curr_states)
            if new_states is None:
                new_states = self.adapters(curr_states, adapter_idx)
            else:
                new_states += self.adapters(curr_states, adapter_idx)
        new_states /= self.mix_sequences
        return new_states

    def cat_tokens_inline(self, hidden_states, loss_mask):
        target_states = []
        for i in range(hidden_states.shape[0]):
            replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]  #[6, 7, 9, 10, 12, 13]
            replace_indices_groups = replace_indices.view(-1, self.jacobi_token_nums)  # [[6, 7], [9, 10], [12, 13]]
            first_replace_indice_group = replace_indices_groups[0]
            normal_replace_indice_groups = replace_indices_groups[1:]
            first_prev_seq_indices_groups = torch.cat([first_replace_indice_group[[0]] - i for i in range(self.mix_sequences, 0, -1)], dim=-1).view(-1, self.mix_sequences)
            start_indices = (normal_replace_indice_groups[:, 0] - 1 - (self.jacobi_token_nums+1) * (self.mix_sequences - 1)).view(-1, 1)
            normal_prev_seq_indices_groups = torch.cat([start_indices + i * (self.jacobi_token_nums + 1) for i in range(self.mix_sequences)], dim=-1)
            prev_seq_indices_groups = torch.cat([first_prev_seq_indices_groups, normal_prev_seq_indices_groups], dim=0)
            all_indices = torch.cat([prev_seq_indices_groups, replace_indices_groups], dim=-1)  #[[5, 6, 7], [8, 9, 10], [11, 12, 13]]
            target_states.append(hidden_states[i, all_indices])  # [groups, jacobi_token_nums + mix_seq, hidden_dim]
        target_states = torch.cat(target_states, dim=0)
        curr_states = torch.cat([target_states[:, -(self.jacobi_token_nums + i):-i if i > 0 else None, :] for i in range(self.mix_sequences + 1)], dim=-1)
        return curr_states
    
    def cat_tokens_with_index(self, hidden_states, cat_indices):
        curr_states = []
        for i in range(cat_indices.shape[0]):
            all_related_hidden_states = hidden_states[i, cat_indices[i]]
            sets, mix, hidden_dims = all_related_hidden_states.shape
            concatenate_hidden_states = all_related_hidden_states.view(sets, mix*hidden_dims)
            group_hidden_states = concatenate_hidden_states.view(-1, self.jacobi_token_nums, mix*hidden_dims)
            curr_states.append(group_hidden_states)
        return torch.cat(curr_states, dim=0)
    
    
    def cat_tokens_split(self, hidden_states, loss_mask):
        target_states = []
        for i in range(hidden_states.shape[0]):
            replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]  #[30, 31, 32, 33, 34, 35]
            replace_indices_groups = replace_indices.view(-1, self.jacobi_token_nums)  # [[30, 31], [32, 33], [34, 35]]
            group_counts = replace_indices_groups.shape[0]
            reference_array = torch.arange(replace_indices[0], replace_indices[0]+group_counts).view(-1, 1)
            end = group_counts - 1
            start = end + self.mix_sequences
            mix_array = torch.arange(start, end, step=-1).view(1, -1)
            prev_seq_indices_groups = (reference_array - mix_array).to(replace_indices_groups.device)  #  [[26, 27], [27, 28], [28, 29]]
            all_indices = torch.cat([prev_seq_indices_groups, replace_indices_groups], dim=-1)  #[[26, 27, 30, 31], [27, 28, 32, 33], [28, 29, 34, 35]]
            target_states.append(hidden_states[i, all_indices])  # [groups, jacobi_token_nums + mix_seq, hidden_dim]

        target_states = torch.cat(target_states, dim=0)  # [mix groups from all bs, jacobi_token_nums + mix_seq, hidden_dim]
        curr_states = torch.cat([target_states[:, -(self.jacobi_token_nums + i):-i if i > 0 else None, :] for i in range(self.mix_sequences + 1)], dim=-1)
        return curr_states

    def handle_attention_mask(self, bs, target_length, attention_mask, past_seen_tokens, loss_mask, device, dtype):
        # device, dtype = hidden_states.device, hidden_states.dtype
        min_dtype = torch.finfo(dtype).min
        is_4d_mask = attention_mask is not None and len(attention_mask.shape) == 4
        if is_4d_mask:
            previous_mask = torch.zeros((attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2], past_seen_tokens), dtype=dtype, device=device)
            causal_mask = torch.cat([previous_mask, attention_mask], dim=-1).type(dtype)
        else:
            # target_length = hidden_states.shape[1]
            deny_mask = torch.full((target_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            causal_mask = []
            # for batch_idx in range(hidden_states.shape[0]):
            for batch_idx in range(bs):
                diagonal_attend_mask = torch.arange(target_length, device=device, dtype=torch.int32)
                diagonal_attend_mask = diagonal_attend_mask > diagonal_attend_mask.reshape(-1, 1)

                replace_indices_groups = torch.nonzero(loss_mask[batch_idx] == 1, as_tuple=True)[0].view(-1, self.jacobi_token_nums)
                curr_loss_mask = loss_mask[batch_idx].repeat(diagonal_attend_mask.shape[-1], 1)
                for i in replace_indices_groups:
                    curr_loss_mask[i[0]:i[-1]+1, i[0]:i[-1]+1] = 0
                diagonal_attend_mask.bitwise_or_(curr_loss_mask.type(torch.bool))
                final_mask = deny_mask * diagonal_attend_mask
                causal_mask.append(final_mask.unsqueeze(0))
            causal_mask = torch.stack(causal_mask, dim=0)
        return causal_mask

    def update_hidden_states(self, hidden_states, new_states, loss_mask, jacobi_indices=None):
        pointer = 0
        for i in range(hidden_states.shape[0]):
            if jacobi_indices is not None:
                replace_indices = jacobi_indices[i]
            else:
                replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
            group = replace_indices.shape[0] // self.jacobi_token_nums
            bs_new_states = new_states[pointer:pointer+group]
            hidden_states[i, replace_indices] = bs_new_states.view(-1, hidden_states.shape[-1])
            
            pointer += group
            
    def forward_backbone_decoder_layer(self, decoder_layer, hidden_states, causal_mask, loss_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, jacobi_indices, cat_indices, all_hidden_states):
        residual = hidden_states
        # Self Attention
        if PERFORMANCE_CHECK:
            kwargs = {"hidden_states":hidden_states}
            hidden_states = timer.record_time("layernorm", decoder_layer.input_layernorm, **kwargs)
            kwargs = {
                "hidden_states":hidden_states,
                "attention_mask":causal_mask,
                "position_ids":position_ids,
                "past_key_value":past_key_values,
                "output_attentions":output_attentions,
                "use_cache":use_cache,
                "cache_position":cache_position,
                "position_embeddings":position_embeddings,
                "jacobi_tokens":self.jacobi_token_nums
            }
            hidden_states, self_attn_weights, present_key_value = timer.record_time("attn", decoder_layer.self_attn, **kwargs)
        else:
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                jacobi_tokens=self.jacobi_token_nums
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        
        if PERFORMANCE_CHECK:
            kwargs = {"hidden_states":hidden_states}
            hidden_states = timer.record_time("layernorm", decoder_layer.post_attention_layernorm, **kwargs)
            kwargs = {"hidden_state":hidden_states}
            hidden_states = timer.record_time("mlp", decoder_layer.mlp, **kwargs)
        else:
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # previous token projection
        layer_idx = decoder_layer.self_attn.layer_idx
        if (layer_idx + 1) % self.proj_freq == 0:
            if self.shared_adapter:
                adapter_idx = 0
            else:
                adapter_idx = layer_idx // self.proj_freq

            if self.fuse_previous_hidden_states:
                used_states = torch.cat(all_hidden_states[(-self.proj_freq+1):] + (hidden_states,), dim=-1)
            else:
                used_states = hidden_states

            if cat_indices is not None:
                if PERFORMANCE_CHECK:
                    kwargs = {
                        "hidden_states":used_states,
                        "cat_indices":cat_indices
                        }
                    curr_states = timer.record_time("cat_tokens", self.cat_tokens_with_index, **kwargs)
                else:
                    curr_states = self.cat_tokens_with_index(used_states, cat_indices)

            else:
                if self.token_sets_inline:
                    if self.mix_sequences != 1:
                        raise NotImplementedError(f"current only support concatenate wit previous token, but receive mix_sequence = {self.mix_sequences}")
                    # new_states = self.cat_tokens_inline_lagacy(hidden_states, loss_mask, adapter_idx)

                    if PERFORMANCE_CHECK:
                        kwargs = {
                        "hidden_states":used_states,
                        "loss_mask":loss_mask
                        }
                        curr_states = timer.record_time("cat_tokens", self.cat_tokens_inline, **kwargs)
                    else:
                        curr_states = self.cat_tokens_inline(used_states, loss_mask)

                else:
                    curr_states = self.cat_tokens_split(used_states, loss_mask)
            
            if self.layer_norm:
                curr_states = self.pre_adapter_layernorm(curr_states)

            # print(curr_states.shape, curr_states.dtype)
            if PERFORMANCE_CHECK:
                kwargs = {
                    "hidden_state":curr_states,
                    "idx":adapter_idx
                }
                new_states = timer.record_time("adapters", self.adapters, **kwargs)
            else:
                new_states = self.adapters(curr_states, adapter_idx)

            if PERFORMANCE_CHECK:
                kwargs = {
                    "hidden_states":hidden_states,
                    "new_states":new_states,
                    "loss_mask":loss_mask,
                    "jacobi_indices":jacobi_indices
                }
                timer.record_time("update_hidden_states", self.update_hidden_states, **kwargs)
            else:
                self.update_hidden_states(hidden_states, new_states, loss_mask, jacobi_indices)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def forward_backbone_decoder_layers(self, hidden_states, causal_mask, loss_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, output_hidden_states, jacobi_indices, cat_indices):
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.model.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = self.forward_backbone_decoder_layer(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                causal_mask=causal_mask,
                loss_mask=loss_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                jacobi_indices = jacobi_indices,
                cat_indices = cat_indices,
                all_hidden_states=all_hidden_states)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return hidden_states, all_hidden_states, next_decoder_cache, all_self_attns

    def forward_backbone_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        jacobi_indices: Optional[torch.Tensor] = None,
        cat_indices: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        (output_attentions, 
         output_hidden_states, 
         return_dict, 
         use_cache, 
         return_legacy_cache) = self.model.get_pre_setting(input_ids=input_ids, 
                                                          past_key_values=past_key_values, 
                                                          inputs_embeds=inputs_embeds, 
                                                          use_cache=use_cache, 
                                                          output_attentions=output_attentions, 
                                                          output_hidden_states=output_hidden_states, 
                                                          return_dict=return_dict)
        
        inputs_embeds = self.model.embedding(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # insert jacobi tokens
        if PERFORMANCE_CHECK:
            kwargs = {
                "inputs_embeds":inputs_embeds,
                "loss_mask":loss_mask
                }
            inputs_embeds = timer.record_time("merge_jacobi_tokens", self.merge_jacobi_tokens, **kwargs)
        else:
            inputs_embeds = self.merge_jacobi_tokens(inputs_embeds, loss_mask)

        # if model is training -> don't allow cache_position
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.ones_like(input_ids, device=input_ids.device) * -1
            jacobi_position = torch.arange(self.jacobi_token_nums, device=input_ids.device)
            for batch_idx in range(inputs_embeds.shape[0]):
                # handle normal input position
                inputs_position = torch.arange(loss_mask[batch_idx].shape[0] - loss_mask[batch_idx].sum(-1), device=input_ids.device)
                replace_indices = torch.nonzero(loss_mask[batch_idx] == 0, as_tuple=True)[0]
                cache_position[batch_idx, replace_indices] = inputs_position

                # handle jacobi tokens' position
                replace_indices = torch.nonzero(cache_position[batch_idx] == -1, as_tuple=True)[0]
                replace_indices_groups = replace_indices.view(-1, self.jacobi_token_nums)            
                prefix_position = cache_position[batch_idx, (replace_indices_groups[:, 0] - 1)].repeat(self.jacobi_token_nums, 1).transpose(-1, -2)
                true_jacobi_position = (prefix_position + jacobi_position + 1).flatten()
                cache_position[batch_idx, replace_indices] = true_jacobi_position
        
        cache_position = cache_position + past_seen_tokens
        position_ids = cache_position

        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        if PERFORMANCE_CHECK:
            kwargs = {
                "bs":hidden_states.shape[0], 
                "target_length":hidden_states.shape[1], 
                "attention_mask":attention_mask, 
                "past_seen_tokens":past_seen_tokens, 
                "loss_mask":loss_mask, 
                "device":hidden_states.device, 
                "dtype":hidden_states.dtype
            }
            causal_mask = timer.record_time("handle_attention_mask", self.handle_attention_mask, **kwargs)
        else:
            causal_mask = self.handle_attention_mask(hidden_states.shape[0], hidden_states.shape[1], attention_mask, past_seen_tokens, loss_mask, hidden_states.device, hidden_states.dtype)
        
        # decoder layers with jacobi tokens
        hidden_states, all_hidden_states, next_decoder_cache, all_self_attns = self.forward_backbone_decoder_layers(hidden_states=hidden_states, 
                                                                                                                          causal_mask=causal_mask,
                                                                                                                          loss_mask=loss_mask, 
                                                                                                                          position_ids=position_ids, 
                                                                                                                          past_key_values=past_key_values, 
                                                                                                                          output_attentions=output_attentions, 
                                                                                                                          use_cache=use_cache, 
                                                                                                                          cache_position=cache_position, 
                                                                                                                          position_embeddings=position_embeddings, 
                                                                                                                          output_hidden_states=output_hidden_states,
                                                                                                                          jacobi_indices = jacobi_indices,
                                                                                                                          cat_indices = cat_indices)
        
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        # print(hidden_states.shape, next_cache)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        jacobi_indices: Optional[torch.Tensor] = None,
        cat_indices: Optional[torch.Tensor] = None,
        num_logits_to_keep: int = 0,
        inference=False,
        **loss_kwargs,
    ) -> Union[Tuple, JacobiCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.decoding_mode == "jacobi":
            if PERFORMANCE_CHECK:
                kwargs = {
                    "input_ids":input_ids,
                    "attention_mask":attention_mask,
                    "loss_mask":loss_mask,
                    "position_ids":position_ids,
                    "past_key_values":past_key_values,
                    "inputs_embeds":inputs_embeds,
                    "use_cache":use_cache,
                    "output_attentions":output_attentions,
                    "output_hidden_states":output_hidden_states,
                    "return_dict":return_dict,
                    "cache_position":cache_position,
                    "jacobi_indices":jacobi_indices,
                    "cat_indices":cat_indices
                }
                outputs = timer.record_time("backbone_model", self.forward_backbone_model, **kwargs)
            else:
                outputs = self.forward_backbone_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loss_mask=loss_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                        jacobi_indices=jacobi_indices,
                        cat_indices=cat_indices
                        )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)  # cheap

            # self.wtf2 = torch.cat([self.wtf2, logits[:, 0, :].argmax(dim=-1)], dim=-1) if hasattr(self, "wtf2") else logits[:, -3, :].argmax(dim=-1)

            # max_sequence = 0
            jacobi_logits, jacobi_hidden_states, jacobi_all_hidden_states = None, None, None
            if not inference:
                if output_hidden_states:
                    all_hidden_states = outputs["hidden_states"]
                    jacobi_all_hidden_states = []
        
                # Iterate through the batch dimension
                jacobi_hidden_states, jacobi_logits = [], []

                for i in range(hidden_states.shape[0]):
                    replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
                    jacobi_hidden_states.append(hidden_states[i, replace_indices, :])
                    jacobi_logits.append(logits[i, replace_indices, :])
                    
                    if output_hidden_states:
                        temp = []
                        for mid_hidden_state in all_hidden_states:
                            temp.append(mid_hidden_state[i, replace_indices, :])
                        jacobi_all_hidden_states.append(torch.stack(temp, dim=0))  # (layers, seq, hidden)

                jacobi_hidden_states = torch.cat(jacobi_hidden_states, dim=0)
                jacobi_logits = torch.cat(jacobi_logits, dim=0)
        
                if output_hidden_states:
                    jacobi_all_hidden_states = torch.cat(jacobi_all_hidden_states, dim=1)  # (layers, seqs, hidden) ~= 24, 194, 896

            if not return_dict:
                output = (logits,) + outputs[1:]
                return output

            return JacobiCausalLMOutputWithPast(
                logits=logits,
                jacobi_logits=jacobi_logits,
                past_key_values=outputs.past_key_values,
                jacobi_hidden_states=jacobi_hidden_states,
                jacobi_all_hidden_states=jacobi_all_hidden_states if output_hidden_states else None,
                attentions=outputs.attentions,
            )

        elif self.decoding_mode == "naive":
            if PERFORMANCE_CHECK:
                kwargs = {
                    "input_ids":input_ids,
                    "attention_mask":attention_mask,
                    "position_ids":position_ids,
                    "past_key_values":past_key_values,
                    "inputs_embeds":inputs_embeds,
                    "use_cache":use_cache,
                    "output_attentions":output_attentions,
                    "output_hidden_states":output_hidden_states,
                    "return_dict":return_dict,
                    "cache_position":cache_position,
                }
                outputs = timer.record_time("model", self.model, **kwargs)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
            hidden_states = outputs[0]
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])  #cheap

            self.wtf = torch.cat([self.wtf, logits.argmax(dim=-1)], dim=-1) if hasattr(self, "wtf") else logits.argmax(dim=-1)
            
            loss = None
            if labels is not None:
                loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
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

        input_processor = InputProcessor(input_ids.dtype, torch.float32, input_ids.dtype, input_ids.device, self.jacobi_token_nums, self.mix_sequences)
        past_key_values = DynamicCache()
        tt = 0
        ct = {}

        # do the first inference
        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        input_ids = torch.cat([input_ids]+[padding]*self.jacobi_token_nums, dim=-1)
        loss_mask = []
        prev_index, jacobi_index = [], []
        for i in range(input_ids.shape[0]):
            jacobi_indices = torch.nonzero(input_ids[i] == -1, as_tuple=True)
            jacobi_indices_groups = jacobi_indices[0].view(-1, self.jacobi_token_nums)
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
            if self.confg.eos_token_id in current_decoded_tokens:
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
            jacobi_index_end = jacobi_index_start + self.jacobi_token_nums
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

def update_kv_cache_lagacy(output, route_indices, past_seen_tokens):
    prev_cache = torch.arange(0, past_seen_tokens)
    curr_cache = torch.tensor(route_indices) + past_seen_tokens
    cache_indices = torch.cat([prev_cache, curr_cache], dim=-1)
    for layer_idx in range(len(output["past_key_values"].key_cache)):
        output["past_key_values"].key_cache[layer_idx] = output["past_key_values"].key_cache[layer_idx][:, :, cache_indices, :]
        output["past_key_values"].value_cache[layer_idx] = output["past_key_values"].value_cache[layer_idx][:, :, cache_indices, :]
        print(output["past_key_values"].value_cache[layer_idx].shape)