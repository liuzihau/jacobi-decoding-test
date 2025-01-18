from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
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

class Qwen2JacobiForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    
    def __init__(self, config, jacobi_token_nums=2, mix_sequences=1, proj_freq=4, adapter_type='Linear', shared_adapter=True, shared_jacobi_token=True, jacobi_adapter_kwargs=None, layer_norm=False, decoding_mode="jacobi"):
        super().__init__(config)
        self.confg = config
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.mix_sequences = mix_sequences
        self.proj_freq = proj_freq
        self.jacobi_token_nums = jacobi_token_nums
        self.shared_adapter = shared_adapter
        self.shared_jacobi_token = shared_jacobi_token
        self.layer_norm = layer_norm
        self.decoding_mode = decoding_mode

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        attn_hidden_size = config.hidden_size
        attn_layers = config.num_hidden_layers
        if adapter_type == "Linear":
            adapter_module = ProjectionLinear
        elif adapter_type == 'Qwen2MLP':
            adapter_module = ProjectionQwen2MLP
        else:
            raise NotImplementedError(f"{adapter_type} hasn't been implemented")
        
        if self.shared_adapter:
            adapter_layers = 1
        else:
            adapter_layers = attn_layers // self.proj_freq
        jacobi_adapter_kwargs = {} if jacobi_adapter_kwargs is None else jacobi_adapter_kwargs        
        self.adapters = nn.ModuleList([adapter_module((n+2)*attn_hidden_size, attn_hidden_size, layers=adapter_layers, **jacobi_adapter_kwargs) for n in range(mix_sequences)])
        self.pre_adapter_layernorm = Qwen2RMSNorm(attn_hidden_size*2)

        temp_weight = torch.ones((attn_hidden_size,), device=self.model.device, dtype=torch.float32) * 1e-5
        temp_weight = temp_weight.to(dtype=torch.bfloat16)  # can be removed?
        if self.shared_jacobi_token:
            self.jacobi_weight = nn.Parameter(temp_weight)
        else:
            self.jacobi_weight = torch.stack([nn.Parameter(temp_weight)] * self.jacobi_token_nums, dim=0)
        
        self.post_init()

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
    
    def merge_jacobi_tokens(self, inputs_embeds, loss_mask, jacobi_tokens=None):
        if jacobi_tokens is None:
            # Clone inputs_embeds to avoid modifying the original tensor
            modified_embeds = inputs_embeds.clone()

            # Iterate through the batch dimension
            for i in range(inputs_embeds.shape[0]):
                # Find indices where loss_mask == 1 for this batch
                replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
                if self.shared_jacobi_token:
                    modified_embeds[i, replace_indices] = self.jacobi_weight
                else:
                    expand_weight = self.jacobi_weight.repeat(replace_indices.shape[0] // self.jacobi_token_nums, 1)
                    modified_embeds[i].scatter_(0, replace_indices.unsqueeze(-1).expand(-1, self.jacobi_weight.shape[-1]), expand_weight)

            return modified_embeds

    def forward_backbone_decoder_layer(self, decoder_layer, hidden_states, causal_mask, loss_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings):
        residual = hidden_states

        hidden_states = decoder_layer.input_layernorm(hidden_states)

        # Self Attention
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

            target_states = []
            for i in range(hidden_states.shape[0]):
                replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]  #[6, 7, 9, 10, 12, 13]
                replace_indices_groups = replace_indices.view(-1, self.jacobi_token_nums)  # [[6, 7], [9, 10], [12, 13]]
                prev_seq_indices_groups = (replace_indices_groups[:, 0] - self.mix_sequences).reshape(-1, 1)  #[[5], [8], [11]]
                all_indices = torch.cat([prev_seq_indices_groups, replace_indices_groups], dim=-1)  #[[5, 6, 7], [8, 9, 10], [11, 12, 13]]
                
                # prev_seq_indices = replace_indices[:self.mix_sequences] - self.mix_sequences
                # all_indices = torch.cat([prev_seq_indices, replace_indices], dim=-1)
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
                    new_states = self.adapters[i](curr_states, adapter_idx)
                else:
                    new_states += self.adapters[i](curr_states, adapter_idx)
            new_states /= self.mix_sequences

            pointer = 0
            for i in range(hidden_states.shape[0]):
                replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
                group = replace_indices.shape[0] // self.jacobi_token_nums
                bs_new_states = new_states[pointer:pointer+group]
                hidden_states[i, replace_indices] = bs_new_states.reshape(-1, hidden_states.shape[-1])
                pointer += group

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def forward_backbone_decoder_layers(self, hidden_states, causal_mask, loss_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, output_hidden_states):
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
                position_embeddings=position_embeddings)

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
        inputs_embeds = self.merge_jacobi_tokens(inputs_embeds, loss_mask)

        # if model is training -> don't allow cache_position
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
            
        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        # else:
        #     last_cache_num = cache_position[-1].item()
        #     jacobi_position = torch.arange(last_cache_num+1, last_cache_num + self.jacobi_token_nums + 1, device=inputs_embeds.device)
        #     cache_position = torch.cat([cache_position, jacobi_position], dim=-1)

        # if position_ids is None:
        # position_ids = cache_position.unsqueeze(0)
        position_ids = cache_position

        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        # causal_mask = self.model._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
        target_length = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype
        min_dtype = torch.finfo(dtype).min
        deny_mask = torch.full((target_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = []
        for batch_idx in range(hidden_states.shape[0]):
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
                                                                                                                          output_hidden_states=output_hidden_states)
        
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
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, JacobiCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.decoding_mode == "jacobi":
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
                )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            if output_hidden_states:
                all_hidden_states = outputs["hidden_states"]
                jacobi_all_hidden_states = []
        
            # Iterate through the batch dimension
            jacobi_hidden_states, jacobi_logits = [], []

            # max_sequence = 0
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
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

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
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False
            ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        
        max_length = max_length - self.jacobi_token_nums

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()

        # try using hugging face's kv logic
        # self.ea_layer.reset_kv()
        past_key_values = DynamicCache()


        # # Initialize the past key and value states
        # if hasattr(self, "past_key_values"):
        #     past_key_values = self.past_key_values
        #     past_key_values_data = self.past_key_values_data
        #     current_length_data = self.current_length_data
        #     # Reset the past key and value states
        #     current_length_data.zero_()
        # else:
        #     (
        #         past_key_values,
        #         past_key_values_data,
        #         current_length_data,
        #     ) = initialize_past_key_values(self.base_model)
        #     self.past_key_values = past_key_values
        #     self.past_key_values_data = past_key_values_data
        #     self.current_length_data = current_length_data

        
        # do the first inference
        input_len = input_ids.shape[1]
        input_ids = torch.cat([input_ids]+[padding]*self.jacobi_token_nums, dim=-1)
        loss_mask = []
        prev_index = []
        jacobi_index = []
        for i in range(input_ids.shape[0]):
            jacobi_indices = torch.nonzero(input_ids[i] == -1, as_tuple=True)
            jacobi_indices_groups = jacobi_indices[0].view(-1, self.jacobi_token_nums)
            prev_index.append(jacobi_indices_groups[:, 0] - 1)
            jacobi_index.append(jacobi_indices_groups)
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
            return_dict=True
            )
        for i in range(output["logits"].shape[0]):
            _, top3_pred_normal = output["logits"][i, prev_index[i]].topk(3, -1, True, True) 
            _, top3_pred_jacobi = output["logits"][i, jacobi_index[i]].topk(3, -1, True, True) 
            print(top3_pred_normal, top3_pred_jacobi)

        # _, pred = output["jacobi_logits"].topk(3, -1, True, True)
        # print(pred.shape)
        # print(pred)
        # for key_cache in output.past_key_values.key_cache:
        #     print(key_cache.shape)

        reset_tree_mode(self)
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

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


def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers

    devices=[]
    for i in range(config.num_hidden_layers):
        try:
            device = model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device=model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    past_key_values_data_list=[]
    startnum=0
    startdevice=devices[0]
    for id,i in enumerate(devices):
        if startdevice!=i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                config.max_position_embeddings,
                config.hidden_size // config.num_attention_heads,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum=0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        config.max_position_embeddings,
        config.hidden_size // config.num_attention_heads,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers

    bias=0
    start_data_m=devices[0].index
    for i in range(config.num_hidden_layers):
        data_m=devices[i].index
        if data_m!=start_data_m:
            bias=0
            start_data_m=data_m
        try:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[data_m-devices[0].index][2*bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[0][2 * bias + j],
                            current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        bias+=1
    return past_key_values, past_key_values_data_list, current_length_data



