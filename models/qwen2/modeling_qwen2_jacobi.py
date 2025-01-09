from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput

from models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2Model


@dataclass
class JacobiCausalLMOutputWithPast(ModelOutput):
    
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    jacobi_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    jacobi_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Qwen2MLP(nn.Module):
    def __init__(self, input_size, output_size, intermediate_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.intermediate_size = int(input_size * intermediate_ratio) if intermediate_ratio is not None else input_size * 2
        self.hidden_size = output_size
        self.gate_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

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
    def __init__(self, input_size, output_size, intermediate_ratio=None, layers=1):
        super().__init__()
        self.module_list = nn.ModuleList([Qwen2MLP(input_size, output_size, intermediate_ratio) for _ in range(layers)])
    
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
    
    def __init__(self, config, jacobi_token_nums=2, mix_sequences=1, proj_freq=4, adapter_type='Linear', shared_adapter=True, shared_jacobi_token=True, adapter_kwargs=None):
        super().__init__(config)
        self.confg = config
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.mix_sequences = mix_sequences
        self.proj_freq = proj_freq
        self.jacobi_token_nums = jacobi_token_nums
        self.shared_adapter = shared_adapter
        self.shared_jacobi_token = shared_jacobi_token
        
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
        adapter_kwargs = {} if adapter_kwargs is None else adapter_kwargs        
        self.adapters = nn.ModuleList([adapter_module((n+2)*attn_hidden_size, attn_hidden_size, layers=adapter_layers, **adapter_kwargs) for n in range(mix_sequences)])
        

        temp_weight = torch.ones((attn_hidden_size,), device=self.model.device, dtype=torch.float32) * 1e-5
        temp_weight = temp_weight.to(dtype=torch.bfloat16)  # can be remove?

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
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        hidden_dim = hidden_states.shape[-1]
        logits_dim = logits.shape[-1]

        # Iterate through the batch dimension
        jacobi_hidden_states, jacobi_logits = [], []
        # max_sequence = 0
        for i in range(hidden_states.shape[0]):
            replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
            # curr_sequence = replace_indices.shape[0]
            # if curr_sequence> max_sequence:
            #     max_sequence = curr_sequence
            jacobi_hidden_states.append(hidden_states[i, replace_indices, :])
            jacobi_logits.append(logits[i, replace_indices, :])
        
        # for i in range(jacobi_hidden_states):
        #     curr_sequence = jacobi_hidden_states[i].shape[0]
        #     if curr_sequence < max_sequence:
        #         seq_pad_hidden = torch.zeros((1, (max_sequence - curr_sequence), hidden_dim)) 
        #         seq_pad_target = torch.zeros((1, (max_sequence - curr_sequence), logits_dim))
        #         jacobi_hidden_states[i] = torch.cat([jacobi_hidden_states[i], seq_pad_hidden], dim=0)
        #         jacobi_logits[i] = torch.cat([jacobi_logits[i], seq_pad_target], dim=0)
        jacobi_hidden_states = torch.cat(jacobi_hidden_states, dim=0)
        jacobi_logits = torch.cat(jacobi_logits, dim=0)
        
        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(jacobi_logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            # return (loss,) + output if loss is not None else output
            return output

        return JacobiCausalLMOutputWithPast(
            # loss=loss,
            # logits=lm_logits,
            jacobi_logits=jacobi_logits,
            past_key_values=outputs.past_key_values,
            # hidden_states=lm_hidden_states,
            jacobi_hidden_states=jacobi_hidden_states,
            attentions=outputs.attentions,
        )