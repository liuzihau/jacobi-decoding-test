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

class Qwen2JacobiForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    
    def __init__(self, config, jacobi_token_nums=2, mix_sequences=1):
        super().__init__(config)
        self.confg = config
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        attn_hidden_size = config.hidden_size
        attn_layers = config.num_hidden_layers

        self.adapters = nn.ModuleList([nn.Linear((n+2)*attn_hidden_size, attn_hidden_size) for n in range(mix_sequences)])
        self.mix_sequences = mix_sequences
        
        self.jacobi_weight = nn.Parameter(torch.ones((attn_hidden_size,), device=self.model.device, dtype=torch.float32) * 1e-5).to(dtype=torch.bfloat16)        self.jacobi_token_nums = jacobi_token_nums

        # for adapter in self.adapters:
            # self.init_weights(adapter)
        # self.init_weights(self.jacobi_weight)
        
        self.post_init()

    # def init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Parameter):
    #         module.data.normal_(mean=0.0, std=std)

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
    
    def merge_with_jacobi_tokens(self, inputs_embeds, loss_mask, jacobi_tokens=None):
        if jacobi_tokens is None:
            # Clone inputs_embeds to avoid modifying the original tensor
            modified_embeds = inputs_embeds.clone()

            # Iterate through the batch dimension
            for i in range(inputs_embeds.shape[0]):
                # Find indices where loss_mask == 1 for this batch
                replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]

                # Ensure the number of replace indices matches the jacobi tokens
                assert len(replace_indices) >= self.jacobi_token_nums, "Not enough positions in loss_mask to replace with all jacobi tokens"

                # Select the first `jacobi_token_nums` indices to replace
                replace_indices = replace_indices[:self.jacobi_token_nums]

                # Replace embeddings at the specified indices with jacobi tokens
                modified_embeds[i, replace_indices] = self.jacobi_weight

            return modified_embeds

        # if jacobi_tokens is None:
        #     jacobi_sequence = self.jacobi_weight.unsqueeze(0).unsqueeze(0).repeat(inputs_embeds.shape[0], self.jacobi_token_nums, 1)
        #     return torch.cat([inputs_embeds, jacobi_sequence], dim=1)
    
    def run_decoder_layer_with_previous_hidden_proj(self, decoder_layer, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings):
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
        if layer_idx % 4 == 0 and layer_idx > 0:
            target_states = hidden_states[:, -(self.jacobi_token_nums+self.mix_sequences):, :]
            curr_states = target_states[:, -self.jacobi_token_nums:, :]
            new_states = None
            for i in range(self.mix_sequences):
                prev_states = target_states[:, -(self.jacobi_token_nums+i+1):-(i+1), :]
                curr_states = torch.cat([curr_states, prev_states], dim=-1)
                
                if new_states is None:
                    new_states = self.adapters[i](curr_states)
                else:
                    new_states += self.adapters[i](curr_states)

            new_states /= self.mix_sequences
            hidden_states = torch.cat([hidden_states[:, :-self.jacobi_token_nums, :], new_states], dim=-2)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def run_decoder_layers_with_jacobi_tokens(self, hidden_states, causal_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings, output_hidden_states):
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.model.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.run_decoder_layer_with_previous_hidden_proj(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                causal_mask=causal_mask,
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

    def get_feature(
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

        # Qwen2Model forward
        output_attentions, output_hidden_states, return_dict, use_cache, return_legacy_cache = self.model.get_pre_setting(input_ids, 
                                                                                                                          past_key_values, 
                                                                                                                          inputs_embeds, 
                                                                                                                          use_cache, 
                                                                                                                          output_attentions, 
                                                                                                                          output_hidden_states, 
                                                                                                                          return_dict)
        inputs_embeds = self.model.run_embedding(input_ids, inputs_embeds)

        # insert jacobi tokens
        inputs_embeds = self.merge_with_jacobi_tokens(inputs_embeds, loss_mask)

        # print(cache_position, position_ids)

        # add positions for jacobi tokens
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        else:
            last_cache_num = cache_position[-1].item()
            jacobi_position = torch.arange(last_cache_num+1, last_cache_num + self.jacobi_token_nums + 1, device=inputs_embeds.device)
            cache_position = torch.cat([cache_position, jacobi_position], dim=-1)

        # if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        causal_mask = self.model._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        # decoder layers with jacobi tokens
        hidden_states, all_hidden_states, next_decoder_cache, all_self_attns = self.run_decoder_layers_with_jacobi_tokens(hidden_states=hidden_states, 
                                                                                                                          causal_mask=causal_mask, 
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
        outputs = self.get_feature(
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

        jacobi_hidden_states = []
        jacobi_logits = []
        # Iterate through the batch dimension
        for i in range(inputs_embeds.shape[0]):
            replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
            replace_indices = replace_indices[:self.jacobi_token_nums]
            # lm_hidden_states = hidden_states[:, :-self.jacobi_token_nums, :]
            jacobi_hidden_states.append(hidden_states[i, replace_indices, :])

            # lm_logits = logits[:, :-self.jacobi_token_nums, :]
            jacobi_logits.append(logits[i, replace_indices, :])
        jacobi_hidden_states = torch.stack(jacobi_hidden_states, dim=0)
        jacobi_logits = torch.stack(jacobi_logits, dim=0)
        
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