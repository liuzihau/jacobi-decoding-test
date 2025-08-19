from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput

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