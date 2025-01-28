import json

import torch
from accelerate.utils import set_seed
from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2 import Qwen2Tokenizer


# environment
CONFIG_PATH = './configs/inference_config_local.json'
# CONFIG_PATH = './configs/inference_config_colab.json'
with open(CONFIG_PATH, 'r') as f:
    inference_config = json.loads(f.read())
set_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

tokenizer = Qwen2Tokenizer.from_pretrained(inference_config["basepath"], use_fast=False)
model = Qwen2JacobiForCausalLM.from_pretrained(
    pretrained_model_name_or_path=inference_config["basepath"],
    jacobi_token_nums=inference_config["jacobi_token_nums"],
    mix_sequences=inference_config["mix_sequences"],
    proj_freq=inference_config["projection_frequency"],
    adapter_type=inference_config["adapter_type"],
    shared_adapter=inference_config["shared_adapter"],
    shared_jacobi_token=inference_config["shared_jacobi_token"],
    jacobi_adapter_kwargs=inference_config["jacobi_adapter_kwargs"],
    torch_dtype="auto",
    device_map="auto"
)

