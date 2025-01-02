from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

# model_name = "./Qwen2.5-0.5B-Instruct"
# # model = Qwen2ForCausalLM.from_pretrained(
# model = Qwen2JacobiForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)

# for name, param in model.named_parameters():
#     print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")



import json

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = Qwen2JacobiForCausalLM.from_pretrained(
    "./Qwen2.5-0.5B-Instruct",
    10,
    torch_dtype="auto",
    device_map="auto"
)

# freeze target model's parameter
for param in model.model.parameters():
    param.requires_grad = False

# Example usage:
# Assuming `model` is your PyTorch model
total_params = count_trainable_parameters(model)
print(f"Total trainable parameters: {total_params}")