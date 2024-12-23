from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

model_name = "./Qwen2.5-0.5B-Instruct"
# model = Qwen2ForCausalLM.from_pretrained(
model = Qwen2JacobiForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)

print(model)