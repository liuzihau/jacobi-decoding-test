import time
from models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
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

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

s = time.time()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32
)
delta = time.time() - s

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

generated_token_nums = generated_ids[0].shape
print(delta, generated_token_nums)
# ar 23.408718585968018 torch.Size([32])
# 01 37.196016788482666 torch.Size([32])
# 10 146.281005859375 torch.Size([32])