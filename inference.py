import json
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from accelerate.utils import set_seed

from tools.data_processing import InferenceDataset, list_files
from tools.utils import load_jacobi_weight
from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

def naive_decoding(model, input_ids, attention_mask, max_new_tokens=128, do_sample=False, num_beams=1, top_p=1.0, top_k=50, repetition_penalty=1.0, temperature=1.0):
    model.decoding_mode = "naive"
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,  # Disable repetition penalty
        temperature=temperature,  # Ensure no scalin
        )
    return generated_ids[:, input_ids.shape[1]:]

def jacobi_decoding(model, input_ids, max_new_tokens=128, do_sample=False, num_beams=1, top_p=1.0, top_k=50, repetition_penalty=1.0, temperature=1.0, force_autoregressive=False):
    model.decoding_mode = "jacobi"
    generated_ids, tt, ct = model.jagenerate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,  # Disable repetition penalty
        temperature=temperature,
        force_autoregressive=force_autoregressive
        )
    return generated_ids, tt, ct

# environment
# CONFIG_PATH = './configs/inference_config_local.json'
CONFIG_PATH = './configs/inference_config_colab.json'
with open(CONFIG_PATH, 'r') as f:
    inference_config = json.loads(f.read())
set_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

# model / tokenizer
tokenizer = Qwen2TokenizerFast.from_pretrained(inference_config["basepath"], use_fast=False)
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

# load weight
print(f"Loading model states in {inference_config['statepath']}")
load_jacobi_weight(model, f"{inference_config['statepath']}/model.safetensors")
print("State restored successfully!")

model = model.to('cuda')
model = model.to(torch.float32)
model.eval()

# data part
datapath = list_files(inference_config["datapath"])
testdatapath = datapath[int(len(datapath) * inference_config["test_data_portion"]):]
testdataset = InferenceDataset(testdatapath)
test_loader = DataLoader(testdataset, batch_size=inference_config["bs"], shuffle=False, 
                         num_workers=inference_config["num_workers"], pin_memory=True)

# evaluate
prefix = torch.tensor([[151644,  77091,    198]])
n_tokens, ja_tokens, j_tokens, n_time, ja_time, j_time = 0, 0, 0, 0, 0, 0
total, correct =0, {}
for batch_idx, data in enumerate(tqdm(test_loader)):
    text = torch.cat([data, prefix], dim=-1).to(model.device)
    attention_mask = torch.ones_like(text)

    naive_start = time.time()
    n = naive_decoding(model, text, attention_mask, inference_config["max_new_tokens"], **inference_config["naive_kwargs"])
    naive_delta = time.time() - naive_start
    n_tokens += n.shape[1]
    n_time += naive_delta
    
    jacobi_ar_start =  time.time()
    ja, _, _ = jacobi_decoding(model, text, inference_config["max_new_tokens"], force_autoregressive=True, **inference_config["naive_kwargs"])
    jacobi_ar_delta = time.time() - jacobi_ar_start
    ja_tokens += ja.shape[1]
    ja_time += jacobi_ar_delta
    
    jacobi_start =  time.time()
    j, tt, ct = jacobi_decoding(model, text, inference_config["max_new_tokens"], force_autoregressive=False, **inference_config["naive_kwargs"])
    jacobi_delta = time.time() - jacobi_start
    j_tokens += j.shape[1]
    j_time += jacobi_delta
    total += tt
    for key in ct:
        if key in correct:
            correct[key] += ct[key]
        else:
            correct[key] = ct[key]

    if batch_idx % 100 == 0:
        print(f"batch {batch_idx}")
        print("="*50)
        print(f"naive decoding:")
        print(f"inference time: {naive_delta:.3f}")
        res = tokenizer.decode(n[0].detach().cpu().tolist())
        print(f"Res: {res}")    
        print("="*50)
        print(f"speculative decoding:")
        print(f"inference time: {jacobi_delta:.3f}")
        res = tokenizer.decode(j[0].detach().cpu().tolist())
        print(f"Res: {res}")

report = {
    "jacobi_decoding_chances": total,
    "naive_tokens": n_tokens,
    "naive_time": n_time,
    "jacobi_tokens": j_tokens,
    "jacobi_time": j_time,
    "jacobi_ar_tokens": ja_tokens,
    "jacobi_ar_time": ja_time
}

print(total, correct)
print(f"[token generated] naive: {n_tokens}, jacobi_ar: {ja_tokens}, jacobi: {j_tokens}")
print(f"[final speed compare] naive: {n_tokens / n_time:.3f}(tokens/sec), jacobi_ar: {ja_tokens / ja_time:.3f}(tokens/sec), jacobi: {j_tokens / j_time:.3f}(tokens/sec)")

keys = sorted(correct.keys(), key=lambda x: (len(x), x[0]))
acceptance_rate = {}
for key in keys:
    if len(key) == 1:
        token_name = key[0]
        acceptance_rate[token_name] = {}
        acceptance_rate[token_name]['accept_rate'] = correct[key] / total
    elif len(key) == 2:
        token_name = key[0]
        sub_token_name = key[1]
        acceptance_rate[token_name][sub_token_name] = correct[key] / correct[(key[0],)]

report["acceptance_rate"] = acceptance_rate
with open(inference_config["reportpath"], "w") as f:
    json.dump(report, f, indent=4)  # Use indent for pretty formatting

for key in report:
    print(f"{key}")
    if key == 'acceptance_rate':
        for k in report[key]:
            print(f"{k}:")
            print(f"\t{report[key][k]}")
    else:
        print(f"\t{report[key]}")
