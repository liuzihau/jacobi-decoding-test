import os
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

CONFIG_PATH = './configs/data_generate_config.json'


def build_dataset_rank(tokenizer, ge_config, split="train", select=None):
    ds = load_dataset(f"{ge_config["data_path"]}/{ge_config["data_name"]}", data_files='ShareGPT_V3_unfiltered_cleaned_split.json')
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(ge_config["data_start"], ge_config["data_end"]))
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": []
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                "content": "You are a helpful assistant."},
            ]
            convroles=["user","assistant"]
            roles = {
                "human": "user", 
                "user": "user", 
                "gpt": "assistant", 
                "chatgpt": "assistant", 
                "bard": "assistant",
                "bing": "assistant", 
                "system": "system"
                }
            source= examples['conversations'][i]
            if len(source) == 0:
                continue
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            continuous_role = False
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                if role != convroles[j % 2]:
                    continuous_role = True
                    break
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            if continuous_role:
                continue
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id
            for i, turn in enumerate(messages[:-1]):
                if turn["role"] == "system":
                    continue
                elif turn["role"] == "user":
                    conversation= tokenizer.apply_chat_template(
                        messages[:i+1],
                        tokenize=False,
                        add_generation_prompt=False
                        )
                    input_ids = tokenizer(
                        conversation,
                        return_tensors="pt",
                        max_length=2048,
                        add_special_tokens=False
                        ).input_ids[0]
                    if input_ids.shape[0] + ge_config["jacobi_tokens"] > ge_config["max_len"]:
                        break
                    new_examples["conversation"].append(conversation)
                    new_examples["input_ids"].append(input_ids[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        #num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
        )

    ds1.set_format(type="torch")
    return ds1

@torch.no_grad()
def ge(data, model):
    input_ids=data["input_ids"]
    outs_big = model(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    probs = torch.softmax(outs_big.logits, dim=-1)
    td={
        "input_ids":input_ids.cpu()[0],
        "hidden_state":hidden_state_big.cpu()[0],
        # "loss_mask":data["loss_mask"].cpu()[0]
        }
    return td

@torch.no_grad()
def ge_jacobi(data, model, token_nums=100, do_sample=False, temperature=0.7):
    input_ids = data["input_ids"].cuda()

    generated_tokens = []
    hidden_states = []
    current_input = input_ids
    past_key_values = None

    for _ in range(token_nums):
        outputs = model(
            input_ids=current_input,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

        # Collect hidden states for the current token
        last_layer_hidden_states = outputs.hidden_states[-1]  # last layer'
        hidden_states.append(last_layer_hidden_states[:, -1, :])  # new token's hidden state

        # Update cached past_key_values
        past_key_values = outputs.past_key_values

        # Use greedy decoding or sampling to select the next token
        logits = outputs.logits[:, -1, :]
        if do_sample:
            next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)[0]  # sampling
        else:
            next_token = torch.argmax(logits, dim=-1)  # Greedy decoding

        generated_tokens.append(next_token)
        current_input = next_token.unsqueeze(-1)

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    hidden_states = torch.cat(hidden_states, dim=0)

    td = {
        "input_ids": input_ids[0].cpu(),
        "generated_tokens": generated_tokens.cpu(),
        "hidden_states": hidden_states.cpu()
    }
    return td

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')


def main():
    with open(CONFIG_PATH, 'r') as f:
        ge_config = json.loads(f.read())

    out_dir = f'{ge_config["output_folder"]}/{ge_config["data_name"]}_{ge_config["model_name"]}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # generate data for inference
    model_card = f"{ge_config["model_path"]}/{ge_config["model_name"]}"

    tokenizer = AutoTokenizer.from_pretrained(model_card, use_fast=False)
    dataset = build_dataset_rank(tokenizer, ge_config)
    model = AutoModelForCausalLM.from_pretrained(model_card, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()

    for id, data in enumerate(tqdm(dataset)):
        if data['input_ids'].shape[0] + ge_config["jacobi_tokens"] >= ge_config["max_len"]:
            print(f"sample[{id}] has length {data['input_ids'].shape[0]}, which is too long for training, discard")
            continue
        outdata = ge_jacobi(data, model, ge_config["jacobi_tokens"], do_sample=True)
        writedata(out_dir, outdata)