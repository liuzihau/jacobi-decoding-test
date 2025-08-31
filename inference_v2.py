from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.tata import TaTa
from configs.config import load_train_config, load_infer_config
from tools.utils import load_jacobi_weight
from tools.data_processing_v2 import load_all_test_datasets, collate_data

# import torch, os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for one run to get a precise stack

def load_infer_model(infer_cfg):
    tr_cfg = load_train_config(f"{infer_cfg.model.path}/{infer_cfg.model.cfg_name}")
    infer_cfg.data.pad_token_id = tr_cfg.data.pad_token_id
    tokenizer = AutoTokenizer.from_pretrained(tr_cfg.model.basepath, use_fast=False)
    model = TaTa(
        pretrained_model_name_or_path=tr_cfg.model.basepath,
        num_jacobi_tokens=tr_cfg.model.num_jacobi_tokens,
        num_prev_sequences=tr_cfg.model.num_prev_sequences,
        token_sets_inline=infer_cfg.model.token_sets_inline,
        adapter_insertion_freq=tr_cfg.model.adapter_insertion_freq,
        adapter_type=tr_cfg.model.adapter_type,
        shared_adapter=tr_cfg.model.shared_adapter,
        fuse_prev_hidden_states=tr_cfg.model.fuse_prev_hidden_states,
        fuse_jacobi_with_prev_sample=tr_cfg.model.fuse_jacobi_with_prev_sample,
        shared_jacobi_token=tr_cfg.model.shared_jacobi_token,
        use_pre_layer_norm=tr_cfg.model.use_pre_layer_norm,
        jacobi_adapter_kwargs=tr_cfg.model.jacobi_adapter_kwargs,
        device_map="balanced",#"auto",
        precision=tr_cfg.train.mixed_precision,
        pad_token_id=tr_cfg.data.pad_token_id
    )
    load_jacobi_weight(model, f"{infer_cfg.model.path}/{infer_cfg.model.state}/{infer_cfg.model.weight_name}")
    return model, tokenizer

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

def tata_decoding(model, input_ids, loss_mask, infer_cfg, force_autoregressive=False, tokenizer=None):
    model.decoding_mode = "jacobi"
    generated_ids, tt, ct = model.tagenerate(
        input_ids=input_ids,
        loss_mask=loss_mask,
        max_new_tokens=infer_cfg.infer.max_new_tokens,
        do_sample=infer_cfg.infer.do_sample,
        top_p=infer_cfg.infer.top_p,
        top_k=infer_cfg.infer.top_k,
        repetition_penalty=infer_cfg.infer.repetition_penalty,  # Disable repetition penalty
        temperature=infer_cfg.infer.temperature,
        force_autoregressive=force_autoregressive,
        tokenizer=tokenizer
        )
    return generated_ids, tt, ct


CONFIG_PATH = './configs/infer_cfg.yaml'
infer_cfg = load_infer_config(CONFIG_PATH)

model, tokenizer = load_infer_model(infer_cfg)

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", use_fast=False)
all_datasets = load_all_test_datasets(infer_cfg.data.test_data_path, tokenizer, model.num_jacobi_tokens, model.pad_token_id)

for data_name in all_datasets:
    collate_fn=partial(collate_data, pad_id=infer_cfg.data.pad_token_id)
    test_loader = DataLoader(
        all_datasets[data_name], 
        batch_size=infer_cfg.infer.bs, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=infer_cfg.data.num_workers, 
        pin_memory=True
        )
    for batch in test_loader:
        j, tt, ct = tata_decoding(model, batch["input_ids"], batch["loss_mask"], infer_cfg, False, tokenizer)
        print("-"*60)
        # for i in j:
        #     print(tokenizer.decode(i))
        break
        # res = model.causal.generate(batch["input_ids"][:,:-3], do_sample=False, max_new_tokens=7)
        # s = ""
        # for b in res.tolist():
        #     for idx, t in enumerate(b):
        #         k = tokenizer.decode(t)
        #         if idx < batch["input_ids"][:, :-3].shape[-1]:
        #             s += k
        #         else:
        #             s += f"[{k.replace("\n", "\\n")}] "
        # print(s)
        # break

