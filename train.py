import os
import gc
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from transformers import get_linear_schedule_with_warmup
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

from tools.data_processing_v2 import JacobiDatasetV2, JacobiCollatorV2
from tools.utils import top_accuracy, save_trainable_weights
from models.tata import TaTa
from configs.config import TDataCfg, TMetaCfg, TModelCfg, TrainCfg, TrainScriptConfig, to_obj     


def load_config(path: str) -> TrainScriptConfig:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed, but a YAML config was provided.")
        obj = yaml.safe_load(text)
    else:
        obj = json.loads(text)

    return TrainScriptConfig(
        meta=to_obj(obj["meta"], TMetaCfg),
        data=to_obj(obj["data"], TDataCfg),
        model=to_obj(obj["model"], TModelCfg),
        train=to_obj(obj["train"], TrainCfg),
    )


class Criterion:
    def __init__ (self, jcb_nums, device,
                  method="SmoothL1Loss", pratio=0.2, vratio=1.0,
                  do_clip=False, clip_max=1e2, clip_min=-1e2,
                  gamma_weight=None,
                  jcb_weight_reg=False, jratio=1e-5,
                  adp_weight_reg=False, aratio=1e-5,
                  ):
        self.jcb_nums = jcb_nums
        self.device = device
        self.set_ploss_method(pratio)
        self.set_vloss_method(method, vratio)
        self.set_clip_parameter(do_clip, clip_max, clip_min)
        self.set_decay_weight(gamma_weight)
        self.set_jcb_weight_reg(jcb_weight_reg, jratio)
        self.set_adp_weight_reg(adp_weight_reg, aratio)
        self.ploss = 0
        self.vloss = 0
        self.adp_reg = 0
        self.jcb_reg = 0
 
    def set_ploss_method(self, ratio):
        self.ploss_ratio = ratio

    def set_vloss_method(self, method, ratio):
        if method == "SmoothL1Loss":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        else:
            raise NotImplementedError(f"{method} has not implement")
        self.vloss_ratio = ratio

    def set_clip_parameter(self, do_clip, clip_max, clip_min):
        self.do_clip = do_clip
        self.clip_max = clip_max
        self.clip_min = clip_min
    
    def set_decay_weight(self, gamma_weight):
        """
        Give a weighted scale to all different jcb tokens
        For example, a weight [1.0, 0.9, 0.8] for 3 jcb tokens means the first jcb token has 1.0 loss back,
        while the second has 0.9 and the third has 0.8
        """
        if gamma_weight is None:
            self.weight = torch.ones(self.jcb_nums, device=self.device)
        else:
            self.weight = gamma_weight.to(self.device)

    def set_jcb_weight_reg(self, jcb_weight_reg, ratio):
        self.jcb_weight_reg = jcb_weight_reg
        self.jcb_weight_reg_ratio = ratio

    def set_adp_weight_reg(self, adp_weight_reg, ratio):
        self.adp_weight_reg = adp_weight_reg
        self.adp_weight_reg_ratio = ratio

    def clip(self, tensor):
        return torch.clamp(tensor, min=self.clip_min, max=self.clip_max)
    
    def compute_ploss(self, logits_tgt, logits_jcb):
        # cross entropy -> sample distribution difference
        target_p = nn.Softmax(dim=-1)(logits_tgt)
        out_logp = nn.LogSoftmax(dim=-1)(logits_jcb)
        plogp = target_p * out_logp
        plogp = plogp.view(-1, self.jcb_nums, plogp.shape[-1]).to(self.weight.device)
        plogp_weighted = plogp * self.weight.view(1, -1, 1)
        return -torch.sum(plogp_weighted) / (plogp_weighted.shape[0] * plogp_weighted.shape[1] + 1e-5)  # Normalize by batch and sequence

    def compute_vloss(self, hidden_tgt, hidden_jcb):
        # regression -> hidden states difference
        vloss_full = self.criterion(hidden_jcb, hidden_tgt)
        vloss_full = vloss_full.view(-1, self.jcb_nums, vloss_full.shape[-1])
        vloss_full = torch.mean(vloss_full, dim=-1).to(self.weight.device)
        vloss_weighted = vloss_full * self.weight.view(1, -1)
        return torch.sum(vloss_weighted) / (vloss_weighted.shape[0] * vloss_weighted.shape[1] + 1e-5)

    def compute_loss(self, hidden_tgt, logits_tgt, hidden_jcb, logits_jcb, adps_weight=None, jcb_weight=None):
        hidden_tgt = hidden_tgt.to(self.device)
        if self.do_clip:
            logits_tgt = self.clip(logits_tgt)
            logits_jcb = self.clip(logits_jcb)
            hidden_jcb = self.clip(hidden_jcb)
            hidden_tgt = self.clip(hidden_tgt)

        ploss = self.compute_ploss(logits_tgt, logits_jcb)
        vloss = self.compute_vloss(hidden_tgt, hidden_jcb)
        loss = self.ploss_ratio* ploss + self.vloss_ratio * vloss
        
        self.ploss = ploss.item()
        self.vloss = vloss.item()
        
        if self.adp_weight_reg and adps_weight is not None:
            adp_reg = 0
            for adp_weight in adps_weight:
                adp_reg += torch.sum(adp_weight.float() ** 2) ** (1/2) / (adp_weight.shape[0] * adp_weight.shape[1] + 1e-5)
            loss += self.adp_weight_reg_ratio * adp_reg
            self.adp_reg = adp_reg.item()

        if self.jcb_weight_reg and jcb_weight is not None:
            jcb_reg = torch.sum(torch.abs(jcb_weight.float())) / (jcb_weight.shape[-1] + 1e-5)   # L1 regularization
            loss += self.jcb_weight_reg_ratio * jcb_reg
            self.jcb_reg = jcb_reg.item()
    
        return loss


def load_model(tr_cfg):
    # if tr_cfg.train.mixed_precision == "no":
    #     torch_dtype = torch.float32
    # elif tr_cfg.train.mixed_precision == "fp16":
    #     torch_dtype = torch.float16
    # elif tr_cfg.train.mixed_precision == "bf16":
    #     torch_dtype = torch.bfloat16
    # else:
    #     raise NotImplementedError(f"unknown precision: {tr_cfg.train.mixed_precision}")

    model = TaTa(
        pretrained_model_name_or_path=tr_cfg.model.basepath,
        num_jacobi_tokens=tr_cfg.model.num_jacobi_tokens,
        num_prev_sequences=tr_cfg.model.num_prev_sequences,
        token_sets_inline=tr_cfg.model.token_sets_inline,
        adapter_insertion_freq=tr_cfg.model.adapter_insertion_freq,
        adapter_type=tr_cfg.model.adapter_type,
        shared_adapter=tr_cfg.model.shared_adapter,
        fuse_prev_hidden_states=tr_cfg.model.fuse_prev_hidden_states,
        shared_jacobi_token=tr_cfg.model.shared_jacobi_token,
        use_pre_layer_norm=tr_cfg.model.use_pre_layer_norm,
        jacobi_adapter_kwargs=tr_cfg.model.jacobi_adapter_kwargs,
        device_map="balanced",#"auto",
        precision=tr_cfg.train.mixed_precision
    )

    # model = Qwen2JacobiForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path=tr_cfg.model.basepath,
    #     num_jacobi_tokens=tr_cfg.model.num_jacobi_tokens,
    #     num_prev_sequences=tr_cfg.model.num_prev_sequences,
    #     token_sets_inline=tr_cfg.model.token_sets_inline,
    #     adapter_insertion_freq=tr_cfg.model.adapter_insertion_freq,
    #     adapter_type=tr_cfg.model.adapter_type,
    #     shared_adapter=tr_cfg.model.shared_adapter,
    #     fuse_prev_hidden_states=tr_cfg.model.fuse_prev_hidden_states,
    #     shared_jacobi_token=tr_cfg.model.shared_jacobi_token,
    #     use_pre_layer_norm=tr_cfg.model.use_pre_layer_norm,
    #     jacobi_adapter_kwargs=tr_cfg.model.jacobi_adapter_kwargs,
    #     torch_dtype=torch_dtype,
    #     device_map="auto",
    #     precision=tr_cfg.train.mixed_precision
    # )
    
    for name, p in model.named_parameters():
        if "jacobi" not in name:
            p.requires_grad = False
        print(f"{name:<70} shape={tuple(p.shape)}  device={p.device}  dtype={p.dtype}  grad={p.requires_grad}")

    initialise_method = 'kaiming'
    for name, param in model.named_parameters():
        if param.requires_grad:   
            model.init_trainable_weights(name, param, initialise_method)
    

    from collections import defaultdict
    ta = accelerator.unwrap_model(model) if "accelerator" in globals() else model
    by_dev = defaultdict(int)
    for n, p in ta.named_parameters():
        if p.requires_grad:
            by_dev[str(p.device)] += p.numel() * p.element_size()
    def pretty_bytes(n):
        for unit in ["B","KB","MB","GB","TB"]:
            if n < 1024: return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"
    print("\nTrainable bytes by device:")
    for dev, b in sorted(by_dev.items()):
        print(f"  {dev}: {pretty_bytes(b)}  (est peak with Adam+grads ~ {pretty_bytes(b*3)})")
    return model


CONFIG_PATH = './configs/train_cfg.yaml'

set_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
tr_cfg = load_config(CONFIG_PATH) 

accelerator = Accelerator(mixed_precision=tr_cfg.train.mixed_precision, gradient_accumulation_steps=tr_cfg.train.gradient_accumulation_steps)
if accelerator.is_main_process:
    wandb.login(key=tr_cfg.meta.api_key)
    wandb.init(project=tr_cfg.meta.project, name=tr_cfg.meta.name, config=tr_cfg)
    if not os.path.exists(tr_cfg.meta.cpdir):
        os.makedirs(tr_cfg.meta.cpdir)

tokenizer = Qwen2Tokenizer.from_pretrained(tr_cfg.model.basepath, use_fast=False)
model = load_model(tr_cfg)
v_model = model.lm_head.out_features

# data part
traindataset = JacobiDatasetV2(
    tr_cfg.data.tr_path, 
    jacobi_J=tr_cfg.model.num_jacobi_tokens, 
    schedule=tr_cfg.data.schedule, 
    pad_id=tr_cfg.data.pad_token_id, 
    vocab_size=v_model,
    dtype=tr_cfg.train.mixed_precision
    )

testdataset = JacobiDatasetV2(
    tr_cfg.data.te_path, 
    jacobi_J=tr_cfg.model.num_jacobi_tokens, 
    schedule=tr_cfg.data.schedule, 
    pad_id=tr_cfg.data.pad_token_id, 
    vocab_size=v_model,
        dtype=tr_cfg.train.mixed_precision
    )

train_loader = DataLoader(
    traindataset, 
    batch_size=tr_cfg.train.bs, 
    shuffle=True,
    collate_fn=JacobiCollatorV2(pad_id=151643), 
    num_workers=tr_cfg.data.num_workers,
    pin_memory=True
    )

test_loader = DataLoader(
    testdataset, 
    batch_size=tr_cfg.train.bs, 
    shuffle=False,
    collate_fn=JacobiCollatorV2(pad_id=151643), 
    num_workers=tr_cfg.data.num_workers, 
    pin_memory=True
    )


# loss / optimizer setting
if tr_cfg.train.gamma is None:
    gamma_weight = None
else:
    w_dtype = model.lm_head.weight.dtype
    base = torch.tensor(tr_cfg.train.gamma, device=model.device, dtype=w_dtype)
    exp  = torch.arange(tr_cfg.model.num_jacobi_tokens, device=model.device, dtype=w_dtype)
    gamma_weight = torch.pow(base, exp)

criterion = Criterion(
                tr_cfg.model.num_jacobi_tokens, 
                model.model.device,
                method=tr_cfg.train.loss_method, 
                pratio=tr_cfg.train.pratio, 
                vratio=tr_cfg.train.vratio,
                do_clip=tr_cfg.train.do_clip, 
                clip_max=tr_cfg.train.clip_max, 
                clip_min=tr_cfg.train.clip_min,
                gamma_weight=gamma_weight,
                jcb_weight_reg=tr_cfg.train.jcb_weight_reg, 
                jratio=tr_cfg.train.jratio,
                adp_weight_reg=tr_cfg.train.adp_weight_reg, 
                aratio=tr_cfg.train.aratio
            )

optimizer = optim.AdamW(model.parameters(), lr=tr_cfg.train.lr, betas=(tr_cfg.train.b1, tr_cfg.train.b2))

num_epochs = tr_cfg.train.num_epochs
num_warmup_steps = tr_cfg.train.num_warmup_steps
total_steps = tr_cfg.train.total_steps
is_warmup = tr_cfg.train.is_warmup
num_jacobi_tokens = tr_cfg.model.num_jacobi_tokens
debug_mode = tr_cfg.meta.debug_mode

# if is_warmup:
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

#     model, optimizer, criterion, scheduler = accelerator.prepare(#, train_loader, test_loader = accelerator.prepare(
#         model, optimizer, criterion, scheduler#, train_loader, test_loader
#     )
# else:
#     model, optimizer, criterion  = accelerator.prepare(#, train_loader, test_loader = accelerator.prepare(
#         model, optimizer, criterion#, train_loader, test_loader
#     )

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

# def print_hook_device_map(root):
#     for name, m in root.named_modules():
#         hk = getattr(m, "_hf_hook", None)
#         if hk is not None and getattr(hk, "execution_device", None) is not None:
#             print(f"{name} -> exec:{hk.execution_device}")

# unwrapped = accelerator.unwrap_model(model)   # TaTa
# print_hook_device_map(unwrapped)        # the Qwen2 backbone inside TaTa

if tr_cfg.train.statepath is not None:
    # Load accelerator state
    print("Loading model, optimizer, and scheduler states in {tr_cfg['statepath']}")
    accelerator.load_state(tr_cfg.train.statepath)

    # Restore random states
    # random_state_file = os.path.join(tr_cfg["statepath"], "random_states_0.pkl")
    # with open(random_state_file, "rb") as f:
    #     random_states = pickle.load(f)

    # torch.random.set_rng_state(random_states["torch"])
    # torch.cuda.random.set_rng_state(random_states["cuda"])

    print("State restored successfully!")

continuous_loss_nan = 0
epoch_counts = []
previous_data = []
for epoch in range(num_epochs + 1):
    top_3acc = [[0 for _ in range(num_jacobi_tokens)] for _ in range(3)]
    correct = [0 for _ in range(num_jacobi_tokens)]
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    counts = torch.zeros((num_jacobi_tokens, v_model), dtype=torch.int32, device=model.device)
    for batch_idx, data in enumerate(tqdm(train_loader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            output = model(input_ids=data["input_ids"], 
                            attention_mask=data["attention_mask"],
                            loss_mask=data["loss_mask"],
                            use_cache=False,
                            output_hidden_states=True,
                            return_dict=True)
            
            if torch.isnan(output['jacobi_hidden_states']).any():
                continuous_loss_nan += 1
                print(f"outputs contain nan : {data['filenames']}")
                print(f"previous data : {previous_data}")
                for name, param in model.named_parameters():
                    if "model." in name:
                        continue
                    print(f"[{name}] contain nan: {torch.isnan(param).any()}")
                    print(param.numel(), torch.abs(param).max(), torch.abs(param).min(), torch.abs(param).sum())

                for i, tensor in enumerate(output['jacobi_all_hidden_states']):
                    print(f"[layer {i}] contain nan: {torch.isnan(tensor).any()}")
                    print(tensor.numel(), torch.abs(tensor).max(), torch.abs(tensor).min(), torch.abs(tensor).sum())

                gc.collect()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                if continuous_loss_nan >= 3:
                    break
                continue
            continuous_loss_nan = max(0, continuous_loss_nan-1)
            previous_data = data['filenames']

            with torch.no_grad():
                tgt = data["hidden_state_target"]#.to(model.lm_head.weight.dtype)
                target_head = model.lm_head(tgt)#.detach()
                
            if debug_mode:
                print("="*30 + "DEBUG LOG" + "="*30)
                
                input_tokens = ""
                for i, token in enumerate(data["input_ids"][0].tolist()):
                    input_tokens += f"<[{i}]{tokenizer.decode([token])}>"
                print(input_tokens)

                target_tokens = ""
                for i, token in enumerate(data["labels"][0].tolist()):
                    target_tokens += f"<[{i}]{tokenizer.decode([token])}>"
                print(target_tokens)

                print("top_3 tokens:")
                for i, distribution in enumerate(target_head[0]):
                    top_3 = distribution.argsort(descending=True)[:3]
                    print(f"<[{i}-1]{tokenizer.decode([top_3[0]])}>")
                    print(f"<[{i}-2]{tokenizer.decode([top_3[1]])}>")
                    print(f"<[{i}-3]{tokenizer.decode([top_3[2]])}>")

                print(f"attn_mask len and sum: {data['attention_mask'].shape}, {data['attention_mask'].sum()}")
                print(f"loss_mask len and index: {data['loss_mask'].shape}, {torch.nonzero(data['loss_mask'][0] == 1, as_tuple=True)[0]}")

            # --- Efficient top-K counting over jacobi logits (OOM-safe) ---
            with torch.no_grad():
                K = 3
                logits = output['jacobi_logits']                 # [N, V]
                dev = logits.device
                N, V = logits.shape
                T = num_jacobi_tokens                            # J

                # Ensure counts matches the model head vocab size & device
                if counts.device != dev or counts.size(0) != T or counts.size(1) != V:
                    counts = torch.zeros((T, V), dtype=torch.int32, device=dev)

                # Use a smaller working dtype just for topk’s temporary buffers if helpful
                work_dtype = torch.float16 if logits.dtype == torch.float32 else logits.dtype

                # Choose a chunk size based on free memory (fallback to a safe constant)
                try:
                    free_mem, _ = torch.cuda.mem_get_info(dev)
                    bytes_per_row = V * (2 if work_dtype == torch.float16 else 4)
                    rows_per_chunk = max(64, min(N, int(0.20 * free_mem // bytes_per_row)))
                except Exception:
                    rows_per_chunk = 512

                row_ids = torch.arange(N, device=dev)

                for s in range(0, N, rows_per_chunk):
                    e = min(N, s + rows_per_chunk)
                    rids = row_ids[s:e]                 # [R]
                    token_ids = (rids % T)              # [R] → which Jacobi token each row belongs to

                    # top-K indices only (no need for values; no full sort)
                    topk_idx = torch.topk(
                        logits[s:e].to(work_dtype), k=K, dim=-1, largest=True, sorted=False
                    ).indices                           # [R, K]

                    # In-place sparse accumulation: counts[token_ids, topk_idx] += 1
                    counts.index_put_(
                        (token_ids.repeat_interleave(K), topk_idx.reshape(-1)),
                        torch.ones(topk_idx.numel(), dtype=counts.dtype, device=dev),
                        accumulate=True
                    )
                    
            # visualize
            # if batch_idx % 1000 == 0:
            #     target_ids = data['labels'].view(-1, num_jacobi_tokens)
            #     target_jacobi_logits = target_head.view(-1, num_jacobi_tokens, target_head.shape[-1])
            #     output_jacobi_logits = output['jacobi_logits'].view(-1, num_jacobi_tokens, output['jacobi_logits'].shape[-1])
                
            #     sample_counts = min(target_jacobi_logits.shape[0], 3)
            #     group_nums = torch.randint(low=0, high=target_jacobi_logits.shape[0], size=(sample_counts,)).tolist()
            #     for group_num in group_nums:
            #         print(f"top_3 tokens of group {group_num}:")
            #         for i, distribution in enumerate(output_jacobi_logits[group_num][:num_jacobi_tokens]):
            #             top_3 = distribution.argsort(descending=True)[:3]
            #             target_decode = tokenizer.decode(target_ids[group_num][i])
            #             target_decode = target_decode.replace('\n', '\\n') if '\n' in target_decode else target_decode
            #             report = f"<[{i}-Target]{target_decode}>, "
            #             for idx, token in enumerate(top_3):
            #                 decode = tokenizer.decode([token])
            #                 decode = decode.replace('\n', '\\n') if '\n' in decode else decode
            #                 report += f"<[{i}-{idx+1}]{decode}>, "
            #             print(report)
            #     top_5 = counts.argsort(dim=-1, descending=True)[:, :5]
            #     report = f"[batch {batch_idx}] most freq predict tokens:\n"
            #     for seq_id, seq_data in enumerate(top_5):
            #         report += f"[token {seq_id}] "
            #         for i, token in enumerate(seq_data):
            #             decode = tokenizer.decode([token])
            #             decode = decode.replace('\n', '\\n') if '\n' in decode else decode
            #             report += f"<top {i+1}: {decode}({counts[seq_id, token]} times)>, "
            #         report = report[:-2] + "\n"
            #     print(report)
            loss = criterion.compute_loss(data["hidden_state_target"], target_head, output['jacobi_hidden_states'], output['jacobi_logits'], output['jacobi_all_hidden_states'], model.jacobi_weight)
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), tr_cfg.train.grad_clip)
            optimizer.step()
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(output['jacobi_logits'], -1)
            _, target = torch.max(target_head, -1)
            ct = predicted.shape[0] // num_jacobi_tokens
            cc = (predicted == target) 

            topkacc = top_accuracy(output['jacobi_logits'], target, num_jacobi_tokens, (1, 2, 3))
            for i, cor_seq in enumerate(topkacc):
                cor_seq = cor_seq.view(-1, num_jacobi_tokens)
                cor_seq = cor_seq.sum(0)
                for seq_id in range(len(cor_seq)):
                    top_3acc[i][seq_id] += topkacc[i][seq_id]
            total += ct

        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": criterion.vloss,
                       "train/ploss": criterion.ploss, "train/rloss": criterion.adp_reg, "train/hloss": criterion.jcb_reg, "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                for seq in range(len(i)):
                    logdict[f'train/top_{id + 1}_token_{seq}_acc'] = top_3acc[id][seq].item() / total
            wandb.log(logdict)

        del target_head
        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss += loss.item()
        num_batches += 1

        if debug_mode and batch_idx % 500 == 0:
            print(torch.cuda.memory_summary(device='cuda', abbreviated=True), flush=True)
    
    if continuous_loss_nan >= 3:
        accelerator.save_state(output_dir=f"{tr_cfg.meta.cpdir}/{tr_cfg.meta.name}/state_{epoch}_abnormal")
        break

    epoch_counts.append(counts)
    epoch_loss /= num_batches
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        wandb.log({"train/epochloss": epoch_loss})

    # evaluation
    if (epoch ) % tr_cfg.train.save_freq == 0:
        top_3acc = [[0 for _ in range(tr_cfg.model.num_jacobi_tokens)] for _ in range(3)]
        correct = [0 for _ in range(tr_cfg.model.num_jacobi_tokens)]
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader)):
                output = model(
                    input_ids=data["input_ids"], 
                    attention_mask=data["attention_mask"],
                    loss_mask=data["loss_mask"],
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True
                    )
                
                tgt = data["hidden_state_target"].to(model.lm_head.weight.dtype)
                target_head = model.lm_head(tgt).detach()

                loss = criterion.compute_loss(data["hidden_state_target"], target_head, output['jacobi_hidden_states'], output['jacobi_logits'], output['jacobi_all_hidden_states'], model.jacobi_weight)
            
                _, predicted = torch.max(output['jacobi_logits'], -1)
                _, target = torch.max(target_head, -1)
                ct = predicted.shape[0] // num_jacobi_tokens
                cc = (predicted == target) 

                topkacc = top_accuracy(output['jacobi_logits'], target, num_jacobi_tokens, (1, 2, 3))
                for i, cor_seq in enumerate(topkacc):
                    cor_seq = cor_seq.view(-1, num_jacobi_tokens)
                    cor_seq = cor_seq.sum(0)
                    for seq_id in range(len(cor_seq)):
                        top_3acc[i][seq_id] += topkacc[i][seq_id]
                total += ct

            if accelerator.is_main_process and ct != 0:
                logdict = {"test/vloss": criterion.vloss, "test/ploss": criterion.ploss, "test/rloss": criterion.adp_reg, "test/hloss": criterion.jcb_reg, "test/loss": loss.item(), "test/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                for seq in range(len(i)):
                    logdict[f'test/top_{id + 1}_token_{seq}_acc'] = top_3acc[id][seq].item() / total
            wandb.log(logdict)

        epoch_loss += loss.item()
        num_batches += 1

        del target_head, loss
        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            wandb.log({"test/epochloss": epoch_loss})
            output_dir=f"{tr_cfg.meta.cpdir}/{tr_cfg.meta.name}/state_{epoch}"
            save_trainable_weights(model, f"{output_dir}/model_weight.pt")
            # accelerator.save_state(output_dir=output_dir)

torch.save(torch.stack(epoch_counts, dim=0), f"{tr_cfg.meta.cpdir}/{tr_cfg.meta.name}epoch_counts.pt")
