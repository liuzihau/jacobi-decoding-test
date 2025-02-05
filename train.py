import os
import gc
import json
import pickle
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from transformers import AutoConfig, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from tools.data_processing import CustomDataset, DataCollatorWithPadding, list_files
from tools.utils import top_accuracy
from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    
def compute_loss(hidden_state_target, target_logits, jacobi_hidden_states, jacobi_logits, all_layers_outputs, jacobi_weight, criterion, jacobi_token_nums, discount=1):
    target_logits = torch.clamp(target_logits, min=-1e2, max=1e2)
    jacobi_logits = torch.clamp(jacobi_logits, min=-1e2, max=1e2)
    jacobi_hidden_states = torch.clamp(jacobi_hidden_states, min=-1e3, max=1e3)
    hidden_state_target = torch.clamp(hidden_state_target, min=-1e3, max=1e3)

    # cross entropy -> sample distribution difference
    target_p = nn.Softmax(dim=-1)(target_logits)
    out_logp = nn.LogSoftmax(dim=-1)(jacobi_logits)
    plogp = target_p * out_logp
    plogp = plogp.view(-1, jacobi_token_nums, plogp.shape[-1])
    ploss = -torch.sum(plogp) / (plogp.shape[0] * plogp.shape[1] + 1e-5)  # Normalize by batch and sequence

    # regression -> hidden states difference
    vloss_full = criterion(jacobi_hidden_states, hidden_state_target)
    vloss_full = vloss_full.view(-1, jacobi_token_nums, vloss_full.shape[-1])
    vloss_full = torch.mean(vloss_full, dim=-1)  
    vloss = torch.sum(vloss_full) / (vloss_full.shape[0] * vloss_full.shape[1] + 1e-5)

    # Regularization term for Jacobi weight
    reg_term_jacobi = torch.sum(torch.abs(jacobi_weight.float())) / (jacobi_weight.shape[-1] + 1e-5)   # L1 regularization
    reg_term_hidden = 0
    for tensor in all_layers_outputs:
        reg_term_hidden += torch.sum(tensor.float() ** 2) * (1/2) / (tensor.shape[0] * tensor.shape[1] + 1e-5)

    return vloss, ploss, reg_term_jacobi, reg_term_hidden


# CONFIG_PATH = '/content/jacobi-decoding-test/configs/train_config_colab.json'
CONFIG_PATH = './configs/train_config_local.json'
PROJECT = 'Jacobi-test'
GAMMA = 0.9

with open(CONFIG_PATH, 'r') as f:
    train_config = json.loads(f.read())

with open(f"{train_config['basepath']}/config.json", 'r') as f:
    model_config = json.loads(f.read())

set_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

accelerator = Accelerator(mixed_precision=train_config['mixed_precision'], gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

if accelerator.is_main_process:
    wandb.login(key=train_config["api_key"])
    wandb.init(project=PROJECT, name=train_config["name"], config=train_config)

jacobi_adapter_kwargs = train_config["jacobi_adapter_kwargs"]
tokenizer = Qwen2Tokenizer.from_pretrained(train_config["basepath"], use_fast=False)
model = Qwen2JacobiForCausalLM.from_pretrained(
    pretrained_model_name_or_path=train_config["basepath"],
    jacobi_token_nums=train_config["jacobi_token_nums"],
    mix_sequences=train_config["mix_sequences"],
    token_sets_inline=train_config["token_sets_inline"],
    proj_freq=train_config["projection_frequency"],
    adapter_type=train_config["adapter_type"],
    shared_adapter=train_config["shared_adapter"],
    shared_jacobi_token=train_config["shared_jacobi_token"],
    layer_norm=train_config["layer_norm"],
    jacobi_adapter_kwargs=jacobi_adapter_kwargs,
    torch_dtype="auto",
    device_map="auto"
)


model = model.to('cuda')
for param in model.model.parameters():
    param.requires_grad = False

initialise_method = train_config["initialise_method"] if "initialise_method" in train_config else 'kaiming'
for name, param in model.named_parameters():
    if param.requires_grad:   
        model.init_trainable_weights(name, param, initialise_method)

# data part
datapath = list_files(train_config["datapath"])
traindatapath = datapath[:int(len(datapath) * train_config["train_data_portion"])]
testdatapath = datapath[int(len(datapath) * train_config["test_data_portion"]):]

shuffle_data = True
if train_config["debug_mode"]:
    shuffle_data = False
    traindatapath = datapath[:4]
    testdatapath = datapath[int(len(datapath) * 0.1):int(len(datapath) * 0.15)]
print(f"train data: {len(traindatapath)}")
print(f"test data: {len(testdatapath)}")

traindataset = CustomDataset(traindatapath, jacobi_tokens=train_config["jacobi_token_nums"], use_multi_token_sets=train_config["use_multi_token_sets"], token_sets_inline=train_config["token_sets_inline"], pad_id=train_config['pad_token_id'], vocab_size=model_config['vocab_size'])
testdataset = CustomDataset(testdatapath, jacobi_tokens=train_config["jacobi_token_nums"], use_multi_token_sets=train_config["use_multi_token_sets"], token_sets_inline=train_config["token_sets_inline"], pad_id=train_config['pad_token_id'], vocab_size=model_config['vocab_size'])

train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=shuffle_data,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(train_config["cpdir"]):
        os.makedirs(train_config["cpdir"])

criterion = nn.SmoothL1Loss(reduction="none")  
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]
jacobi_token_nums = train_config["jacobi_token_nums"]
debug_mode = train_config["debug_mode"]
if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

if train_config["statepath"] is not None:
    # Load accelerator state
    print("Loading model, optimizer, and scheduler states in {train_config['statepath']}")
    accelerator.load_state(train_config["statepath"])

    # Restore random states
    # random_state_file = os.path.join(train_config["statepath"], "random_states_0.pkl")
    # with open(random_state_file, "rb") as f:
    #     random_states = pickle.load(f)

    # torch.random.set_rng_state(random_states["torch"])
    # torch.cuda.random.set_rng_state(random_states["cuda"])

    print("State restored successfully!")

continuous_loss_nan = 0
epoch_counts = []
previous_data = []
for epoch in range(num_epochs + 1):
    top_3acc = [[0 for _ in range(jacobi_token_nums)] for _ in range(3)]
    correct = [0 for _ in range(jacobi_token_nums)]
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    counts = torch.zeros((jacobi_token_nums, model_config['vocab_size']), dtype=torch.int32).to(model.device)
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
                print(f"outputs contain nan : {data['filename']}")
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
            previous_data = data['filename']

            with torch.no_grad():
                target_head = model.lm_head(data["hidden_state_target"])  # [sum(jacobi_token * sets), logits]
                target_head = target_head.detach()
            
            if debug_mode:
                print("="*30 + "DEBUG LOG" + "="*30)
                
                input_tokens = ""
                for i, token in enumerate(data["input_ids"][0].tolist()):
                    input_tokens += f"<[{i}]{tokenizer.decode([token])}>"
                print(input_tokens)

                target_tokens = ""
                for i, token in enumerate(data["target"][0].tolist()):
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

            # record total generated top_k tokens
            K = 3
            top_k = output['jacobi_logits'].argsort(dim=-1, descending=True)[:, :K]  # [jacobi_tokens * sets, top_k]
            seq, k = top_k.shape
            sets = seq // jacobi_token_nums
            top_k = top_k.view(sets, jacobi_token_nums, k)
            top_k = top_k.permute(1, 0, 2).reshape(jacobi_token_nums, -1)
            for seq_idx, ith_data in enumerate(top_k):
                c = torch.bincount(ith_data)
                ids = torch.nonzero(c, as_tuple=True)[0]
                counts[seq_idx, ids] += c[ids]

            if batch_idx % 1000 == 0:
                target_ids = data['target'].view(-1, jacobi_token_nums)
                target_jacobi_logits = target_head.view(-1, jacobi_token_nums, target_head.shape[-1])
                output_jacobi_logits = output['jacobi_logits'].view(-1, jacobi_token_nums, output['jacobi_logits'].shape[-1])
                
                sample_counts = min(target_jacobi_logits.shape[0], 3)
                group_nums = torch.randint(low=0, high=target_jacobi_logits.shape[0], size=(sample_counts,)).tolist()
                for group_num in group_nums:
                    print(f"top_3 tokens of group {group_num}:")
                    for i, distribution in enumerate(output_jacobi_logits[group_num][:jacobi_token_nums]):
                        top_3 = distribution.argsort(descending=True)[:3]
                        target_decode = tokenizer.decode(target_ids[group_num][i])
                        target_decode = target_decode.replace('\n', '\\n') if '\n' in target_decode else target_decode
                        report = f"<[{i}-Target]{target_decode}>, "
                        for idx, token in enumerate(top_3):
                            decode = tokenizer.decode([token])
                            decode = decode.replace('\n', '\\n') if '\n' in decode else decode
                            report += f"<[{i}-{idx+1}]{decode}>, "
                        print(report)
                top_5 = counts.argsort(dim=-1, descending=True)[:, :5]
                report = f"[batch {batch_idx}] most freq predict tokens:\n"
                for seq_id, seq_data in enumerate(top_5):
                    report += f"[token {seq_id}] "
                    for i, token in enumerate(seq_data):
                        decode = tokenizer.decode([token])
                        decode = decode.replace('\n', '\\n') if '\n' in decode else decode
                        report += f"<top {i+1}: {decode}({counts[seq_id, token]} times)>, "
                    report = report[:-2] + "\n"
                print(report)

            
            vloss, ploss, reg_term_jacobi, reg_term_hidden = compute_loss(data["hidden_state_target"], target_head, output['jacobi_hidden_states'], output['jacobi_logits'], output['jacobi_all_hidden_states'], model.jacobi_weight, criterion, jacobi_token_nums)
            
            # if torch.isnan(vloss).any() or torch.isnan(ploss).any() or torch.isinf(vloss).any() or torch.isinf(ploss).any():
            #     continuous_loss_nan += 1
            #     print(f"loss contain nan : {data['filename']}")
            #     print(f"previous data : {previous_data}")
            #     report = output_abnormal_message(target_p, output_logp, output['jacobi_hidden_states'], data["hidden_state_target"], pshape0, pshape1, vshape0, vshape1)
            #     print(report)
            #     del ploss, vloss, target_head
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     optimizer.zero_grad()
            #     if continuous_loss_nan >= 3:
            #         break
            #     continue

            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + train_config["r_w"] * reg_term_jacobi + train_config["h_w"] * reg_term_hidden
            # loss.backward()
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(output['jacobi_logits'], -1)
            _, target = torch.max(target_head, -1)
            ct = predicted.shape[0] // jacobi_token_nums
            cc = (predicted == target) 

            topkacc = top_accuracy(output['jacobi_logits'], target, jacobi_token_nums, (1, 2, 3))
            for i, cor_seq in enumerate(topkacc):
                cor_seq = cor_seq.view(-1, jacobi_token_nums)
                cor_seq = cor_seq.sum(0)
                for seq_id in range(len(cor_seq)):
                    top_3acc[i][seq_id] += topkacc[i][seq_id]
            total += ct

        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/rloss": reg_term_jacobi.item(), "train/hloss": reg_term_hidden.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                for seq in range(len(i)):
                    logdict[f'train/top_{id + 1}_token_{seq}_acc'] = top_3acc[id][seq].item() / total
            wandb.log(logdict)

        del ploss, vloss, target_head
        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss += loss.item()
        num_batches += 1

        if debug_mode and batch_idx % 500 == 0:
            print(torch.cuda.memory_summary(device='cuda', abbreviated=True), flush=True)
    
    if continuous_loss_nan >= 3:
        accelerator.save_state(output_dir=f"{train_config['cpdir']}/{train_config['name']}/state_{epoch}_abnormal")
        break
    epoch_counts.append(counts)
    epoch_loss /= num_batches
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        wandb.log({"train/epochloss": epoch_loss})

    # evaluation
    if (epoch ) % train_config["save_freq"] == 0:
        top_3acc = [[0 for _ in range(train_config["jacobi_token_nums"])] for _ in range(3)]
        correct = [0 for _ in range(train_config["jacobi_token_nums"])]
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader)):
                output = model(input_ids=data["input_ids"], 
                            attention_mask=data["attention_mask"],
                            loss_mask=data["loss_mask"],
                            use_cache=False,
                            output_hidden_states=True,
                            return_dict=True)
                
                target_head = model.lm_head(data["hidden_state_target"])
                target_head = target_head.detach()

                
                vloss, ploss, reg_term_jacobi, reg_term_hidden = compute_loss(data["hidden_state_target"], target_head, output['jacobi_hidden_states'], output['jacobi_logits'], output['jacobi_all_hidden_states'], model.jacobi_weight, criterion, jacobi_token_nums)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss + train_config["r_w"] * reg_term_jacobi + train_config["h_w"] * reg_term_hidden
            
                _, predicted = torch.max(output['jacobi_logits'], -1)
                _, target = torch.max(target_head, -1)
                ct = predicted.shape[0] // jacobi_token_nums
                cc = (predicted == target) 

                topkacc = top_accuracy(output['jacobi_logits'], target, jacobi_token_nums, (1, 2, 3))
                for i, cor_seq in enumerate(topkacc):
                    cor_seq = cor_seq.view(-1, jacobi_token_nums)
                    cor_seq = cor_seq.sum(0)
                    for seq_id in range(len(cor_seq)):
                        top_3acc[i][seq_id] += topkacc[i][seq_id]
                total += ct

            if accelerator.is_main_process and ct != 0:
                logdict = {"test/vloss": vloss.item(), "test/ploss": ploss.item(), "test/rloss": reg_term_jacobi.item(), "test/hloss": reg_term_hidden.item(), "test/loss": loss.item(), "test/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                for seq in range(len(i)):
                    logdict[f'test/top_{id + 1}_token_{seq}_acc'] = top_3acc[id][seq].item() / total
            wandb.log(logdict)

        del ploss, vloss, target_head
        gc.collect()
        torch.cuda.empty_cache()

        epoch_loss += loss.item()
        num_batches += 1

        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            wandb.log({"test/epochloss": epoch_loss})

            accelerator.save_state(output_dir=f"{train_config['cpdir']}/{train_config['name']}/state_{epoch}")

torch.save(torch.stack(epoch_counts, dim=0), f"{train_config['cpdir']}/{train_config['name']}epoch_counts.pt")