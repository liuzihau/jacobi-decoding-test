import os
import gc
import json
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from transformers import AutoConfig, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open

from data_processing import CustomDataset, DataCollatorWithPadding, list_files
from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

def load_jacobi_weight(model, cpdir):
    with safe_open(cpdir, framework="pt") as f:
        keys = f.keys()        
        for name, param in model.named_parameters():
            all_set = True
            if "model." in name:
                continue
            if name in keys:
                tensor_slice = f.get_slice(name)
                tensor = tensor_slice[:].clone().detach()
                if tensor.shape == param.shape:
                    param.data.copy_(tensor) 
                else:
                    print(f"Shape mismatch for {name}: Model shape {param.shape}, File shape {tensor.shape}")
            else:
                all_set = False
                print(f"Key {name} not found in SafeTensor file.")
        if all_set:
            print("All parameters has been loaded.")


def top_accuracy(output, target, jacobi_token_nums, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.view(-1, jacobi_token_nums, output.shape[-1])
    target = target.view(-1, jacobi_token_nums)
    with torch.no_grad():
        maxk = max(topk)
        group_size = target.size(0)
        jacobi_seq_size = target.size(1)
        _, pred = output.topk(maxk, -1, True, True)  # bs, seq, topk ex [4, 10, 3]
        # pred = pred.t()
        correct = pred.eq(target.view(group_size, jacobi_seq_size, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:, :, :k].float().sum(0).sum(-1)
            res.append(correct_k)
        return res
    
def compute_loss(hidden_state_target, target_logits, jacobi_hidden_states, jacobi_logits, criterion, jacobi_token_nums, discount=1):
    # cross entropy -> sample distribution difference
    target_p = nn.Softmax(dim=-1)(target_logits)
    out_logp = nn.LogSoftmax(dim=-1)(jacobi_logits)
    plogp = target_p * out_logp
    plogp = plogp.view(-1, jacobi_token_nums, plogp.shape[-1])
    ploss = -torch.sum(plogp) / (plogp.shape[0] * plogp.shape[1] + 1e-5)  # Normalize by batch and sequence

    # regression -> hidden states difference
    vloss = criterion(jacobi_hidden_states, hidden_state_target)
    vloss = vloss.view(-1, jacobi_token_nums, vloss.shape[-1])
    vloss = torch.mean(vloss, dim=-1)  
    vloss = torch.sum(vloss) / (vloss.shape[0] * vloss.shape[1] + 1e-5)

    return vloss, ploss

def cllm_loss():
    ### compute AutoRegression loss ###
    # use labels to avoid pattern collapse
    if self.use_gt_labels:
        labels = inputs['labels_ids']
    else:
        labels = inputs['teacher_output_ids']
    # TODO: check if it's right when batch size > 1
    labels = torch.tensor(labels).to(model.device)
    attention_mask = torch.full_like(labels, 1).to(model.device)
    label_student_model_output = model(labels, attention_mask)

    attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(model.device)
    attention_mask = jacobian_trajectory[-1] != self.tokenizer.pad_token_id
    logits_last =  self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask)

    label_smoother = LabelSmoother(epsilon=0.1, ignore_index= -100)
    loss_ar = label_smoother(label_student_model_output, labels, shift_labels=True)
    loss_ar*=10
    if self.args.qlora:
        loss_ar.requires_grad = True
    print(f'loss ar: {loss_ar} computed! performing backward pass...')
    with self.accelerator.accumulate(model):
        self.accelerator.backward(loss_ar)

    ### compute Consistency loss (global) ###
    # random select one point from trajectory
    i = random.choice(range(len(jacobian_trajectory))[:-1])

    attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
    attention_mask = jacobian_trajectory[i] != self.tokenizer.pad_token_id
    logits_i = self.get_logits(model, jacobian_trajectory[i].clone().detach(), attention_mask)

    output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id
    # We do not calculate the cross entrophy of same logits to alleviate misleading gradients
    for j in range(bsz):
        end_of_mask_position = torch.where(jacobian_trajectory[i][j, 1:] != jacobian_trajectory[-1][j, 1:])[0]
        if len(end_of_mask_position)==0:
            output_mask[j, :] = True
        else:
            output_mask[j, :end_of_mask_position[0]] = True
    
    loss_global = self.soft_cross_entropy(
                logits_i[..., :-1, :].float(), # logits generated by the last token is dropped
                logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
                output_mask.to(logits_i.device)
    )
    if self.args.qlora:
        loss_global.requires_grad = True
    print(f'loss global {loss_global} computed! performing backward pass...')
    with self.accelerator.accumulate(model):
        self.accelerator.backward(loss_global)
    
    if self.args.local_rank == 0:
        wandb.log({"ar loss": loss_ar})
        wandb.log({"consistency loss": loss_global})

    # sync processes
    torch.distributed.barrier()
    # total loss = ar_loss + consistency_global_loss
    loss = loss_ar.detach() + loss_global.detach()

    return loss

CONFIG_PATH = '/content/jacobi-decoding-test/configs/train_config.json'
PROJECT = 'Jacobi-test'
GAMMA = 0.9

with open(CONFIG_PATH, 'r') as f:
    train_config = json.loads(f.read())

with open(f"{train_config['basepath']}/config.json", 'r') as f:
    model_config = json.loads(f.read())

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
set_seed(0)

torch.backends.cuda.matmul.allow_tf32 = True

accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

if accelerator.is_main_process:
    wandb.login(key=train_config["api_key"])
    wandb.init(project=PROJECT, name=train_config["name"], config=train_config)

jacobi_adapter_kwargs = train_config["jacobi_adapter_kwargs"]
tokenizer = Qwen2Tokenizer.from_pretrained(train_config["basepath"], use_fast=False)
model = Qwen2JacobiForCausalLM.from_pretrained(
    pretrained_model_name_or_path=train_config["basepath"],
    jacobi_token_nums=train_config["jacobi_token_nums"],
    mix_sequences=train_config["mix_sequences"],
    proj_freq=train_config["projection_frequency"],
    adapter_type=train_config["adapter_type"],
    shared_adapter=train_config["shared_adapter"],
    shared_jacobi_token=train_config["shared_jacobi_token"],
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

traindataset = CustomDataset(traindatapath, jacobi_tokens=train_config["jacobi_token_nums"], use_multi_token_sets=train_config["use_multi_token_sets"], pad_id=train_config['pad_token_id'], vocab_size=model_config['vocab_size'])
testdataset = CustomDataset(testdatapath, jacobi_tokens=train_config["jacobi_token_nums"], use_multi_token_sets=train_config["use_multi_token_sets"], pad_id=train_config['pad_token_id'], vocab_size=model_config['vocab_size'])

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

epoch_counts = []
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

            vloss, ploss = compute_loss(data["hidden_state_target"], target_head, output['jacobi_hidden_states'], output['jacobi_logits'], criterion, jacobi_token_nums)#, loss_mask)
            if torch.isnan(vloss).any() or torch.isnan(ploss).any():
                print(f"loss contain nan : {data['filename']}")
                continue
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
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
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
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

                
                vloss, ploss = compute_loss(data["hidden_state_target"], target_head, output['jacobi_hidden_states'], output['jacobi_logits'], criterion, jacobi_token_nums)#, loss_mask)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            
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
                logdict = {"test/vloss": vloss.item(), "test/ploss": ploss.item(), "test/loss": loss.item(), "test/acc": cc / ct}
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