import os
import gc
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from transformers import AutoConfig, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from data_processing import CustomDataset, DataCollatorWithPadding, list_files
from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
    
def compute_loss(hidden_state_target, target_logits, jacobi_hidden_states, jacobi_logits, criterion, loss_mask):
    # cross entropy -> sample distribution difference
    target_p = nn.LogSoftmax(dim=2)(target_logits)
    out_logp = nn.LogSoftmax(dim=2)(jacobi_logits)
    plogp = target_p * out_logp
    # ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    ploss = -torch.sum(torch.sum(plogp, 2)) / (plogp.shape[1] + 1e-5)

    # regression -> hidden states difference
    vloss = criterion(jacobi_hidden_states, hidden_state_target)
    # vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    vloss = torch.sum(torch.mean(vloss, 2)) / (vloss.shape[1] + 1e-5)
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
set_seed(0)

torch.backends.cuda.matmul.allow_tf32 = True

accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

if accelerator.is_main_process:
    import wandb
    wandb.login(key=train_config["api_key"])
    wandb.init(project=PROJECT, name=train_config["name"], config=train_config)

baseconfig = AutoConfig.from_pretrained(train_config["basepath"])

# model = Qwen2ForCausalLM.from_pretrained(
model = Qwen2JacobiForCausalLM.from_pretrained(
    train_config["basepath"],
    train_config["jacobi_token_nums"],
    torch_dtype="auto",
    device_map="auto"
)

# freeze target model's parameter
for param in model.model.parameters():
    param.requires_grad = False

datapath = list_files(train_config["datapath"])

# data part
traindatapath = datapath[:int(len(datapath) * 0.1)]
testdatapath = datapath[int(len(datapath) * 0.1):int(len(datapath) * 0.15)]

traindataset = CustomDataset(traindatapath, jacobi_tokens=train_config["jacobi_token_nums"])
testdataset = CustomDataset(testdatapath, jacobi_tokens=train_config["jacobi_token_nums"])
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(train_config["cpdir"]):
        os.makedirs(train_config["cpdir"])

# config = EConfig.from_pretrained(train_config["config_path"])
# model = Model(config, load_emb=True, path=train_config["basepath"])

criterion = nn.SmoothL1Loss(reduction="none")  
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

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
# accelerator.load_state("checkpoints/state_5")
for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            output = model(input_ids=data["input_ids"], 
                            attention_mask=data["attention_mask"],
                            loss_mask=data["loss_mask"],
                            output_hidden_states=True,
                            return_dict=True)
            with torch.no_grad():
                target_head = model.lm_head(data["hidden_state_target"])
                target_head = target_head.detach()
            # loss_mask = data["loss_mask"][:, :, None]
            vloss, ploss = compute_loss(data["hidden_state_target"], target_head, output['jacobi_hidden_states'], output['jacobi_logits'], criterion)#, loss_mask)
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            # loss.backward()
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(output['jacobi_logits'], 2)
            _, target = torch.max(target_head, 2)
            ct = predicted.shape[1]
            cc = (predicted == target) 
            # out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            # target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(output['jacobi_logits'], target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)
            # for id,i in enumerate(top_3acc):
            #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

        del ploss, vloss, out_head, target_head, target_p
        gc.collect()
        torch.cuda.empty_cache()
        epoch_loss += loss.item()
        num_batches += 1

        if train_config["debug_mode"] and batch_idx % 50 == 0:
            print(torch.cuda.memory_summary(device='cuda', abbreviated=True), flush=True)


    # correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    # correct, total = accelerator.gather_for_metrics((correct, total))
    # correct, total = correct.sum().item(), total.sum().item()
    # epoch_loss /= num_batches
    # top_3acc = accelerator.gather_for_metrics(top_3acc)
    # if accelerator.is_local_main_process:
    #     for id, i in enumerate(top_3acc):
    #         wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    # if accelerator.is_local_main_process:
    #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
    #     print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
    #     wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    # if (epoch + 1) % train_config["save_freq"] == 0:
    #     top_3acc = [0 for _ in range(3)]
    #     correct = 0
    #     total = 0
    #     epoch_loss = 0
    #     num_batches = 0
    #     model.eval()

    #     k_acc = [[] for i in range(5)]
    #     for batch_idx, data in enumerate(tqdm(test_loader)):
    #         with torch.no_grad():
    #             if batch_idx < 5:
    #                 acces = getkacc(model, data, head, max_length=5)
    #                 for i in range(len(acces)):
    #                     k_acc[i].append(acces[i])
    #             predict = model(data["hidden_states"], input_ids=data["input_ids"],
    #                             attention_mask=data["attention_mask"])
    #             target_head = head(data["target"])
    #             target_p = nn.Softmax(dim=2)(target_head)
    #             target_p = target_p.detach()
    #             loss_mask = data["loss_mask"][:, :, None]
    #             vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, criterion, loss_mask)
    #             loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
    #             _, predicted = torch.max(out_head, 2)
    #             _, target = torch.max(target_head, 2)
    #             ct = loss_mask.sum().item()
    #             cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
    #             out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
    #             target = target.view(-1)[loss_mask.view(-1) == 1]
    #             topkacc = top_accuracy(out_head, target, (1, 2, 3))
    #             for top_i in range(len(topkacc)):
    #                 top_3acc[top_i] += topkacc[top_i]
    #             total += ct
    #             correct += cc

    #             del ploss, vloss, out_head, target_head, target_p
    #             gc.collect()
    #             torch.cuda.empty_cache()
                
    #         epoch_loss += loss.item()
    #         num_batches += 1

    #     mean_acces = []
    #     for id, i in enumerate(k_acc):
    #         mean_acc = np.array(i).mean()
    #         mean_acc = torch.tensor(mean_acc).cuda()
    #         mean_acces.append(mean_acc)

    #     mean_acces = accelerator.gather_for_metrics(mean_acces)
    #     if accelerator.is_local_main_process:
    #         for id, i in enumerate(mean_acces):
    #             mean_acc = i.mean().item()
    #             wandb.log({f"test/{id}_acc": mean_acc})

    #     correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    #     correct, total = accelerator.gather_for_metrics((correct, total))
    #     correct, total = correct.sum().item(), total.sum().item()
    #     top_3acc = accelerator.gather_for_metrics(top_3acc)
    #     if accelerator.is_local_main_process:
    #         for id, i in enumerate(top_3acc):
    #             wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
    #     epoch_loss /= num_batches
    #     if accelerator.is_local_main_process:
    #         print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
    #         print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
    #         wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
    #         if not os.path.exists(train_config['cpdir']):
    #             os.mkdir(train_config['cpdir'])
    #         if not os.path.exists(f"{train_config['cpdir']}/{train_config['name']}"):
    #             os.mkdir(f"{train_config['cpdir']}/{train_config['name']}")
    #         accelerator.save_state(output_dir=f"{train_config['cpdir']}/{train_config['name']}/state_{epoch}")