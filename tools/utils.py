import time
import torch
from safetensors import safe_open

PERFORMANCE_CHECK = False

class Timer:
    def __init__(self):
        self.report = {}

    def record_time(self, key, func, **kwargs):
        s = time.time()
        res = func(**kwargs)
        delta = time.time() - s
        if key in self.report:
            self.report[key]["time"] += delta
            self.report[key]["count"] += 1
        else:
            self.report[key] = {}
            self.report[key]["time"] = delta
            self.report[key]["count"] = 1        
        return res

timer = Timer()


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
def output_abnormal_message(target_p, output_logp, jacobi_hidden_states, target_hidden_state, pshape0, pshape1, vshape0, vshape1):
    report = ""
    report += f"[target_p contain inf]: {torch.isinf(target_p).any()}\n"
    report += f"[target_hidden contain inf]: {torch.isinf(target_hidden_state).any()}\n"                
    report += f"[output_logp contain inf]: {torch.isinf(output_logp).any()}\n"
    report += f"[output_hidden contain inf]: {torch.isinf(jacobi_hidden_states).any()}\n"
    report += f"[target_p contain nan]: {torch.isnan(target_p).any()}\n"
    report += f"[target_hidden contain nan]: {torch.isnan(target_hidden_state).any()}\n"                
    report += f"[output_logp contain nan]: {torch.isnan(output_logp).any()}\n"
    report += f"[output_hidden contain nan]: {torch.isnan(jacobi_hidden_states).any()}\n"
    report += f"[pshape]: {pshape0}, {pshape1}\n[vshape]: {vshape0}, {vshape1}"
    return report

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