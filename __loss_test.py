import torch 
from torch import nn

def compute_loss(target_hidden, target_head, out_hidden, out_head, criterion, loss_mask):
    target_p = nn.Softmax(dim=2)(target_head)  # [bs, seq, vocab_size]
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    vloss = criterion(out_hidden, target_hidden)  # [bs, seq, hidden_dim]
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head


def compute_loss_ours(hidden_state_target, target_logits, jacobi_hidden_states, jacobi_logits, criterion, jacobi_token_nums, discount=1):
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

bs = 2
seq = 20
hidden_dim = 16 
vocab_size = 2048
jacobi_nums = 5 

target_hidden = torch.randn((bs, seq, hidden_dim))
target_head = torch.randn((bs, seq, vocab_size))
out_hidden = torch.randn((bs, seq, hidden_dim))
out_head = torch.randn((bs, seq, vocab_size))

loss_mask = torch.randint(0, 2, (bs, seq, 1))

# loss_mask = torch.full_like(loss_mask, 1)

criterion = nn.SmoothL1Loss(reduction="none")
vloss, ploss, out_head = compute_loss(target_hidden, target_head, out_hidden, out_head, criterion, loss_mask)
print(vloss, ploss)


vloss, ploss = compute_loss_ours(target_hidden.reshape(-1, hidden_dim), target_head.reshape(-1, vocab_size),
                  out_hidden.reshape(-1, hidden_dim), out_head.reshape(-1, vocab_size), criterion, jacobi_nums)
print(vloss, ploss)