import torch
from torch import nn

"""
Test jacobi token shape
"""
# x = nn.Parameter(torch.randn(896))
# print(x.unsqueeze(0).unsqueeze(0).repeat(1, 10, 1).shape)

"""
test top accuracy function
"""
# output = torch.randn(4, 10, 128)
# target = torch.randint(0, 127, (4, 10))
# _, pred = output.topk(3, 2, True, True)
# print(pred.shape)
# print(target.shape)
# correct = pred.eq(target.view(4, 10, -1).expand_as(pred))
# print(correct.float().sum(0).sum(-1).shape)

# res = []
# for k in [1, 2, 3]:
#     correct_k = correct[:, :, :k].float().sum(0).sum(-1)
#     res.append(correct_k)
# print(res)


"""
Test input data correct or not
"""
# import json
# from tqdm import tqdm
# import torch
# from torch.utils.data import DataLoader
# from models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
# from data_processing import CustomDataset, DataCollatorWithPadding, list_files

# if __name__ == '__main__':
#     CONFIG_PATH = './configs/train_config.json'
#     with open(CONFIG_PATH, 'r') as f:
#         train_config = json.loads(f.read())

#     tokenizer = Qwen2Tokenizer.from_pretrained("./Qwen2.5-0.5B-Instruct", use_fast=False)
#     datapath = list_files("./data_root/ShareGPT_Vicuna_unfiltered_Qwen2.5-0.5B-Instruct")
#     shuffle_data = False
#     traindatapath = datapath[:4]
#     traindataset = CustomDataset(traindatapath, jacobi_tokens=train_config["jacobi_token_nums"])
#     train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=shuffle_data,
#                             collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
#                             pin_memory=True)
#     for batch_idx, data in enumerate(tqdm(train_loader)):
#         input_tokens = ""
#         for i, token in enumerate(data["input_ids"][0].tolist()):
#             input_tokens += f"<[{i}]{tokenizer.decode([token])}>"
#         print(input_tokens)

#         target_tokens = ""
#         for i, token in enumerate(data["target"][0].tolist()):
#             target_tokens += f"<[{i}]{tokenizer.decode([token])}>"
#         print(target_tokens)

#         print(f"attn_mask len and sum: {data['attention_mask'].shape}, {data['attention_mask'].sum()}")
#         print(f"loss_mask len and index: {data['loss_mask'].shape}, {torch.nonzero(data['loss_mask'][0] == 1, as_tuple=True)[0]}")
#         break


# mix_sequence = 1
# hidden_states = torch.randn((2, 24, 4))
# loss_mask = torch.Tensor([
#     [0] * 10 + [1] * 5 + [0] * 9,
#     [0] * 19 + [1] * 5
# ])

# replace_matrix = []
# for i in range(loss_mask.shape[0]):
#     replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
#     all_indices = torch.cat([replace_indices[:mix_sequence] - mix_sequence, replace_indices], dim=-1)
#     replace_matrix.append(hidden_states[i, all_indices])
# replace_matrix = torch.stack(replace_matrix, 0)
# print(hidden_states)
# print(replace_matrix.shape)


# replace_indices = torch.nonzero(loss_mask == 1, as_tuple=True)
# # all_indices = torch.cat([replace_indices[:mix_sequence] - mix_sequence, replace_indices], dim=-1)
# print(replace_indices)


# counts = torch.zeros((10, 20))
# a = torch.randint(0, 60, (180,)).reshape(2, 10, 9)
# print(a)
# b = a.argsort(descending=True, dim=-1)[:, :, :3]
# print(b)
# c = b.permute(1, 0, 2).reshape(10, -1)
# print(c)
# for seq_idx, ith_data in enumerate(c):
#     e = torch.bincount(ith_data)
#     ids = torch.nonzero(e, as_tuple=True)[0]
#     counts[seq_idx, ids] += e[ids]
# print(counts)


# top_5 = counts.argsort(dim=-1, descending=True)[:, :5]
# report = f""
# for seq_id, seq_data in enumerate(top_5):
#     report += f"[token {seq_id}] "
#     for i, token in enumerate(seq_data):
#         report += f"<top {i+1}: {token}({counts[seq_id, token]} times)>, "
#     report = report[:-2] + "\n"
# print(report)

# generated_tokens = torch.ones((12), dtype=torch.int32)
# padding_tensor = torch.ones((generated_tokens.shape[0], 10), dtype=generated_tokens.dtype) * 159292
# merged_tokens = torch.cat([generated_tokens[:, None], padding_tensor], dim=-1).flatten()
# jacobi_indices = torch.nonzero(merged_tokens == 159292, as_tuple=True)
# loss_mask = torch.zeros_like(merged_tokens)
# loss_mask[jacobi_indices] = 1
# loss_mask = loss_mask.tolist()
# print(merged_tokens)
# print(loss_mask)

# tensor = torch.arange(30)
# print(tensor.unfold(0, 4, 1).reshape(-1))

# hidden_states = torch.arange(60).reshape(20, 3)
# print(hidden_states)
# hidden_states = hidden_states.unfold(0, 4, 1)
# hidden_states = torch.permute(hidden_states, (0, 2, 1))
# print(hidden_states)
# print(hidden_states.reshape(-1, hidden_states.shape[-1]))

# Example input tensors
# input_tensor = torch.zeros(40, 4)  # Shape (40, 4)
# index = torch.tensor([5, 6, 7, 9, 10, 11, 16, 17, 18])  # Shape (9,)
# weight = torch.ones(3, 4)  # Shape (3, 4)

# # Reshape index to groups of 3
# index_groups = index.view(-1, 3)  # Shape (num_groups, 3)

# # Create a scatter mask
# scatter_indices = index_groups.flatten()  # Flatten to (54,)
# print(scatter_indices.unsqueeze(-1).expand(-1, 4))
# # Create a scatter source tensor with repeated weights
# scatter_source = weight.repeat(index_groups.shape[0], 1)  # Shape (54, 896)
# print(scatter_source.shape)
# # Scatter the weights into the input tensor
# input_tensor.scatter_(0, scatter_indices.unsqueeze(-1).expand(-1, 4), scatter_source)

# print(input_tensor)


# input_embeds = torch.Tensor([[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
#                              [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]])

# loss_mask = torch.Tensor([[0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
#                           [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]])

# jacobi_tokens = 2
# jacobi_position = torch.arange(jacobi_tokens)
# cache_position = torch.ones_like(input_embeds) * -1

# for batch_idx in range(input_embeds.shape[0]):
#     # handle normal input position
#     inputs_position = torch.arange(loss_mask[batch_idx].shape[0] - loss_mask[batch_idx].sum(-1))
#     replace_indices = torch.nonzero(loss_mask[batch_idx] == 0, as_tuple=True)[0]
#     cache_position[batch_idx, replace_indices] = inputs_position

#     # handle jacobi tokens' position
#     replace_indices = torch.nonzero(cache_position[batch_idx] == -1, as_tuple=True)[0]
#     replace_indices_groups = replace_indices.view(-1, jacobi_tokens)
#     prefix_position = cache_position[batch_idx, (replace_indices_groups[:, 0] - 1)].repeat(jacobi_tokens, 1).transpose(-1, -2)
#     true_jacobi_position = (prefix_position+jacobi_position+1).flatten()
#     cache_position[batch_idx, replace_indices] = true_jacobi_position

# print(cache_position)


input_embeds = torch.Tensor([[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                             [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]])

loss_mask = torch.Tensor([[0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                          [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]])

jacobi_tokens = 2

target_length = input_embeds.shape[1]

device = input_embeds.device
dtype = input_embeds.dtype
min_dtype = torch.finfo(dtype).min

final_mask = []
for batch_idx in range(input_embeds.shape[0]):
    causal_mask = torch.full(
        (target_length, target_length), fill_value=min_dtype
        , dtype=dtype, device=device
    )
    diagonal_attend_mask = torch.arange(target_length, device=device, dtype=torch.int32)
    diagonal_attend_mask = diagonal_attend_mask > diagonal_attend_mask.reshape(-1, 1)

    replace_indices_groups = torch.nonzero(loss_mask[batch_idx] == 1, as_tuple=True)[0].view(-1, jacobi_tokens)
    x = loss_mask[batch_idx].repeat(diagonal_attend_mask.shape[-1], 1)
    for i in replace_indices_groups:
        x[i[0]:i[-1]+1, i[0]:i[-1]+1] = 0

    diagonal_attend_mask.bitwise_or_(x.type(torch.bool))
    causal_mask *= diagonal_attend_mask
    final_mask.append(causal_mask.unsqueeze(0))
    print(diagonal_attend_mask.type(torch.int32))

final_mask = torch.stack(final_mask, dim=0)

print(final_mask.shape)




    # x = loss_mask[batch_idx].repeat(diagonal_attend_mask.shape[-1], 1)
    # print(x)
    # print(replace_indices_groups)
    # x[replace_indices_groups]=0
    # print(x)
    # loss_mask
    # diagonal_attend_mask = cache_position[batch_idx] >= cache_position[batch_idx].reshape(-1, 1)
    # print(diagonal_attend_mask.type(torch.int32))
    # print(diagonal_attend_mask.bitwise_or_(loss_mask[batch_idx].type(torch.bool)))
    # causal_mask *= diagonal_attend_mask
    # print(causal_mask + torch.eye(target_length, dtype=dtype))
    