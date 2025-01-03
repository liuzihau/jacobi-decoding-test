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


mix_sequence = 1
hidden_states = torch.randn((2, 24, 4))
loss_mask = torch.Tensor([
    [0] * 10 + [1] * 5 + [0] * 9,
    [0] * 19 + [1] * 5
])

# replace_matrix = []
# for i in range(loss_mask.shape[0]):
#     replace_indices = torch.nonzero(loss_mask[i] == 1, as_tuple=True)[0]
#     all_indices = torch.cat([replace_indices[:mix_sequence] - mix_sequence, replace_indices], dim=-1)
#     replace_matrix.append(hidden_states[i, all_indices])
# replace_matrix = torch.stack(replace_matrix, 0)
# print(hidden_states)
# print(replace_matrix.shape)


replace_indices = torch.nonzero(loss_mask == 1, as_tuple=True)
# all_indices = torch.cat([replace_indices[:mix_sequence] - mix_sequence, replace_indices], dim=-1)
print(replace_indices)