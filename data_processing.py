import os
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

class CustomDataset(Dataset):
    def __init__(self, datapath, max_len=2048, jacobi_tokens=10, transform=None):
        self.data = datapath

        self.max_len = max_len
        self.jacobi_tokens = jacobi_tokens

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        new_data = {}

        raw_data = torch.load(self.data[index])
        input_ids = raw_data['input_ids']#[:self.max_len][None, :]
        generated_tokens = raw_data['generated_tokens']#[:self.max_len][None, :]
        hidden_states = raw_data['hidden_states']#[:self.max_len][None, :]
        
        # truncate to max length
        input_nums = input_ids.shape[0]
        generated_nums = generated_tokens.shape[0]
        if input_nums + generated_nums > self.max_len:
            generated_nums = self.max_len - input_nums

        start_limitation = generated_nums - self.jacobi_tokens - 2 
        start_index = random.randint(0, start_limitation)
        
        input_ids = torch.cat([input_ids, generated_tokens[:start_index]],dim=-1) if start_index > 0 else input_ids
        input_ids_target = generated_tokens[start_index+1:start_index+self.jacobi_tokens+1]
        attention_mask = [1] * (input_ids.shape[0]) #  + self.jacobi_tokens)
        loss_mask = [0] * (input_ids.shape[0]) + [1] *self.jacobi_tokens

        hidden_states = hidden_states[start_index:start_index+self.jacobi_tokens]
        if self.transform:
            hidden_states = self.transform(hidden_states)

        new_data["input_ids"] = input_ids[None, :]
        new_data["target"] = input_ids_target[None, :]
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["hidden_state_target"] = hidden_states[None, :]

        return new_data



        # loss_mask = data["loss_mask"][:self.max_len][None, :]

        
        # length = hidden_states.shape[1]
        # # length_q = data['query_ids'].shape[1]
        # attention_mask = [1] * length
        # loss_mask = loss_mask[0].tolist()
        # loss_mask[-1] = 0

        # input_ids_target = input_ids[:, 1:]
        # zeropadding = torch.tensor([[0]])
        # input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        # target = hidden_state[:, 1:, :]
        # zeropadding = torch.zeros(1, 1, target.shape[2])
        # target = torch.cat((target, zeropadding), dim=1)
        # loss_mask[-1] = 0
        # new_data["attention_mask"] = attention_mask
        # new_data["loss_mask"] = loss_mask
        # new_data["target"] = target
        # new_data["hidden_state_big"] = hidden_state
        # new_data["input_ids"] = input_ids_target


        # if self.transform:
        #     new_data = self.transform(new_data)

        return new_data

class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(len(item['loss_mask']) for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        # batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        # batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        # batch_loss_mask = torch.tensor(
        #     [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])

        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_state_target": torch.cat([item['hidden_state_target'] for item in features]),
            "target": torch.cat([item['target'] for item in features]),
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask
        }
        return batch

        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

# for i in traindataset:
#     for key in i:
#         if isinstance(i[key], list):
#             print(len(i[key]))
#         else:
#             print(key, i[key].shape)
#     break

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_config = {
    "datapath": "./data_root/ShareGPT_Vicuna_unfiltered_Qwen2.5-0.5B-Instruct",
    "bs": 4,
    "num_workers": 0
    }

    datapath = list_files(train_config["datapath"])
    traindatapath = datapath  #[:int(len(datapath) * 0.95)]

    traindataset = CustomDataset(traindatapath)
    train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                            pin_memory=True)

    count = 0
    for i in train_loader:
        print("="*50)
        for key in i:
            if (key in ['loss_mask','attention_mask']):
                print(key, i[key].shape, torch.sum(i[key], dim=-1))
            else:
                print(key, i[key].shape)
        count += 1
        if count == 10:
            break