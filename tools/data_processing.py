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
    def __init__(self, datapath, max_len=2048, jacobi_tokens=10, use_multi_token_sets=False, transform=None, pad_id=151643, vocab_size=151936):
        self.data = datapath
        self.max_len = max_len
        self.jacobi_tokens = jacobi_tokens
        self.use_multi_token_sets = use_multi_token_sets
        self.transform = transform
        self.pad = pad_id
        self.fake_id = vocab_size + 2025

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

        if self.use_multi_token_sets:
            possible_sets = (self.max_len - input_nums - self.jacobi_tokens) // (self.jacobi_tokens+1) + 1
            start_limitation = generated_nums - self.jacobi_tokens - 2
            if possible_sets > start_limitation:
                start, end = 0, start_limitation
            else:
                start = 0  # random.randint(0, start_limitation-possible_sets)
                end = start + possible_sets

            input_ids_target = generated_tokens[start+1:end+self.jacobi_tokens+1].unfold(0, self.jacobi_tokens, 1).reshape(-1)
            
            hidden_states = hidden_states[start+1:end+self.jacobi_tokens+1].unfold(0, self.jacobi_tokens, 1)
            hidden_states = torch.permute(hidden_states, (0, 2, 1))
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

            input_ids = torch.cat([input_ids, generated_tokens[:start]],dim=-1) if start > 0 else input_ids
            generated_tokens = generated_tokens[start:end]
            padding_tensor = torch.ones((self.jacobi_tokens,), dtype=generated_tokens.dtype) * self.fake_id
            padding_tensors = torch.stack([padding_tensor] * generated_tokens.shape[0], dim=0)
            merged_tokens = torch.cat([generated_tokens[:, None], padding_tensors], dim=-1).flatten()  #[token1, 0, 0, token2, 0, 0, ...]
            input_ids = torch.cat([input_ids, padding_tensor, merged_tokens])

            jacobi_indices = torch.nonzero(input_ids == self.fake_id, as_tuple=True)
            loss_mask = torch.zeros_like(input_ids)
            input_ids[jacobi_indices] = self.pad
            loss_mask[jacobi_indices] = 1
            loss_mask = loss_mask.tolist()

            attention_mask = [1] * input_ids.shape[0]

        elif True:
            # ================================================================================================ the same
            possible_sets = (self.max_len - input_nums - self.jacobi_tokens) // (self.jacobi_tokens+1) + 1
            start_limitation = generated_nums - self.jacobi_tokens - 2
            if possible_sets > start_limitation:
                start, end = 0, start_limitation
            else:
                start = 0  # random.randint(0, start_limitation-possible_sets)
                end = start + possible_sets

            input_ids_target = generated_tokens[start+1:end+self.jacobi_tokens+1].unfold(0, self.jacobi_tokens, 1).reshape(-1)
            
            hidden_states = hidden_states[start+1:end+self.jacobi_tokens+1].unfold(0, self.jacobi_tokens, 1)
            hidden_states = torch.permute(hidden_states, (0, 2, 1))
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            # ================================================================================================
            print(input_ids.shape)
            print(generated_tokens.shape)
            input_ids = torch.cat([input_ids, generated_tokens[:start]],dim=-1) if start > 0 else input_ids
            print(input_ids.shape)
            generated_tokens = generated_tokens[start:end]
            print(generated_tokens.shape)
            padding_tensors = torch.ones((self.jacobi_tokens * (generated_tokens.shape[0] + 1),), dtype=generated_tokens.dtype) * self.fake_id
            # padding_tensors = torch.cat([padding_tensor] * generated_tokens.shape[0], dim=0)
            print(padding_tensors.shape)
            # merged_tokens = torch.cat([generated_tokens[:, None], padding_tensors], dim=-1).flatten()  #[token1, 0, 0, token2, 0, 0, ...]
            input_ids = torch.cat([input_ids, generated_tokens, padding_tensors])
            print(input_ids.shape)

            jacobi_indices = torch.nonzero(input_ids == self.fake_id, as_tuple=True)
            loss_mask = torch.zeros_like(input_ids)
            input_ids[jacobi_indices] = self.pad
            loss_mask[jacobi_indices] = 1
            loss_mask = loss_mask.tolist()

            attention_mask = [1] * input_ids.shape[0]

        else:
            start_limitation = generated_nums - self.jacobi_tokens - 2 
            start_index = 33  # random.randint(0, start_limitation)
            
            input_ids = torch.cat([input_ids, generated_tokens[:start_index]],dim=-1) if start_index > 0 else input_ids
            input_ids_target = generated_tokens[start_index+1:start_index+self.jacobi_tokens+1]
            attention_mask = [1] * (input_ids.shape[0] + self.jacobi_tokens)
            loss_mask = [0] * (input_ids.shape[0]) + [1] *self.jacobi_tokens

            hidden_states = hidden_states[start_index+1:start_index+self.jacobi_tokens+1]
        
        if self.transform:
            hidden_states = self.transform(hidden_states)

        new_data["input_ids"] = input_ids[None, :]
        # new_data["target"] = input_ids_target[None, :]
        new_data["target"] = input_ids_target
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        # new_data["hidden_state_target"] = hidden_states[None, :]
        new_data["hidden_state_target"] = hidden_states
        new_data["filename"] = self.data[index]

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
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch = {
            "input_ids": batch_input_ids,
            # "hidden_state_target": torch.cat([item['hidden_state_target'] for item in features]),
            "hidden_state_target": torch.cat([item['hidden_state_target'] for item in features]),
            "target": torch.cat([item['target'] for item in features]),
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "filename": [item['filename'] for item in features]
        }
        return batch

class InferenceDataset(Dataset):
    def __init__(self, datapath):
        self.data = datapath

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        new_data = {}

        raw_data = torch.load(self.data[index])
        input_ids = raw_data['input_ids']#[:self.max_len][None, :]
        return input_ids