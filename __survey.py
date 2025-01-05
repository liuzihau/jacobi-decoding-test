from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data_processing import CustomDataset, DataCollatorWithPadding, list_files
from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM
from models.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer


def survey_total_trainable_parameters(pretrained_model_name_or_path="./Qwen2.5-0.5B-Instruct", jacobi_token_nums=10, mix_sequences=1, proj_freq=4, adapter_type='Linear', shared_adapter=False, shared_jacobi_token=True):
    
    def count_trainable_parameters(model):
        x = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
                x+=p.numel()
        return x

    model = Qwen2JacobiForCausalLM.from_pretrained(pretrained_model_name_or_path, jacobi_token_nums, mix_sequences, proj_freq, adapter_type, shared_adapter, shared_jacobi_token, torch_dtype="auto", device_map="auto")
    for param in model.model.parameters():
        param.requires_grad = False

    total_params = count_trainable_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    return total_params


def survey_training_data_token_distribution(pretrained_model_name_or_path="./Qwen2.5-0.5B-Instruct", datapath='./data_root/ShareGPT_Vicuna_unfiltered_Qwen2.5-0.5B-Instruct', jacobi_tokens=10, batch_size=16, vocal_dim=151936):
    datapath = list_files(datapath)
    custom_dataset = CustomDataset(datapath, jacobi_tokens=jacobi_tokens)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=4, pin_memory=True)
    
    tokenizer = Qwen2Tokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)
    counts = torch.zeros((vocal_dim,),dtype=torch.int16)
    for batch_idx, data in enumerate(tqdm(data_loader)):
        flattened_data = data['target'].flatten()
        c = torch.bincount(flattened_data)
        ids = torch.nonzero(c, as_tuple=True)[0]
        counts[ids] += c[ids]
        if (batch_idx + 1) % 500 == 0:
            report = f"[{batch_idx + 1:5d}] "
            top_5 = counts.argsort(descending=True)[:5]
            for i, token in enumerate(top_5):
                decode = tokenizer.decode([token])
                decode = "\\n" if decode == '\n' else decode
                report += f"<top {i+1}: {decode}({counts[token]} times)>, "
            print(report)
    torch.save(counts, "./counts.pt")
    return counts

def load():
    x = torch.load('./counts.pt')
    print(x.shape)

if __name__ == "__main__":
    survey_total_trainable_parameters()
    # counts = survey_training_data_token_distribution()
    
