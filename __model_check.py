import torch
from safetensors import safe_open

from models.qwen2.modeling_qwen2_jacobi import Qwen2JacobiForCausalLM

model = Qwen2JacobiForCausalLM.from_pretrained(
    pretrained_model_name_or_path="./Qwen2.5-0.5B-Instruct",
    adapter_type="Qwen2MLP",
    shared_adapter=False,
    torch_dtype="auto",
    device_map="auto"
)

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

cpdir = "./jacobi_test_weights/test-13/state_25/model.safetensors"
load_jacobi_weight(model, cpdir)