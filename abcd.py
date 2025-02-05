from safetensors.torch import load_file, save_file
import re

# Load safetensors file
input_file = "./jacobi_test_weights/CFG9.9-con/state_3/model.safetensors"   # Change to your actual file path
output_file = "./jacobi_test_weights/CFG9.9-con/state_3/model_modify.safetensors"

# Load tensor dictionary
tensors = load_file(input_file)

# Modify the keys
updated_tensors = {}
for key, value in tensors.items():
    print(key)
    new_key = re.sub(r'adapters\.0\.', 'adapters.', key)  # Remove '.0'
    updated_tensors[new_key] = value  # Assign the tensor to new key

# Save back to a new safetensors file
save_file(updated_tensors, output_file)

print(f"✅ Updated safetensors saved to: {output_file}")