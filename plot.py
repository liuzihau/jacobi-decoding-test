import torch
import matplotlib.pyplot as plt

# Data
tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Jacobi token numbers
CFG2_res = {
    "top_1": [0.12388, 0.09454, 0.07987, 0.06928, 0.06194, 0.06316, 0.06276, 0.05787, 0.05868, 0.05420],
    "top_2": [0.17930, 0.14711, 0.12755, 0.11002, 0.11206, 0.09658, 0.09984, 0.09536, 0.10921, 0.09373],
    "top_3": [0.21883, 0.19519, 0.16096, 0.15363, 0.14914, 0.13081, 0.12918, 0.12877, 0.13570, 0.12510]
}

# Plot
plt.figure(figsize=(10, 6))

plt.plot(tokens, CFG2_res["top_1"], label="Top 1 Accuracy", marker='o')
plt.plot(tokens, CFG2_res["top_2"], label="Top 2 Accuracy", marker='s')
plt.plot(tokens, CFG2_res["top_3"], label="Top 3 Accuracy", marker='^')

# Labels and Title
plt.xlabel("Jacobi Token Number", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy vs Jacobi Token Number", fontsize=14)
plt.legend()

# Grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.savefig("accuracy_plot.png")



import matplotlib.pyplot as plt

sample = 300

data = torch.load('./record/training_data_token_counts.pt')
data2 = torch.load('./record/CFG5_epoch_counts.pt', map_location=torch.device('cpu'))

# Prepare data for plot
indices = range(sample)
values = data.numpy()
values = (values / values.sum(-1) )[:sample]
values2 = data2.numpy()[-1, 0, :]
values2 = (values2 / values2.sum(-1) )[:sample]
print(values.shape, values2.shape)

# Plot distribution
plt.figure(figsize=(10, 6))
plt.scatter(indices, values, s=1, color='blue', alpha=0.6, label="Training Data")  # Scatter plot for data
plt.scatter(indices, values2, s=1, color='red', alpha=0.6, label="CFG5 Epoch Data")  # Scatter plot for data2

# Labels and Title
plt.xlabel("Index", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Index vs Value Distribution", fontsize=14)
plt.legend()

# Save plot
plt.tight_layout()
plt.savefig("tensor_index_vs_value_combined_300.png")


