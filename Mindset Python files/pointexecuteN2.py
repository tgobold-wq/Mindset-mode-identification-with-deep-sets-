from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import pickle

#os.environ["OMP_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ------- Data Preparation ------- #

grid = pd.read_pickle("outputs_in_instab_strip.pkl")
l_values = [0, 1, 2] 
l_columns = {l: [col for col in grid.columns if col.endswith(f"_l_{l}")] for l in l_values}
X_frequencies = np.hstack([grid[l_columns[l]].values for l in l_values])  
delta_nu_values = grid["delta_nu"].values[:, np.newaxis] 
mod_frequencies = X_frequencies % delta_nu_values 
X = np.stack((X_frequencies, mod_frequencies), axis=-1)
y = np.hstack([np.full_like(grid[l_columns[l]].values, l) for l in l_values])  

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
size = X_train.shape[0]
inpt = X_train[:size]
print(inpt.shape)

# ------- Model Definition ------- #

class PointNet(nn.Module):
    def __init__(self, num_classes=3, point_dimension=2, dropout=0.3):
        super(PointNet, self).__init__()
        
        a=3
        nfeat=64
        
        self.conv_1 = nn.Conv1d(point_dimension, nfeat, a, padding=2)
        self.conv_2 = nn.Conv1d(nfeat, nfeat, a, padding=2)
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)

    def forward(self, x, plot=False):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = nn.MaxPool1d(num_points)(x)
        return x

class SegmentationPointNet(nn.Module):
    def __init__(self, num_classes, point_dimension=2, dropout=0.3):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = PointNet(point_dimension=point_dimension)

        nhidden=16
        nfeat=64
        
        self.fc_1 = nn.Linear(nfeat+2, nhidden)
        self.fc_2 = nn.Linear(nhidden, 3)

    def forward(self, x):
        global_features = self.base_pointnet(x)
        global_features = global_features.transpose(2, 1)
        global_features = global_features.repeat(1, x.shape[1], 1)
        x = torch.cat((x, global_features), dim=2)
        x = F.relu(self.fc_1(x))
        return F.log_softmax(self.fc_2(x), dim=2)

# ------- Training Setup ------- #

model = SegmentationPointNet(num_classes=3, point_dimension=2)
loss_fn = nn.NLLLoss(reduction='none')  # Per-point loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

device = torch.device("cpu")
model = model.to(device)

target = y_train[:size].to(device)
inpt_test = inpt.to(device)

history_train, history_test = [], []
Nepoch = 150
nbatches = 100
batch_size = inpt.size(0) // nbatches
print(batch_size)

for epoch in range(Nepoch):
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs_batch = inpt[i*batch_size:(i+1)*batch_size].to(device)
        targets_batch = target[i*batch_size:(i+1)*batch_size].to(device)

        output = model(inputs_batch)
        output = output.permute(0, 2, 1)  # [batch, points, classes]

        # Compute per-point loss
        per_point_loss = loss_fn(output, targets_batch)
        cost = per_point_loss.mean()  # Scalar loss for backprop

        cost.backward()
        optimizer.step()

    out_test = model(inpt_test).to(device)
    out_test_permuted = out_test.permute(0, 2, 1)
    cost_test = loss_fn(out_test_permuted, target).mean().to(device)

    history_train.append(float(cost))
    history_test.append(float(cost_test))
    print(f'Epoch {epoch}: Train Loss = {cost:.4f}, Test Loss = {cost_test:.4f}')

# ------- Save Model and History ------- #

torch.save(model.state_dict(), "segmentation_pointnet.pth")

with open("train_historyN2.pkl", "wb") as f:
    pickle.dump(history_train, f)
with open("test_historyN2.pkl", "wb") as f:
    pickle.dump(history_test, f)
with open("outputN2.pkl", "wb") as f:
    pickle.dump(out_test, f)
with open("costN2.pkl", "wb") as f:
    pickle.dump(cost_test, f)
with open("test_targetsN2.pkl", "wb") as f:
    pickle.dump(target.cpu().numpy(), f)
with open("test_predictionsN2.pkl", "wb") as f:
    pickle.dump(out_test.detach().cpu().numpy(), f)



# ------- Compute and Save Per-Model Losses ------- #

model.eval()
with torch.no_grad():
    out_test = model(inpt_test)
    out_test_permuted = out_test.permute(0, 2, 1)  # [batch, points, classes]
    per_point_losses = loss_fn(out_test_permuted, target)  # Shape: [batch, points]
    per_model_losses = per_point_losses.mean(dim=1).cpu().numpy()  # [batch,]
    per_point_losses_np = per_point_losses.cpu().numpy()


print(f"Number of echelle diagrams in test set: {per_model_losses.shape[0]}")
with open("per_point_losses_testN2.pkl", "wb") as f:
    pickle.dump(per_point_losses_np, f)
    
# Save for future inspection
with open("average_loss_per_modelN2.pkl", "wb") as f:
    pickle.dump(per_model_losses, f)
"""
# ------- Plot Histogram of Per-Model Losses ------- #

plt.figure(figsize=(8, 5))
plt.hist(per_model_losses, bins=40, color="steelblue", edgecolor="black")
plt.xlabel("Average loss per echelle diagram")
plt.ylabel("Number of diagrams")
plt.title("Loss Distribution Across Echelle Diagrams")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_per_model_histogram.png", dpi=300)
plt.show()
"""