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

        a = 3 # 5 # for non periodic 1, 3, 5

        nfeat = 64 # 128

        # Separate convolution for freq and nu mod delta nu
        self.conv_1_a = nn.Conv1d(1, nfeat, kernel_size=a, padding=1)
        self.conv_1_b = nn.Conv1d(1, nfeat, kernel_size=a, padding=1)

        # network has to see that for example 0.01 and 0.96 are close to each other in circular space
        # but why kernel size 3? this means that the convolution considers neighbourhood of 3 adjacent values -> current point, left and right

        self.conv_2 = nn.Conv1d(nfeat, nfeat, kernel_size=a)

        self.bn_1 = nn.BatchNorm1d(nfeat)
        self.bn_2 = nn.BatchNorm1d(nfeat)

    def forward(self, x, plot=False):
        # x shape: [batch_size, num_points, 2]
        batch_size, num_points, _ = x.shape # torch.Size([B, 30, 2])
        # Split channels and reshape to [batch_size, 1, num_points]
        x_freq = x[:, :, 0].unsqueeze(1)      # torch.Size([B, 1, 30])
        x_modnu = x[:, :, 1].unsqueeze(1)     # torch.Size([B, 1, 30])

        # circular padding to the periodic input (pad 1 on each side for kernel_size=3)
        x_freq = F.pad(x_freq, (1, 1), mode='replicate')
        x_modnu = F.pad(x_modnu, (1, 1), mode='circular')

        # convolutions as before
        x_freq_feat = self.conv_1_a(x_freq)       # [B, 64, N]
        x_modnu_feat = self.conv_1_b(x_modnu)     # [B, 64, N]

        # Combine features from both channels
        x = x_freq_feat + x_modnu_feat

        # Apply batch norm + relu + next conv
        x = F.relu(self.bn_1(x))
        x = F.relu(self.bn_2(self.conv_2(x))) # ? [B , 64, N-2]

        x = nn.AvgPool1d(num_points)(x)  # THIS IS THE CHAAAAAAAAAAANGE
        x = x.view(batch_size, -1)       # [B, 64]

        return x
        
class SegmentationPointNet(nn.Module):
    def __init__(self, num_classes, point_dimension=2, dropout=0.3):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = PointNet(point_dimension=point_dimension)

        nhidden = 64 # 64
        nfeat = 64 # 128
        
        # Per-point classification layers
        self.fc_1 = nn.Linear(nfeat+2, nhidden)
        self.fc_2 = nn.Linear(nhidden, 3)

    def forward(self, x):
        global_features = self.base_pointnet(x)  # shape: [batch_size, 64]
        global_features = global_features.unsqueeze(1)  # shape: [batch_size, 1, 64]
        global_features = global_features.repeat(1, x.shape[1], 1)  # [batch_size, num_points, 66]
        # 66 comes from 2 coordinates so to speak the the 64 global feature elements.
        x = torch.cat((x, global_features), dim=2)  # x is [batch_size, num_points, 2]
        
        # need to paste the global features after the individual points as inputs for the segmentation (classification each point in the cloud)
        x = F.relu(self.fc_1(x)) 

        return F.log_softmax(self.fc_2(x), dim=2)

# ------- Training Setup ------- #

model = SegmentationPointNet(num_classes=3, point_dimension=2)
loss_fn = nn.NLLLoss(reduction='none')  # Per-point loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

torch.save(model.state_dict(), "segmentation_pointnet_P7meanp.pth")

with open("train_historyP7meanp.pkl", "wb") as f:
    pickle.dump(history_train, f)
with open("test_historyP7meanp.pkl", "wb") as f:
    pickle.dump(history_test, f)
with open("outputP7meanp.pkl", "wb") as f:
    pickle.dump(out_test, f)
with open("costP7meanp.pkl", "wb") as f:
    pickle.dump(cost_test, f)
with open("test_targetsP7meanp.pkl", "wb") as f:
    pickle.dump(target.cpu().numpy(), f)
with open("test_predictionsP7meanp.pkl", "wb") as f:
    pickle.dump(out_test.detach().cpu().numpy(), f)

model.eval()
with torch.no_grad():
    out_test = model(inpt_test)
    out_test_permuted = out_test.permute(0, 2, 1)
    per_point_losses = loss_fn(out_test_permuted, target)
    per_model_losses = per_point_losses.mean(dim=1).cpu().numpy()
    per_point_losses_np = per_point_losses.cpu().numpy()

print(f"Number of echelle diagrams in test set: {per_model_losses.shape[0]}")

with open("per_point_losses_testP7meanp.pkl", "wb") as f:
    pickle.dump(per_point_losses_np, f)
with open("average_loss_per_modelP7meanp.pkl", "wb") as f:
    pickle.dump(per_model_losses, f)
