import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Params
DROPPED_CLASS = -1
DROP_PROB = 0.5
SHAKE_STD = 0.05
FAKE_CLASS = 3  # Still included, just unused

# Load and prepare data
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
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Data Augmentation
def apply_dropper(x, y, dropout_prob=DROP_PROB):
    keep_mask = torch.rand(x.shape[0]) > dropout_prob
    y_dropped = y.clone()
    y_dropped[~keep_mask] = DROPPED_CLASS
    return x, y_dropped

def apply_shaker(x, y, noise_std=SHAKE_STD):
    keep_mask = y != DROPPED_CLASS
    x_shaken = x.clone()
    x_shaken[keep_mask] += torch.randn_like(x_shaken[keep_mask]) * noise_std
    return x_shaken, y

# Model definition
class PointNet(nn.Module):
    def __init__(self, num_classes=4, point_dimension=2):
        super(PointNet, self).__init__()
        a = 3
        nfeat = 64
        self.conv_1_a = nn.Conv1d(1, nfeat, kernel_size=a, padding=1)
        self.conv_1_b = nn.Conv1d(1, nfeat, kernel_size=a, padding=1)
        self.conv_2 = nn.Conv1d(nfeat, nfeat, kernel_size=a)
        self.bn_1 = nn.BatchNorm1d(nfeat)
        self.bn_2 = nn.BatchNorm1d(nfeat)

    def forward(self, x):
        batch_size, num_points, _ = x.shape
        x_freq = x[:, :, 0].unsqueeze(1)
        x_modnu = x[:, :, 1].unsqueeze(1)
        x_freq = F.pad(x_freq, (1, 1), mode='replicate')
        x_modnu = F.pad(x_modnu, (1, 1), mode='circular')
        x = self.conv_1_a(x_freq) + self.conv_1_b(x_modnu)
        x = F.relu(self.bn_1(x))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        return x.view(batch_size, -1)

class SegmentationPointNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = PointNet()
        nfeat = 64
        nhidden = 64
        self.fc_1 = nn.Linear(nfeat + 2, nhidden)
        self.fc_2 = nn.Linear(nhidden, num_classes)

    def forward(self, x):
        global_feat = self.base_pointnet(x).unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, global_feat], dim=2)
        x = F.relu(self.fc_1(x))
        return F.log_softmax(self.fc_2(x), dim=2)

# Training setup
device = torch.device("cpu")
model = SegmentationPointNet(num_classes=4).to(device)
loss_fn = nn.NLLLoss(reduction="none", ignore_index=DROPPED_CLASS)
optimizer = optim.Adam(model.parameters(), lr=0.001)

history_train, history_test = [], []
Nepoch = 150
nbatches = 100
batch_size = X_train.shape[0] // nbatches

for epoch in range(Nepoch):
    model.train()
    for i in range(nbatches):
        optimizer.zero_grad()
        x_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train[i*batch_size:(i+1)*batch_size]

        x_aug_list, y_aug_list = [], []
        for j in range(x_batch.shape[0]):
            xj, yj = apply_dropper(x_batch[j], y_batch[j])
            xj, yj = apply_shaker(xj, yj)
            x_aug_list.append(xj)
            y_aug_list.append(yj)

        max_len = max([xj.shape[0] for xj in x_aug_list])
        x_padded = torch.zeros(len(x_aug_list), max_len, 2)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long)

        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0]] = xj
            y_padded[j, :yj.shape[0]] = yj

        seg_output = model(x_padded)
        cost = loss_fn(seg_output.permute(0, 2, 1), y_padded).mean()
        cost.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_out_eval = model(X_test)
        cost_test = loss_fn(test_out_eval.permute(0, 2, 1), y_test).mean()

    history_train.append(float(cost))
    history_test.append(float(cost_test))
    print(f"Epoch {epoch}: Train Loss = {cost:.4f}, Test Loss = {cost_test:.4f}")

# Final evaluation
model.eval()
with torch.no_grad():
    test_out = model(X_test).permute(0, 2, 1)
    test_losses = loss_fn(test_out, y_test)
    per_model_loss = test_losses.mean(dim=1).cpu().numpy()
    per_point_loss = test_losses.cpu().numpy()
    test_predictions = test_out.argmax(dim=1).cpu().numpy()
    test_targets = y_test.cpu().numpy()
    cost_test = test_losses.mean().item()

    valid_mask = test_targets != DROPPED_CLASS
    correct = (test_predictions[valid_mask] == test_targets[valid_mask]).sum()
    total = valid_mask.sum()
    test_accuracy = correct / total if total > 0 else 0.0

# Save files with suffix P7ds
suffix = "P7ds"
torch.save(model.state_dict(), f"segmentation_pointnet_{suffix}.pth")
with open(f"train_history_{suffix}.pkl", "wb") as f:
    pickle.dump(history_train, f)
with open(f"test_history_{suffix}.pkl", "wb") as f:
    pickle.dump(history_test, f)
with open(f"output_{suffix}.pkl", "wb") as f:
    pickle.dump(test_out, f)
with open(f"cost_{suffix}.pkl", "wb") as f:
    pickle.dump(cost_test, f)
with open(f"test_targets_{suffix}.pkl", "wb") as f:
    pickle.dump(test_targets, f)
with open(f"test_predictions_{suffix}.pkl", "wb") as f:
    pickle.dump(test_predictions, f)
with open(f"per_point_losses_test_{suffix}.pkl", "wb") as f:
    pickle.dump(per_point_loss, f)
with open(f"average_loss_per_model_{suffix}.pkl", "wb") as f:
    pickle.dump(per_model_loss, f)
with open(f"test_accuracy_{suffix}.pkl", "wb") as f:
    pickle.dump(test_accuracy, f)

print(f"Number of test diagrams: {per_model_loss.shape[0]}")
print(f"Final test accuracy (no dropped): {test_accuracy:.4f}")