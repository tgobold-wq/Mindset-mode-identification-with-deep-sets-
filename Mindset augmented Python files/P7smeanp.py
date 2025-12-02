import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHAKE_STD = 0.05
REAL_CLASSES = [0, 1, 2]
DROPPED_CLASS = -1

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

X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

class PointNet(nn.Module):
    def __init__(self, num_classes=3, point_dimension=2, aggregator="mean"):
        super(PointNet, self).__init__()
        nfeat = 64
        a = 3
        self.conv_1_a = nn.Conv1d(1, nfeat, kernel_size=a, padding=1)
        self.conv_1_b = nn.Conv1d(1, nfeat, kernel_size=a, padding=1)
        self.conv_2 = nn.Conv1d(nfeat, nfeat, kernel_size=a, padding=1)
        self.bn_1 = nn.BatchNorm1d(nfeat)
        self.bn_2 = nn.BatchNorm1d(nfeat)
        self.aggregator = aggregator

    def forward(self, x):
        batch_size, num_points, _ = x.shape
        x_freq = x[:, :, 0].unsqueeze(1)
        x_modnu = x[:, :, 1].unsqueeze(1)
        x_freq = F.pad(x_freq, (1, 1), mode='replicate')
        x_modnu = F.pad(x_modnu, (1, 1), mode='circular')
        x_freq_feat = self.conv_1_a(x_freq)
        x_modnu_feat = self.conv_1_b(x_modnu)
        x = x_freq_feat + x_modnu_feat
        x = F.relu(self.bn_1(x))
        x = F.relu(self.bn_2(self.conv_2(x)))
        if self.aggregator == "max":
            x = F.max_pool1d(x, kernel_size=num_points)
        elif self.aggregator == "mean":
            x = F.avg_pool1d(x, kernel_size=num_points)
        x = x.view(batch_size, -1)
        return x

class SegmentationPointNet(nn.Module):
    def __init__(self, num_classes=3, point_dimension=2, aggregator="mean"):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = PointNet(num_classes=num_classes, point_dimension=point_dimension, aggregator=aggregator)
        nhidden = 64
        nfeat = 64
        self.fc_1 = nn.Linear(nfeat + 2, nhidden)
        self.fc_2 = nn.Linear(nhidden, num_classes)

    def forward(self, x):
        global_features = self.base_pointnet(x)
        global_features = global_features.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat((x, global_features), dim=2)
        x = F.relu(self.fc_1(x))
        return F.log_softmax(self.fc_2(x), dim=2)

def apply_shaker(x, y, noise_std=SHAKE_STD):
    x_new = x.clone()
    shake_amount = torch.randn_like(x_new) * noise_std
    x_new = x_new + shake_amount
    return x_new, y, shake_amount

model = SegmentationPointNet(num_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.NLLLoss(reduction="mean")

history_train = []
history_val = []
history_val_acc = []

Nepoch = 150
nbatches = 100
num_stars = X_train.shape[0]
batch_size = max(1, num_stars // nbatches)

for epoch in range(Nepoch):
    model.train()
    perm = torch.randperm(num_stars, device=device)
    epoch_loss = 0.0
    n_batches_done = 0
    for bstart in range(0, num_stars, batch_size):
        optimizer.zero_grad()
        batch_idx = perm[bstart:bstart+batch_size]
        x_batch = torch.stack([X_train[i] for i in batch_idx], dim=0)
        y_batch = torch.stack([y_train[i] for i in batch_idx], dim=0)
        x_aug_list = []
        y_aug_list = []
        for j in range(x_batch.shape[0]):
            xj, yj, _ = apply_shaker(x_batch[j], y_batch[j])
            x_aug_list.append(xj)
            y_aug_list.append(yj)
        max_len = max(xj.shape[0] for xj in x_aug_list)
        x_padded = torch.zeros(len(x_aug_list), max_len, 2, device=device)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long, device=device)
        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0], :] = xj
            y_padded[j, :yj.shape[0]] = yj
        out = model(x_padded)
        out_perm = out.permute(0, 2, 1)
        loss = loss_fn(out_perm, y_padded)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
        n_batches_done += 1
    avg_train_loss = epoch_loss / n_batches_done if n_batches_done > 0 else 0.0
    history_train.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        out_val = model(X_val)
        out_val_perm = out_val.permute(0, 2, 1)
        val_loss = loss_fn(out_val_perm, y_val).item()
        preds_val = out_val.argmax(dim=2)
        mask = y_val != DROPPED_CLASS
        if mask.any():
            correct = (preds_val[mask] == y_val[mask]).sum().item()
            total = mask.sum().item()
            val_acc = correct / total
        else:
            val_acc = 0.0
    history_val.append(val_loss)
    history_val_acc.append(val_acc)
    print(f"Epoch {epoch+1}/{Nepoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

model.eval()
with torch.no_grad():
    out_test = model(X_test)
    out_test_perm = out_test.permute(0, 2, 1)
    loss_fn_no_reduce = nn.NLLLoss(ignore_index=DROPPED_CLASS, reduction="none")
    per_point_loss = loss_fn_no_reduce(out_test_perm, y_test)
    per_point_loss_np = per_point_loss.cpu().numpy()
    per_model_loss = per_point_loss.mean(dim=1).cpu().numpy()
    cost_test = per_point_loss.mean().item()
    test_predictions = out_test.argmax(dim=2).cpu().numpy()
    test_targets = y_test.cpu().numpy()
    test_probs = F.softmax(out_test, dim=2).cpu().numpy()
    mask_all = test_targets != DROPPED_CLASS
    if mask_all.any():
        correct = (test_predictions[mask_all] == test_targets[mask_all]).sum()
        total = mask_all.sum()
        test_accuracy = float(correct) / float(total)
    else:
        test_accuracy = 0.0
    shake_info_test = {}
    for i in range(X_test.shape[0]):
        x_i = X_test[i]
        y_i = y_test[i]
        x_aug, y_aug, shake_i = apply_shaker(x_i, y_i)
        shake_info_test[f"test_model_{i}"] = shake_i.cpu().numpy()

torch.save(model.state_dict(), "segmentation_pointnet_smeanp.pth")
with open("output_logits_smeanp.pkl", "wb") as f: pickle.dump(out_test.cpu().numpy(), f)
with open("output_probs_smeanp.pkl", "wb") as f: pickle.dump(test_probs, f)
with open("cost_smeanp.pkl", "wb") as f: pickle.dump(cost_test, f)
with open("test_targets_smeanp.pkl", "wb") as f: pickle.dump(test_targets, f)
with open("test_predictions_smeanp.pkl", "wb") as f: pickle.dump(test_predictions, f)
with open("per_point_losses_test_smeanp.pkl", "wb") as f: pickle.dump(per_point_loss_np, f)
with open("average_loss_per_model_smeanp.pkl", "wb") as f: pickle.dump(per_model_loss, f)
with open("test_accuracy_smeanp.pkl", "wb") as f: pickle.dump(test_accuracy, f)
with open("shake_info_test_smeanp.pkl", "wb") as f: pickle.dump(shake_info_test, f)
with open("train_history_smeanp.pkl", "wb") as f: pickle.dump(history_train, f)
with open("val_history_smeanp.pkl", "wb") as f: pickle.dump(history_val, f)
with open("val_acc_history_smeanp.pkl", "wb") as f: pickle.dump(history_val_acc, f)

print(f"Number of echelle diagrams in test set: {per_model_loss.shape[0]}")
print(f"Test Accuracy: {test_accuracy:.4f}")