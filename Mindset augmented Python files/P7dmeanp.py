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

DROPPED_CLASS = -1
DROP_PROB = 0.2
REAL_CLASSES = [0, 1, 2]

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

class PointNet(nn.Module):
    def __init__(self, num_classes=4, point_dimension=2, aggregator="mean"):
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
    def __init__(self, num_classes=4, point_dimension=2, aggregator="mean"):
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

def apply_dropper(x, y, dropout_prob=DROP_PROB):
    keep_mask = torch.rand(x.shape[0], device=x.device) > dropout_prob
    dropped_mask = ~keep_mask
    y_new = y.clone()
    y_new[dropped_mask] = DROPPED_CLASS
    dropped_indices = torch.nonzero(dropped_mask, as_tuple=False).flatten()
    return x, y_new, dropped_indices

loss_fn = nn.NLLLoss(reduction="none", ignore_index=DROPPED_CLASS)
model = SegmentationPointNet(num_classes=4)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

history_train, history_test = [], []
history_test_acc = []

Nepoch = 150
nbatches = 100
batch_size = max(1, X_train.shape[0] // nbatches)

for epoch in range(Nepoch):
    model.train()
    epoch_loss = 0
    for i in range(nbatches):
        optimizer.zero_grad()
        i0 = i * batch_size
        i1 = min((i + 1) * batch_size, X_train.shape[0])
        if i0 >= i1:
            continue
        x_batch = X_train[i0:i1]
        y_batch = y_train[i0:i1]
        x_aug_list, y_aug_list = [], []
        for j in range(x_batch.shape[0]):
            xj, yj, _ = apply_dropper(x_batch[j], y_batch[j])
            x_aug_list.append(xj)
            y_aug_list.append(yj)
        max_len = max(xj.shape[0] for xj in x_aug_list)
        x_padded = torch.zeros(len(x_aug_list), max_len, 2, dtype=torch.float32)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long)
        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0], :] = xj
            y_padded[j, :yj.shape[0]] = yj
        out = model(x_padded)
        loss = loss_fn(out.permute(0, 2, 1), y_padded).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    history_train.append(epoch_loss / nbatches)
    model.eval()
    with torch.no_grad():
        test_out = model(X_val)
        test_out_nll = test_out.permute(0, 2, 1)
        val_loss = loss_fn(test_out_nll, y_val).mean().item()
        test_preds = test_out.argmax(dim=2)
        valid_mask_test = y_val != DROPPED_CLASS
        test_correct = (test_preds[valid_mask_test] == y_val[valid_mask_test]).sum().item()
        test_total = valid_mask_test.sum().item()
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        history_test.append(val_loss)
        history_test_acc.append(test_acc)
    print(f"Epoch {epoch+1}/{Nepoch}: Train Loss = {history_train[-1]:.4f}, Val Loss = {history_test[-1]:.4f}, Val Acc = {history_test_acc[-1]:.4f}")

model.eval()
dsf_extra_info_test = {}

with torch.no_grad():
    test_out = model(X_test).permute(0, 2, 1)
    test_losses = loss_fn(test_out, y_test)
    per_point_loss = test_losses.cpu().numpy()
    per_model_loss = test_losses.mean(dim=1).cpu().numpy()
    cost_test = test_losses.mean().item()
    test_predictions = test_out.argmax(dim=1).cpu().numpy()
    test_targets = y_test.cpu().numpy()
    test_probs = F.softmax(test_out, dim=1).cpu().numpy()
    valid_mask = test_targets != DROPPED_CLASS
    correct = (test_predictions[valid_mask] == test_targets[valid_mask]).sum()
    total = valid_mask.sum()
    test_accuracy = float(correct) / float(total) if total > 0 else 0.0
    for i in range(X_test.shape[0]):
        x_i = X_test[i]
        y_i = y_test[i]
        _, _, dropped_i = apply_dropper(x_i, y_i)
        dsf_extra_info_test[f"test_model_{i}"] = {"dropped_indices": dropped_i.cpu().numpy()}

torch.save(model.state_dict(), "segmentation_pointnet_dmeanp.pth")
with open("output_logits_dmeanp.pkl", "wb") as f: pickle.dump(test_out.cpu().numpy(), f)
with open("output_probs_dmeanp.pkl", "wb") as f: pickle.dump(test_probs, f)
with open("cost_dmeanp.pkl", "wb") as f: pickle.dump(cost_test, f)
with open("test_targets_dmeanp.pkl", "wb") as f: pickle.dump(test_targets, f)
with open("test_predictions_dmeanp.pkl", "wb") as f: pickle.dump(test_predictions, f)
with open("per_point_losses_test_dmeanp.pkl", "wb") as f: pickle.dump(per_point_loss, f)
with open("average_loss_per_model_dmeanp.pkl", "wb") as f: pickle.dump(per_model_loss, f)
with open("test_accuracy_dmeanp.pkl", "wb") as f: pickle.dump(test_accuracy, f)
with open("dsf_extra_info_test_dmeanp.pkl", "wb") as f: pickle.dump(dsf_extra_info_test, f)
with open("train_history_dmeanp.pkl", "wb") as f: pickle.dump(history_train, f)
with open("val_history_dmeanp.pkl", "wb") as f: pickle.dump(history_test, f)
with open("val_acc_history_dmeanp.pkl", "wb") as f: pickle.dump(history_test_acc, f)

print(f"Number of echelle diagrams in test set: {per_model_loss.shape[0]}")
print(f"Test Accuracy (excluding DROPPED_CLASS): {test_accuracy:.4f}")