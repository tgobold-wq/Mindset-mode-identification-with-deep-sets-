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

FAKE_CLASS = 3
DROPPED_CLASS = -1
DROP_PROB = 0.2
SHAKE_STD = 0.05
MAX_FAKE_FRACTION = 0.2
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
X_val, X_test, y_val, y_test   = train_test_split(X_temp,  y_temp,  test_size=0.5, random_state=42)

class PointNet(nn.Module):
    def __init__(self, num_classes=4, point_dimension=2, aggregator="mean"):
        super(PointNet, self).__init__()
        a = 3
        nfeat = 64
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
        x = self.conv_1_a(x_freq) + self.conv_1_b(x_modnu)
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
        global_features = self.base_pointnet(x).unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat((x, global_features), dim=2)
        x = F.relu(self.fc_1(x))
        return F.log_softmax(self.fc_2(x), dim=2)

loss_fn = nn.NLLLoss(reduction="none", ignore_index=DROPPED_CLASS)
model = SegmentationPointNet(num_classes=4)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def inject_fake_frequencies(x, y, max_fake_fraction=MAX_FAKE_FRACTION, num_fake_points=10):
    valid_mask = y != DROPPED_CLASS
    x_real = x[valid_mask]
    if x_real.shape[0] < 2:
        return x, y, None
    freqs = x_real[:, 0]
    f_min, f_max = freqs.min().item(), freqs.max().item()
    mod_vals = x_real[:, 1]
    mod_min, mod_max = mod_vals.min().item(), mod_vals.max().item()
    n_real = x_real.shape[0]
    n_fake = min(num_fake_points, int(n_real * max_fake_fraction))
    if n_fake <= 0:
        return x, y, None
    f_fake = torch.rand(n_fake, device=x.device) * (f_max - f_min) + f_min
    mod_fake = torch.rand(n_fake, device=x.device) * (mod_max - mod_min) + mod_min
    fake_points = torch.stack([f_fake, mod_fake], dim=1)
    x_aug = torch.cat([x, fake_points], dim=0)
    y_aug = torch.cat([y, torch.full((n_fake,), FAKE_CLASS, dtype=torch.long, device=x.device)], dim=0)
    return x_aug, y_aug, fake_points

history_train, history_test = [], []
history_train_acc, history_test_acc = [], []
fake_data_test = {}

Nepoch = 150
nbatches = 100
batch_size = max(1, X_train.shape[0] // nbatches)

for epoch in range(Nepoch):
    model.train()
    for i in range(nbatches):
        optimizer.zero_grad()
        i0, i1 = i * batch_size, min((i + 1) * batch_size, X_train.shape[0])
        if i0 >= i1:
            continue
        x_batch, y_batch = X_train[i0:i1], y_train[i0:i1]
        x_aug_list, y_aug_list = [], []
        for j in range(x_batch.shape[0]):
            xj, yj, fake_j = inject_fake_frequencies(x_batch[j], y_batch[j])
            x_aug_list.append(xj)
            y_aug_list.append(yj)
            if fake_j is not None:
                fake_data_test[f"train_model_{i0+j}"] = fake_j.cpu().numpy()
        max_len = max(xj.shape[0] for xj in x_aug_list)
        x_padded = torch.zeros(len(x_aug_list), max_len, 2, dtype=torch.float32)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long)
        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0], :] = xj
            y_padded[j, :yj.shape[0]] = yj
        out = model(x_padded)
        out_nll = out.permute(0, 2, 1)
        loss_mat = loss_fn(out_nll, y_padded)
        loss = loss_mat.mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_out = model(X_test).permute(0, 2, 1)
        test_loss_mat = loss_fn(test_out, y_test)
        val_loss = test_loss_mat.mean()
        train_preds = out.argmax(dim=2)
        valid_mask_train = y_padded != DROPPED_CLASS
        train_acc = (train_preds[valid_mask_train] == y_padded[valid_mask_train]).sum().item() / max(1, valid_mask_train.sum().item())
        test_preds = test_out.argmax(dim=1)
        valid_mask_test = y_test != DROPPED_CLASS
        test_acc = (test_preds[valid_mask_test] == y_test[valid_mask_test]).sum().item() / max(1, valid_mask_test.sum().item())
    history_train.append(float(loss))
    history_test.append(float(val_loss))
    history_train_acc.append(train_acc)
    history_test_acc.append(test_acc)
    print(f"Epoch {epoch:3d}: Train Loss={float(loss):.4f}, Test Loss={float(val_loss):.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

torch.save(model.state_dict(), "segmentation_pointnet_fmeanpstorefake.pth")

with open("train_history_fmeanpstorefake.pkl", "wb") as f: pickle.dump(history_train, f)
with open("test_history_fmeanpstorefake.pkl", "wb") as f: pickle.dump(history_test, f)
with open("train_acc_history_fmeanpstorefake.pkl", "wb") as f: pickle.dump(history_train_acc, f)
with open("test_acc_history_fmeanpstorefake.pkl", "wb") as f: pickle.dump(history_test_acc, f)

model.eval()
with torch.no_grad():
    test_out = model(X_test).permute(0, 2, 1)
    per_point_loss = loss_fn(test_out, y_test).cpu().numpy()
    per_model_loss = per_point_loss.mean(axis=1)
    cost_test = per_point_loss.mean()
    test_predictions = test_out.argmax(dim=1).cpu().numpy()
    test_targets = y_test.cpu().numpy()
    test_probs = F.softmax(test_out, dim=1).cpu().numpy()
    valid_mask = test_targets != DROPPED_CLASS
    test_accuracy = (test_predictions[valid_mask] == test_targets[valid_mask]).sum() / max(1, valid_mask.sum())
    for i in range(X_test.shape[0]):
        _, _, fake_i = inject_fake_frequencies(X_test[i], y_test[i])
        if fake_i is not None:
            fake_data_test[f"test_model_{i}"] = fake_i.cpu().numpy()

with open("output_logits_fmeanpstorefake.pkl", "wb") as f: pickle.dump(test_out.cpu().numpy(), f)
with open("output_probs_fmeanpstorefake.pkl", "wb") as f: pickle.dump(test_probs, f)
with open("cost_fmeanpstorefake.pkl", "wb") as f: pickle.dump(cost_test, f)
with open("test_targets_fmeanpstorefake.pkl", "wb") as f: pickle.dump(test_targets, f)
with open("test_predictions_fmeanpstorefake.pkl", "wb") as f: pickle.dump(test_predictions, f)
with open("per_point_losses_test_fmeanpstorefake.pkl", "wb") as f: pickle.dump(per_point_loss, f)
with open("average_loss_per_model_fmeanpstorefake.pkl", "wb") as f: pickle.dump(per_model_loss, f)
with open("test_accuracy_fmeanpstorefake.pkl", "wb") as f: pickle.dump(test_accuracy, f)
with open("fake_data_test_fmeanpstorefake.pkl", "wb") as f: pickle.dump(fake_data_test, f)

print(f"Number of echelle diagrams in test set: {per_model_loss.shape[0]}")
print(f"Test Accuracy (excluding DROPPED_CLASS): {test_accuracy:.4f}")