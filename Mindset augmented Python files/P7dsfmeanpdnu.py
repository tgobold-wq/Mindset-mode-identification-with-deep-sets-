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

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

class DeltaNuCorrection(nn.Module):
    def __init__(self, n_stars, delta_nu_hard):
        super().__init__()
        self.register_buffer("delta_nu_hard", delta_nu_hard.clone())
        self.delta_nu_corr = nn.Parameter(torch.zeros_like(delta_nu_hard))
        self.eps = 1e-3

    def forward(self, frequencies, star_indices):
        delta = (self.delta_nu_hard[star_indices] + self.delta_nu_corr[star_indices]).clamp_min(self.eps)
        delta = delta.unsqueeze(1)
        mod_freq = frequencies % delta
        return mod_freq

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

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        x_freq = x[:, :, 0].unsqueeze(1)
        x_modnu = x[:, :, 1].unsqueeze(1)
        x = self.conv_1_a(x_freq) + self.conv_1_b(x_modnu)
        x = F.relu(self.bn_1(x))
        x = F.relu(self.bn_2(self.conv_2(x)))
        if self.aggregator == "max":
            if mask is not None:
                m = (mask.unsqueeze(1) > 0).float()
                x = x.masked_fill(m == 0, -1e9)
            x = x.amax(dim=-1, keepdim=True)
        elif self.aggregator == "mean":
            if mask is None:
                x = x.mean(dim=-1, keepdim=True)
            else:
                m = (mask.unsqueeze(1) > 0).float()
                x = (x * m).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
        return x.view(B, -1)

class SegmentationPointNet(nn.Module):
    def __init__(self, n_stars, delta_nu_hard, num_classes=4, point_dimension=2, aggregator="mean"):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = PointNet(num_classes=num_classes, point_dimension=point_dimension, aggregator=aggregator)
        nhidden = 64
        nfeat = 64
        self.fc_1 = nn.Linear(nfeat + 2, nhidden)
        self.fc_2 = nn.Linear(nhidden, num_classes)
        self.delta_nu_module = DeltaNuCorrection(n_stars, delta_nu_hard)

    def forward(self, x, star_indices, mask=None):
        freq = x[:, :, 0]
        modf = self.delta_nu_module(freq, star_indices)
        x2 = torch.stack((freq, modf), dim=2)
        g = self.base_pointnet(x2, mask=mask)
        g = g.unsqueeze(1).expand(-1, x2.shape[1], -1)
        z = torch.cat([x2, g], dim=2)
        z = F.relu(self.fc_1(z))
        return F.log_softmax(self.fc_2(z), dim=2)
                    
def apply_dropper_and_shaker(x, y, dropout_prob=DROP_PROB, noise_std=SHAKE_STD):
    keep_mask = torch.rand(x.shape[0], device=x.device) > dropout_prob
    x_new = x.clone()
    y_new = y.clone()
    y_new[~keep_mask] = DROPPED_CLASS
    x_new[keep_mask] += torch.randn_like(x_new[keep_mask]) * noise_std
    return x_new, y_new
    
def inject_fake_frequencies(x, y, max_fake_fraction=MAX_FAKE_FRACTION, num_fake_points=10):

    valid_mask = y != DROPPED_CLASS
    x_real = x[valid_mask]
    y_real = y[valid_mask]

    if x_real.shape[0] < 2:
        return x, y, None

    # Estimate Δν and observed mod range
    freqs = x_real[:, 0]
    freqs_sorted, _ = freqs.sort()
    dnu = (freqs_sorted[1:] - freqs_sorted[:-1]).mean().item()

    f_min = freqs.min().item()
    f_max = freqs.max().item()

    mod_vals = x_real[:, 1]
    mod_min = mod_vals.min().item()
    mod_max = mod_vals.max().item()

    # Decide number of fake points
    n_real = x_real.shape[0]
    n_fake = min(num_fake_points, int(n_real * max_fake_fraction))
    if n_fake <= 0:
        return x, y, None

    # Sample uniformly within observed frequency and mod ranges
    f_fake = torch.rand(n_fake, device=x.device) * (f_max - f_min) + f_min
    mod_fake = torch.rand(n_fake, device=x.device) * (mod_max - mod_min) + mod_min

    fake_points = torch.stack([f_fake, mod_fake], dim=1)

    # Append to data
    x_aug = torch.cat([x, fake_points], dim=0)
    y_aug = torch.cat([y, torch.full((n_fake,), FAKE_CLASS, dtype=torch.long, device=x.device)], dim=0)
    return x_aug, y_aug, fake_points
    
n_stars = X_train.shape[0]  # Number of stars in training set
delta_nu_hard = torch.tensor(grid["delta_nu"].values[:n_stars], dtype=torch.float32)

model = SegmentationPointNet(
    n_stars=n_stars,
    delta_nu_hard=delta_nu_hard,
    num_classes=4
)

loss_fn = nn.NLLLoss(reduction="none", ignore_index=DROPPED_CLASS)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
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
        i0 = i * batch_size
        i1 = min((i + 1) * batch_size, X_train.shape[0])
        if i0 >= i1:
            continue
        x_batch = X_train[i0:i1]
        y_batch = y_train[i0:i1]

        x_aug_list, y_aug_list = [], []
        for j in range(x_batch.shape[0]):
            # Apply drop/shake
            xj, yj = apply_dropper_and_shaker(x_batch[j], y_batch[j], dropout_prob=DROP_PROB, noise_std=SHAKE_STD)
            # Inject fake points
            xj, yj, fake_j = inject_fake_frequencies(xj, yj, max_fake_fraction=MAX_FAKE_FRACTION)
            x_aug_list.append(xj)
            y_aug_list.append(yj)
            if fake_j is not None:
                fake_data_test[f"train_model_{i0+j}"] = fake_j.cpu().numpy()

        # Pad sequences
        max_len = max(xj.shape[0] for xj in x_aug_list)
        x_padded = torch.zeros(len(x_aug_list), max_len, 2, dtype=torch.float32)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long)

        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0], :] = xj
            y_padded[j, :yj.shape[0]] = yj

        star_indices = torch.arange(x_padded.size(0))
        out = model(x_padded, star_indices=star_indices)
        out_nll = out.permute(0, 2, 1)
        loss_mat = loss_fn(out_nll, y_padded)
        loss = loss_mat.mean()
        loss.backward()
        optimizer.step()

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        star_indices_test = torch.arange(X_test.size(0))
        test_out = model(X_test, star_indices=star_indices_test)
        test_out_nll = test_out.permute(0, 2, 1)
        test_loss_mat = loss_fn(test_out_nll, y_test)
        val_loss = test_loss_mat.mean()

        # Training accuracy
        train_preds = out.argmax(dim=2)
        valid_mask_train = y_padded != DROPPED_CLASS
        train_correct = (train_preds[valid_mask_train] == y_padded[valid_mask_train]).sum().item()
        train_total = valid_mask_train.sum().item()
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Test accuracy
        test_preds = test_out.argmax(dim=2)
        valid_mask_test = y_test != DROPPED_CLASS
        test_correct = (test_preds[valid_mask_test] == y_test[valid_mask_test]).sum().item()
        test_total = valid_mask_test.sum().item()
        test_acc = test_correct / test_total if test_total > 0 else 0.0

    history_train.append(float(loss))
    history_test.append(float(val_loss))
    history_train_acc.append(train_acc)
    history_test_acc.append(test_acc)

    print(f"Epoch {epoch:3d}: Train Loss = {float(loss):.4f}, "
          f"Test Loss = {float(val_loss):.4f}, "
          f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

# --- Save model and metrics ---
torch.save(model.state_dict(), "segmentation_pointnet_dsfmeanpdnu.pth")
with open("train_history_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(history_train, f)
with open("test_history_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(history_test, f)
with open("train_acc_history_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(history_train_acc, f)
with open("test_acc_history_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(history_test_acc, f)

# --- Evaluate test set and save outputs ---
model.eval()
with torch.no_grad():
    star_indices_test = torch.arange(X_test.size(0))
    test_out = model(X_test, star_indices=star_indices_test)
    test_out_perm = test_out.permute(0, 2, 1)
    test_losses = loss_fn(test_out_perm, y_test)
    per_point_loss = test_losses.cpu().numpy()
    per_model_loss = test_losses.mean(dim=1).cpu().numpy()
    cost_test = test_losses.mean().item()
    test_predictions = test_out.argmax(dim=2).cpu().numpy()
    test_targets = y_test.cpu().numpy()

    valid_mask = test_targets != DROPPED_CLASS
    correct = (test_predictions[valid_mask] == test_targets[valid_mask]).sum()
    total = valid_mask.sum()
    test_accuracy = float(correct) / float(total) if total > 0 else 0.0

    fake_data_test_final = {}
    for i in range(X_test.shape[0]):
        _, _, fake_i = inject_fake_frequencies(X_test[i], y_test[i], max_fake_fraction=MAX_FAKE_FRACTION)
        if fake_i is not None:
            fake_data_test_final[f"test_model_{i}"] = fake_i.cpu().numpy()

with open("output_logits_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(test_out.cpu(), f)
with open("cost_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(cost_test, f)
with open("test_targets_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(test_targets, f)
with open("test_predictions_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(test_predictions, f)
with open("per_point_losses_test_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(per_point_loss, f)
with open("average_loss_per_model_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(per_model_loss, f)
with open("test_accuracy_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(test_accuracy, f)
with open("fake_data_test_dsfmeanpdnu.pkl", "wb") as f: pickle.dump(fake_data_test_final, f)

print(f"Number of echelle diagrams in test set: {per_model_loss.shape[0]}")
print(f"Test Accuracy (excluding DROPPED_CLASS): {test_accuracy:.4f}")