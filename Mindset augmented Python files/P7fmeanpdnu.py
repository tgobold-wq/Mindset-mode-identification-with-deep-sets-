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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAKE_CLASS = 3
DROPPED_CLASS = -1
MAX_FAKE_FRACTION = 0.2
Nepoch = 150
nbatches = 100
learning_rate = 0.0005

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
dn_tensor = torch.tensor(delta_nu_values, dtype=torch.float32)

X_train, X_temp, y_train, y_temp, dn_train, dn_temp = train_test_split(
    X_tensor, y_tensor, dn_tensor, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test, dn_val, dn_test = train_test_split(
    X_temp, y_temp, dn_temp, test_size=0.5, random_state=42
)

X_train = X_train.to(device)
y_train = y_train.to(device)
dn_train = dn_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

size = X_train.shape[0]
inpt = X_train[:size]
y_inpt = y_train[:size]
dn_inpt = dn_train[:size]

class DeltaNuCorrection(nn.Module):
    def __init__(self, n_stars, delta_nu_hard):
        super().__init__()
        self.register_buffer("delta_nu_hard", delta_nu_hard.clone())
        self.delta_nu_corr = nn.Parameter(torch.zeros_like(delta_nu_hard))
        self.eps = 1e-3

    def forward(self, frequencies, star_indices):
        delta = (self.delta_nu_hard[star_indices] + self.delta_nu_corr[star_indices]).clamp_min(self.eps)
        mod_freq = torch.remainder(frequencies, delta)  # differentiable w.r.t. frequencies; Δν grads are zero
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
                num = (x * m).sum(dim=-1, keepdim=True)
                den = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
                x = num / den
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

model = SegmentationPointNet(n_stars=inpt.shape[0], delta_nu_hard=dn_inpt).to(device)
loss_fn = nn.NLLLoss(reduction="mean", ignore_index=DROPPED_CLASS)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def inject_fake_frequencies(x, y, max_fake_fraction=MAX_FAKE_FRACTION, num_fake_points=10):
    valid_mask = y != DROPPED_CLASS
    x_real = x[valid_mask]
    if x_real.shape[0] < 2:
        return x, y, None
    freqs = x_real[:, 0]
    mods = x_real[:, 1]
    f_min, f_max = freqs.min().item(), freqs.max().item()
    mod_min, mod_max = mods.min().item(), mods.max().item()
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

batch_size = max(1, inpt.shape[0] // nbatches)
inpt_test = inpt
target_test = y_inpt

for epoch in range(Nepoch):
    model.train()
    train_acc_epoch = []

    for i in range(nbatches):
        optimizer.zero_grad()
        i0 = i * batch_size
        i1 = min((i + 1) * batch_size, inpt.shape[0])
        if i0 >= i1:
            continue

        x_batch = inpt[i0:i1]
        y_batch = y_train[i0:i1]

        x_aug_list, y_aug_list = [], []
        for j in range(x_batch.shape[0]):
            xj, yj, _ = inject_fake_frequencies(x_batch[j], y_batch[j])
            x_aug_list.append(xj)
            y_aug_list.append(yj)

        max_len = max(x.shape[0] for x in x_aug_list)
        x_padded = torch.zeros(len(x_aug_list), max_len, 2, device=inpt.device)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long, device=inpt.device)

        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0], :] = xj
            y_padded[j, :yj.shape[0]] = yj

        star_indices = torch.arange(x_padded.size(0), device=x_padded.device)
        seg_output = model(x_padded, star_indices=star_indices)
        seg_output_perm = seg_output.permute(0, 2, 1)
        loss = loss_fn(seg_output_perm, y_padded)
        loss.backward()
        optimizer.step()

        pred_classes = seg_output.argmax(dim=2)
        mask_valid = y_padded != DROPPED_CLASS
        if mask_valid.any():
            batch_acc = (pred_classes[mask_valid] == y_padded[mask_valid]).float().mean().item()
        else:
            batch_acc = 0.0
        train_acc_epoch.append(batch_acc)

    model.eval()
    with torch.no_grad():
        star_indices_test = torch.arange(inpt_test.size(0), device=inpt_test.device)
        seg_out_test = model(inpt_test, star_indices=star_indices_test)
        seg_out_test_perm = seg_out_test.permute(0, 2, 1)
        val_loss = loss_fn(seg_out_test_perm, target_test)

        train_preds = seg_output.argmax(dim=2)
        valid_mask_train = y_padded != DROPPED_CLASS
        train_acc = (train_preds[valid_mask_train] == y_padded[valid_mask_train]).float().mean().item() if valid_mask_train.any() else 0.0

        test_preds = seg_out_test.argmax(dim=2)
        valid_mask_test = target_test != DROPPED_CLASS
        test_acc = (test_preds[valid_mask_test] == target_test[valid_mask_test]).float().mean().item() if valid_mask_test.any() else 0.0

    history_train.append(float(loss))
    history_test.append(float(val_loss))
    history_train_acc.append(train_acc)
    history_test_acc.append(test_acc)

    print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Test Loss = {val_loss:.4f}, "
          f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

# --- Save model and metrics ---
torch.save(model.state_dict(), "segmentation_pointnet_fmeanpdnu.pth")
with open("train_history_fmeanpdnu.pkl", "wb") as f: pickle.dump(history_train, f)
with open("test_history_fmeanpdnu.pkl", "wb") as f: pickle.dump(history_test, f)
with open("train_acc_history_fmeanpdnu.pkl", "wb") as f: pickle.dump(history_train_acc, f)
with open("test_acc_history_fmeanpdnu.pkl", "wb") as f: pickle.dump(history_test_acc, f)

model.eval()
with torch.no_grad():
    star_indices_test = torch.arange(X_test.shape[0], device=X_test.device)
    test_out = model(X_test, star_indices=star_indices_test)
    test_out_perm = test_out.permute(0, 2, 1)  # (B, C, N)

    loss_fn_no_reduce = nn.NLLLoss(ignore_index=DROPPED_CLASS, reduction="none")
    per_point_loss = loss_fn_no_reduce(test_out_perm, y_test)  # (B, N) with zeros at DROPPED_CLASS

    valid_mask_test = (y_test != DROPPED_CLASS).float()  # (B, N)
    denom_per_model = valid_mask_test.sum(dim=1).clamp_min(1.0)  # (B,)
    per_model_loss = (per_point_loss.sum(dim=1) / denom_per_model).cpu().numpy()

    cost_test = (per_point_loss.sum() / valid_mask_test.sum().clamp_min(1.0)).item()

    test_predictions = test_out.argmax(dim=2).cpu().numpy()
    test_targets = y_test.cpu().numpy()

    valid_mask_np = test_targets != DROPPED_CLASS
    correct = (test_predictions[valid_mask_np] == test_targets[valid_mask_np]).sum()
    total = valid_mask_np.sum()
    test_accuracy = float(correct) / float(total) if total > 0 else 0.0

    fake_data_test = {}
    for i in range(X_test.shape[0]):
        _, _, fake_i = inject_fake_frequencies(X_test[i], y_test[i])
        if fake_i is not None:
            fake_data_test[f"test_model_{i}"] = fake_i.cpu().numpy()

torch.save(model.state_dict(), "segmentation_pointnet_fmeanpdnu.pth")
with open("output_logits_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(test_out.cpu(), f)
with open("cost_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(cost_test, f)
with open("test_targets_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(test_targets, f)
with open("test_predictions_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(test_predictions, f)
with open("per_point_losses_test_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(per_point_loss.cpu().numpy(), f)
with open("average_loss_per_model_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(per_model_loss, f)
with open("test_accuracy_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(test_accuracy, f)
with open("fake_data_test_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(fake_data_test, f)
with open("train_history_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(history_train, f)
with open("test_history_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(history_test, f)
with open("train_acc_history_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(history_train_acc, f)
with open("test_acc_history_fmeanpdnu.pkl", "wb") as f:
    pickle.dump(history_test_acc, f)


print(f"Number of echelle diagrams in test set: {per_model_loss.shape[0]}")
print(f"Test Accuracy (excluding DROPPED_CLASS): {test_accuracy:.4f}")