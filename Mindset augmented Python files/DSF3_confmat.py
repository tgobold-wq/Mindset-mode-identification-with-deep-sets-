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
CHECKPOINT_PATH = "segmentation_pointnet_DSF1cm.pth"

FAKE_CLASS = 3
DROPPED_CLASS = -1
DROP_PROB = 0.2
SHAKE_STD = 0.1
MAX_FAKE_FRACTION = 0.4
NUM_FAKE_POINTS_PER_SAMPLE = 10
REAL_CLASSES = [0, 1, 2]

LEARN_RATE = 5e-4
N_EPOCH = 150
NBATCHES = 100

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

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

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
        else:
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

def apply_dropper_and_shaker(x, y, dropout_prob=DROP_PROB, noise_std=SHAKE_STD):
    if x.numel() == 0:
        return x, y, torch.tensor([], dtype=torch.long), torch.zeros_like(x)
    keep_mask = torch.rand(x.shape[0]) > dropout_prob
    dropped_mask = ~keep_mask
    x_new = x.clone()
    y_new = y.clone()
    y_new[dropped_mask] = DROPPED_CLASS
    dropped_indices = torch.nonzero(dropped_mask, as_tuple=False).flatten()
    shake_amount = torch.zeros_like(x_new)
    if keep_mask.any():
        shake_amount[keep_mask] = torch.randn_like(x_new[keep_mask]) * noise_std
        x_new[keep_mask] += shake_amount[keep_mask]
    return x_new, y_new, dropped_indices, shake_amount

def inject_fake_frequencies(x, y, max_fake_fraction=MAX_FAKE_FRACTION, num_fake_points=NUM_FAKE_POINTS_PER_SAMPLE):
    valid_mask = y != DROPPED_CLASS
    x_real = x[valid_mask]
    if x_real.shape[0] < 2:
        return x, y, None, 0
    freqs = x_real[:, 0]
    f_min, f_max = float(freqs.min().item()), float(freqs.max().item())
    mod_vals = x_real[:, 1]
    mod_min, mod_max = float(mod_vals.min().item()), float(mod_vals.max().item())
    n_real = x_real.shape[0]
    n_fake = min(num_fake_points, max(0, int(n_real * max_fake_fraction)))
    if n_fake <= 0:
        return x, y, None, 0
    f_fake = torch.rand(n_fake) * (f_max - f_min) + f_min
    mod_fake = torch.rand(n_fake) * (mod_max - mod_min) + mod_min
    fake_points = torch.stack([f_fake, mod_fake], dim=1)
    x_aug = torch.cat([x, fake_points], dim=0)
    y_aug = torch.cat([y, torch.full((n_fake,), FAKE_CLASS, dtype=torch.long)], dim=0)
    return x_aug, y_aug, fake_points, n_fake

# -----------------------------
# Training
# -----------------------------
loss_fn = nn.NLLLoss(reduction="none", ignore_index=DROPPED_CLASS)
model = SegmentationPointNet(num_classes=4)
optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

history_train, history_val, history_val_acc = [], [], []

batch_size = max(1, X_train.shape[0] // NBATCHES)

for epoch in range(N_EPOCH):
    model.train()
    epoch_loss = 0
    for i in range(NBATCHES):
        optimizer.zero_grad()
        i0 = i * batch_size
        i1 = min((i + 1) * batch_size, X_train.shape[0])
        if i0 >= i1: continue
        x_batch = X_train[i0:i1]
        y_batch = y_train[i0:i1]
        x_aug_list, y_aug_list = [], []
        for j in range(x_batch.shape[0]):
            xj, yj, _, _ = apply_dropper_and_shaker(x_batch[j], y_batch[j])
            xj, yj, _, _ = inject_fake_frequencies(xj, yj)
            x_aug_list.append(xj); y_aug_list.append(yj)
        max_len = max(xj.shape[0] for xj in x_aug_list)
        x_padded = torch.zeros(len(x_aug_list), max_len, 2, dtype=torch.float32)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long)
        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0], :] = xj
            y_padded[j, :yj.shape[0]] = yj
        out = model(x_padded)
        loss = loss_fn(out.permute(0, 2, 1), y_padded).mean()
        loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    history_train.append(epoch_loss / NBATCHES)

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = loss_fn(val_out.permute(0, 2, 1), y_val).mean().item()
        val_preds = val_out.argmax(dim=2)
        valid_mask_val = y_val != DROPPED_CLASS
        val_acc = (val_preds[valid_mask_val] == y_val[valid_mask_val]).float().mean().item()
        history_val.append(val_loss)
        history_val_acc.append(val_acc)

    print(f"Epoch {epoch+1}/{N_EPOCH} Train Loss={history_train[-1]:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")


torch.save(model.state_dict(), "segmentation_pointnet_DSF3cm.pth")
with open("train_history_DSF3cm.pkl", "wb") as f: pickle.dump(history_train, f)
with open("val_history_DSF3cm.pkl", "wb") as f: pickle.dump(history_val, f)
with open("val_acc_history_DSF3cm.pkl", "wb") as f: pickle.dump(history_val_acc, f)

all_preds, all_targets, all_probs, per_point_loss, avg_loss_per_sample = [], [], [], [], []

model.eval()
with torch.no_grad():
    for i in range(X_test.shape[0]):
        x_i = X_test[i]; y_i = y_test[i]
        x_aug, y_aug, _, _ = apply_dropper_and_shaker(x_i, y_i)
        x_aug, y_aug, _, _ = inject_fake_frequencies(x_aug, y_aug)
        logits = model(x_aug.unsqueeze(0))
        probs = F.softmax(logits, dim=2)[0]
        loss_points = loss_fn(logits.permute(0,2,1), y_aug.unsqueeze(0)).squeeze(0)
        all_preds.append(logits.argmax(dim=2).squeeze(0).cpu().numpy())
        all_targets.append(y_aug.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        per_point_loss.append(loss_points.cpu().numpy())
        avg_loss_per_sample.append(loss_points.mean().item())

pred_flat = np.concatenate(all_preds)
true_flat = np.concatenate(all_targets)
probs_flat = np.concatenate(all_probs)
per_point_loss_flat = np.concatenate(per_point_loss)
avg_loss_per_sample = np.array(avg_loss_per_sample)

valid_mask = true_flat != DROPPED_CLASS
test_predictions = pred_flat[valid_mask]
test_targets = true_flat[valid_mask]
test_probs = probs_flat[valid_mask]

with open("test_predictions_DSF3cm.pkl", "wb") as f: pickle.dump(test_predictions, f)
with open("test_targets_DSF3cm.pkl", "wb") as f: pickle.dump(test_targets, f)
with open("output_probs_DSF3cm.pkl", "wb") as f: pickle.dump(test_probs, f)
with open("per_point_losses_test_DSF3cm.pkl", "wb") as f: pickle.dump(per_point_loss_flat, f)
with open("average_loss_per_model_DSF3cm.pkl", "wb") as f: pickle.dump(avg_loss_per_sample, f)

print(f"Saved model and all DSF1cm outputs: {len(test_targets)} test points ready.")