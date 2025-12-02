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

grid = pd.read_pickle("outputs_in_instab_strip.pkl")
l_values = [0, 1, 2]
l_columns = {l: [col for col in grid.columns if col.endswith(f"_l_{l}")] for l in l_values}

X_frequencies = np.hstack([grid[l_columns[l]].values for l in l_values])          # [num_models, num_points_total]
delta_nu_values = grid["delta_nu"].values[:, np.newaxis]                           # [num_models, 1]
mod_frequencies = X_frequencies % delta_nu_values                                  # same shape
X = np.stack((X_frequencies, mod_frequencies), axis=-1)                            # [num_models, num_points, 2]
y = np.hstack([np.full_like(grid[l_columns[l]].values, l) for l in l_values])      # [num_models, num_points]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_temp,  y_temp,  test_size=0.5, random_state=42)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

class PointNet(nn.Module):
    def __init__(self, num_classes=4, point_dimension=2, aggregator="mean"):
        super(PointNet, self).__init__()
        a = 3
        nfeat = 64 
        self.conv_1_a = nn.Conv1d(1, nfeat, kernel_size=a, padding=2)
        self.conv_1_b = nn.Conv1d(1, nfeat, kernel_size=a, padding=2)
        self.conv_2 = nn.Conv1d(nfeat, nfeat, kernel_size=a, padding=2)
        self.bn_1 = nn.BatchNorm1d(nfeat)
        self.bn_2 = nn.BatchNorm1d(nfeat)
        self.aggregator = aggregator

    def forward(self, x, plot=False):
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
        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator}")
        x = x.view(batch_size, -1)
        return x

class SegmentationPointNet(nn.Module):
    def __init__(self, num_classes=4, point_dimension=2, aggregator="mean"):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = PointNet(
            num_classes=num_classes, point_dimension=point_dimension, aggregator=aggregator)
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
    keep_mask = torch.rand(x.shape[0], device=x.device) > dropout_prob
    x_new = x.clone()
    y_new = y.clone()

    y_new[~keep_mask] = DROPPED_CLASS  # drop class
    x_new[keep_mask] += torch.randn_like(x_new[keep_mask]) * noise_std
    return x_new, y_new

def inject_fake_frequencies(x, y, max_fake_fraction=MAX_FAKE_FRACTION):
    valid_mask = y != DROPPED_CLASS
    x_real = x[valid_mask]
    num_real = x_real.shape[0]
    num_fake = int(num_real * max_fake_fraction)

    if num_real < 2 or num_fake == 0:
        return x, y

    sorted_freqs = x_real[:, 0].sort()[0]
    dnu = (sorted_freqs[1:] - sorted_freqs[:-1]).mean().item()

    f_min = x_real[:, 0].min().item()
    f_max = x_real[:, 0].max().item()

    fake_freqs = torch.zeros(num_fake, 2, device=x.device)
    fake_freqs[:, 0] = torch.linspace(f_min, f_max, steps=num_fake).to(x.device)
    fake_freqs[:, 0] += torch.randn(num_fake, device=x.device) * (0.1 * dnu)
    fake_freqs[:, 1] = fake_freqs[:, 0] % dnu

    x_aug = torch.cat([x, fake_freqs], dim=0)
    y_aug = torch.cat([y, torch.full((num_fake,), FAKE_CLASS, dtype=torch.long, device=x.device)], dim=0)
    return x_aug, y_aug

# Loss/Optimizer
# use reduction="none" so we can save per-point losses later; mean it during training/eval
loss_fn = nn.NLLLoss(reduction="none", ignore_index=DROPPED_CLASS)
model = SegmentationPointNet(num_classes=4)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

history_train, history_test = [], []
history_train_acc, history_test_acc = [], []

Nepoch = 150
nbatches = 100
batch_size = max(1, X_train.shape[0] // nbatches)

# -----------------------------
# Training loop
# -----------------------------
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
            xj, yj = apply_dropper_and_shaker(x_batch[j], y_batch[j])
            xj, yj = inject_fake_frequencies(xj, yj)
            x_aug_list.append(xj)
            y_aug_list.append(yj)

        max_len = max(xj.shape[0] for xj in x_aug_list)
        x_padded = torch.zeros(len(x_aug_list), max_len, 2, dtype=torch.float32)
        y_padded = torch.full((len(y_aug_list), max_len), DROPPED_CLASS, dtype=torch.long)
        for j, (xj, yj) in enumerate(zip(x_aug_list, y_aug_list)):
            x_padded[j, :xj.shape[0], :] = xj
            y_padded[j, :yj.shape[0]]    = yj

        out = model(x_padded)
        out_nll = out.permute(0, 2, 1)
        loss_mat = loss_fn(out_nll, y_padded)
        loss = loss_mat.mean()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_out_nll = test_out.permute(0, 2, 1)
        test_loss_mat = loss_fn(test_out_nll, y_test)
        val_loss = test_loss_mat.mean()

        # Train acc (last batch)
        train_preds = out.argmax(dim=2)
        valid_mask_train = y_padded != DROPPED_CLASS
        train_correct = (train_preds[valid_mask_train] == y_padded[valid_mask_train]).sum().item()
        train_total = valid_mask_train.sum().item()
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Test acc
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


# Save model and training history
torch.save(model.state_dict(), "segmentation_pointnet_dsfmeanp.pth")

# Histories
with open("train_history_dsfmeanp.pkl", "wb") as f:
    pickle.dump(history_train, f)
with open("test_history_dsfmeanp.pkl", "wb") as f:
    pickle.dump(history_test, f)
with open("train_acc_history_dsfmeanp.pkl", "wb") as f:
    pickle.dump(history_train_acc, f)
with open("test_acc_history_dsfmeanp.pkl", "wb") as f:
    pickle.dump(history_test_acc, f)

# Detailed test outputs
model.eval()
with torch.no_grad():
    test_out = model(X_test).permute(0, 2, 1)               # [B, C, N]
    test_losses = loss_fn(test_out, y_test)                 # [B, N] (reduction="none")
    per_point_loss = test_losses.cpu().numpy()              # [B, N]
    per_model_loss = test_losses.mean(dim=1).cpu().numpy()  # [B]
    cost_test = test_losses.mean().item()
    test_predictions = test_out.argmax(dim=1).cpu().numpy() # [B, N]
    test_targets = y_test.cpu().numpy()                     # [B, N]

    valid_mask = test_targets != DROPPED_CLASS
    correct = (test_predictions[valid_mask] == test_targets[valid_mask]).sum()
    total = valid_mask.sum()
    test_accuracy = float(correct) / float(total) if total > 0 else 0.0

# Save outputs
with open("output_logits_dsfmeanp.pkl", "wb") as f:
    pickle.dump(test_out.cpu(), f)
with open("cost_dsfmeanp.pkl", "wb") as f:
    pickle.dump(cost_test, f)
with open("test_targets_dsfmeanp.pkl", "wb") as f:
    pickle.dump(test_targets, f)
with open("test_predictions_dsfmeanp.pkl", "wb") as f:
    pickle.dump(test_predictions, f)
with open("per_point_losses_test_dsfmeanp.pkl", "wb") as f:
    pickle.dump(per_point_loss, f)
with open("average_loss_per_model_dsfmeanp.pkl", "wb") as f:
    pickle.dump(per_model_loss, f)
with open("test_accuracy_dsfmeanp.pkl", "wb") as f:
    pickle.dump(test_accuracy, f)

print(f"Number of echelle diagrams in test set: {per_model_loss.shape[0]}")
print(f"Test Accuracy (excluding DROPPED_CLASS): {test_accuracy:.4f}")

