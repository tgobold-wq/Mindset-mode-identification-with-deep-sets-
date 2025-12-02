import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAKE_CLASS = 3
DROPPED_CLASS = -1
MAX_FAKE_FRACTION = 0.2
Nepoch = 50
nbatches = 20
learning_rate = 0.001
N_candidates = 15
delta_min, delta_max = 1.6037, 8.3909

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

delta_candidates = torch.linspace(delta_min, delta_max, N_candidates, device=device)

class DeltaNuCorrection(nn.Module):
    def __init__(self, n_stars, delta_nu_hard):
        super().__init__()
        self.register_buffer("delta_nu_hard", delta_nu_hard.clone())
        self.delta_nu_corr = nn.Parameter(torch.zeros_like(delta_nu_hard))
        self.eps = 1e-3
    def forward(self, frequencies, star_indices):
        delta = (self.delta_nu_hard[star_indices] + self.delta_nu_corr[star_indices]).clamp_min(self.eps)
        return torch.remainder(frequencies, delta)

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
        B, N, _ = x.shape
        x_freq = x[:, :, 0].unsqueeze(1)
        x_modnu = x[:, :, 1].unsqueeze(1)
        x_freq = F.pad(x_freq, (1,1), mode='replicate')
        x_modnu = F.pad(x_modnu, (1,1), mode='circular')
        x = self.conv_1_a(x_freq) + self.conv_1_b(x_modnu)
        x = F.relu(self.bn_1(x))
        x = F.relu(self.bn_2(self.conv_2(x)))
        if self.aggregator=="mean":
            x = F.avg_pool1d(x, kernel_size=N)
        else:
            x = F.max_pool1d(x, kernel_size=N)
        return x.view(B,-1)

class SegmentationPointNet(nn.Module):
    def __init__(self, n_stars, delta_nu_hard, num_classes=4, point_dimension=2, aggregator="mean"):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = PointNet(num_classes=num_classes, point_dimension=point_dimension, aggregator=aggregator)
        nhidden = 64
        nfeat = 64
        self.fc_1 = nn.Linear(nfeat + 2, nhidden)
        self.fc_2 = nn.Linear(nhidden, num_classes)
        self.delta_nu_module = DeltaNuCorrection(n_stars, delta_nu_hard)
    def forward(self, x, star_indices=None):
        g = self.base_pointnet(x)
        g = g.unsqueeze(1).repeat(1, x.shape[1], 1)
        z = torch.cat([x, g], dim=2)
        z = F.relu(self.fc_1(z))
        logits = self.fc_2(z)
        return F.log_softmax(logits, dim=2)

def inject_fake_frequencies(x, y, max_fake_fraction=MAX_FAKE_FRACTION, num_fake_points=10):
    valid_mask = (y != DROPPED_CLASS)
    x_real = x[valid_mask]
    if x_real.shape[0]<2:
        return x, y, None
    freqs = x_real[:,0]
    mods = x_real[:,1]
    n_real = x_real.shape[0]
    n_fake = min(num_fake_points,int(n_real*max_fake_fraction))
    if n_fake<=0:
        return x, y, None
    f_fake = torch.rand(n_fake, device=x.device)*(freqs.max()-freqs.min()) + freqs.min()
    m_fake = torch.rand(n_fake, device=x.device)*(mods.max()-mods.min()) + mods.min()
    fake_points = torch.stack([f_fake, m_fake], dim=1)
    x_aug = torch.cat([x, fake_points], dim=0)
    y_aug = torch.cat([y, torch.full((n_fake,),FAKE_CLASS,dtype=torch.long,device=x.device)], dim=0)
    return x_aug, y_aug, fake_points

model = SegmentationPointNet(n_stars=X_train.shape[0], delta_nu_hard=dn_train).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn_none = nn.NLLLoss(ignore_index=DROPPED_CLASS, reduction="none")

history_train, history_test, history_train_acc, history_test_acc = [], [], [], []
fake_data_test_dict = {}
batch_size = max(1, X_train.shape[0] // nbatches)

for epoch in range(Nepoch):
    model.train()
    perm = torch.randperm(X_train.shape[0], device=device)
    batch_losses = []
    for bstart in range(0, X_train.shape[0], batch_size):
        batch_idx = perm[bstart:bstart+batch_size].tolist()
        all_min_losses = []
        all_best_dnu = []
        for idx in batch_idx:
            x_star = X_train[idx]
            y_star = y_train[idx]
            x_aug, y_aug, _ = inject_fake_frequencies(x_star, y_star)
            freqs = x_aug[:,0]
            if freqs.numel() == 0:
                continue
            mods_candidates = freqs.unsqueeze(0) % delta_candidates.unsqueeze(1)
            freqs_rep = freqs.unsqueeze(0).repeat(N_candidates, 1)
            x_candidates = torch.stack([freqs_rep, mods_candidates], dim=2)
            logits_c = model(x_candidates)
            logits_perm = logits_c.permute(0,2,1)
            y_expand = y_aug.unsqueeze(0).repeat(N_candidates, 1)
            candidate_losses = loss_fn_none(logits_perm, y_expand)
            mask = (y_expand != DROPPED_CLASS)
            denom = mask.sum(dim=1).clamp_min(1)
            candidate_mean = (candidate_losses * mask.float()).sum(dim=1) / denom.float()
            best_idx = candidate_mean.argmin()
            all_min_losses.append(candidate_mean[best_idx])
            all_best_dnu.append(delta_candidates[best_idx].item())
        if not all_min_losses:
            continue
        batch_loss = torch.stack(all_min_losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batch_losses.append(batch_loss.item())
        history_train.append(batch_loss.item())
    model.eval()
    with torch.no_grad():
        test_logits = []
        val_loss_total = 0.0
        fake_data_test_dict = {}
        for i in range(X_test.shape[0]):
            x_star = X_test[i]
            y_star = y_test[i]
            freqs = x_star[:,0]
            mods_candidates = freqs.unsqueeze(0) % delta_candidates.unsqueeze(1)
            freqs_rep = freqs.unsqueeze(0).repeat(N_candidates,1)
            x_candidates = torch.stack([freqs_rep, mods_candidates], dim=2)
            logits_c = model(x_candidates)
            logits_perm = logits_c.permute(0,2,1)
            y_expand = y_star.unsqueeze(0).repeat(N_candidates,1)
            candidate_losses = loss_fn_none(logits_perm, y_expand)
            mask = (y_expand != DROPPED_CLASS)
            denom = mask.sum(dim=1).clamp_min(1.0)
            candidate_mean = (candidate_losses * mask.float()).sum(dim=1) / denom.float()
            best_idx = candidate_mean.argmin()
            test_logits.append(logits_c[best_idx])
            _, _, fake_i = inject_fake_frequencies(x_star, y_star)
            if fake_i is not None:
                fake_data_test_dict[f"test_model_{i}"] = fake_i.cpu().numpy()
        test_out = torch.stack(test_logits, dim=0)
        test_out_perm = test_out.permute(0,2,1)
        per_point_loss = loss_fn_none(test_out_perm, y_test)
        valid_mask_test = (y_test != DROPPED_CLASS).float()
        denom_per_model = valid_mask_test.sum(dim=1).clamp_min(1.0).cpu().numpy()
        per_model_loss = (per_point_loss.sum(dim=1).cpu().numpy() / denom_per_model).astype(float)
        cost_test = (per_point_loss.sum() / valid_mask_test.sum().clamp_min(1.0)).item()
        test_predictions = test_out.argmax(dim=2).cpu()
        test_targets = y_test.cpu()
        mask_np = test_targets.numpy() != DROPPED_CLASS
        correct = (test_predictions.numpy()[mask_np] == test_targets.numpy()[mask_np]).sum()
        total = mask_np.sum()
        test_acc = float(correct) / float(total) if total > 0 else 0.0
    history_test.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
    history_test_acc.append(test_acc)

torch.save(model.state_dict(), "segmentation_pointnet_fmeanpdnulast.pth")
with open("output_logits_fmeanpdnulast.pkl","wb") as f: pickle.dump(test_out.cpu(), f)
with open("cost_fmeanpdnulast.pkl","wb") as f: pickle.dump(cost_test, f)
with open("test_targets_fmeanpdnulast.pkl","wb") as f: pickle.dump(test_targets.cpu(), f)
with open("test_predictions_fmeanpdnulast.pkl","wb") as f: pickle.dump(test_predictions.cpu(), f)
with open("per_point_losses_test_fmeanpdnulast.pkl","wb") as f: pickle.dump(per_point_loss.cpu().numpy(), f)
with open("average_loss_per_model_fmeanpdnulast.pkl","wb") as f: pickle.dump(per_model_loss, f)
with open("test_accuracy_fmeanpdnulast.pkl","wb") as f: pickle.dump(test_acc, f)
with open("fake_data_test_fmeanpdnulast.pkl","wb") as f: pickle.dump(fake_data_test_dict, f)
with open("train_history_fmeanpdnulast.pkl","wb") as f: pickle.dump(history_train, f)
with open("test_history_fmeanpdnulast.pkl","wb") as f: pickle.dump(history_test, f)
with open("train_acc_history_fmeanpdnulast.pkl","wb") as f: pickle.dump(history_train_acc, f)
with open("test_acc_history_fmeanpdnulast.pkl","wb") as f: pickle.dump(history_test_acc, f)
with open("delta_nu_candidates.pkl","wb") as f: pickle.dump(delta_candidates.cpu().numpy(), f)