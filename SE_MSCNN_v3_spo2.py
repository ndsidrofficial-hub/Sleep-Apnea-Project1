"""
SE-MSCNN v3 with SpO2 Fusion (PyTorch) — Target 95%+ Accuracy
==============================================================
Enhancements over v2 (89.38% test):
  1. 4th SpO2 branch for multi-modal ECG+SpO2 fusion
  2. Subject-aware train/val split (no patient leakage)
  3. Label smoothing + Focal Loss
  4. Mixup training
  5. Stronger data augmentation (varied noise, scaling, channel dropout)
  6. Gradient accumulation (effective batch 128)
  7. SWA (Stochastic Weight Averaging) in final epochs
  8. Reduced classifier dropout (0.3/0.35)
  9. Optimized ensemble weights + threshold
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import random
import math
import warnings
import argparse
import gc

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

# ======================== CONFIG ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPO2_CACHE = os.path.join(BASE_DIR, "spo2_data.pkl")
WEIGHTS_SAVE = os.path.join(BASE_DIR, "weights.v3_spo2.pt")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "SE_MSCNN_v3_predictions.csv")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 32
ACCUM_STEPS = 4       # Gradient accumulation → effective batch = 128
LR = 1e-3
SWA_START = 40         # Start SWA at this epoch
PATIENCE = 25

scale_fn = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


# ======================== DATASET ========================
class ApneaSpO2Dataset(Dataset):
    """Dataset with 3 ECG branches + 1 SpO2 branch."""

    def __init__(self, x1, x2, x3, spo2_1, spo2_2, spo2_3, y, augment=False):
        # ECG: (N, T, 2) → (N, 2, T) for Conv1d
        self.x1 = torch.from_numpy(x1.transpose(0, 2, 1)).float()    # (N, 2, 900)
        self.x2 = torch.from_numpy(x2.transpose(0, 2, 1)).float()    # (N, 2, 540)
        self.x3 = torch.from_numpy(x3.transpose(0, 2, 1)).float()    # (N, 2, 180)
        # SpO2: (N, T, 1) → (N, 1, T) for Conv1d
        self.spo2_1 = torch.from_numpy(spo2_1.transpose(0, 2, 1)).float()  # (N, 1, 900)
        self.spo2_2 = torch.from_numpy(spo2_2.transpose(0, 2, 1)).float()  # (N, 1, 540)
        self.spo2_3 = torch.from_numpy(spo2_3.transpose(0, 2, 1)).float()  # (N, 1, 180)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1, x2, x3 = self.x1[idx], self.x2[idx], self.x3[idx]
        s1, s2, s3 = self.spo2_1[idx], self.spo2_2[idx], self.spo2_3[idx]
        y = self.y[idx]

        if self.augment:
            # (a) Variable Gaussian noise
            noise_level = random.uniform(0.01, 0.05)
            x1 = x1 + torch.randn_like(x1) * noise_level
            x2 = x2 + torch.randn_like(x2) * noise_level
            x3 = x3 + torch.randn_like(x3) * noise_level
            # SpO2 noise (smaller)
            spo2_noise = random.uniform(0.005, 0.02)
            s1 = s1 + torch.randn_like(s1) * spo2_noise
            s2 = s2 + torch.randn_like(s2) * spo2_noise
            s3 = s3 + torch.randn_like(s3) * spo2_noise

            # (b) Random amplitude scaling
            scale = random.uniform(0.85, 1.15)
            x1, x2, x3 = x1 * scale, x2 * scale, x3 * scale

            # (c) Random temporal shift (larger range)
            shift = random.randint(-15, 15)
            if shift != 0:
                x1 = torch.roll(x1, shift, dims=-1)
                x2 = torch.roll(x2, shift, dims=-1)
                x3 = torch.roll(x3, shift, dims=-1)
                s1 = torch.roll(s1, shift, dims=-1)
                s2 = torch.roll(s2, shift, dims=-1)
                s3 = torch.roll(s3, shift, dims=-1)

            # (d) Random channel dropout (10% chance, ECG only)
            if random.random() < 0.1:
                ch = random.randint(0, 1)
                x1[ch] = 0
                x2[ch] = 0
                x3[ch] = 0

        return x1, x2, x3, s1, s2, s3, y


# ======================== MODEL COMPONENTS ========================
class ResidualConvBlock(nn.Module):
    """1D Residual Convolution Block with BatchNorm."""

    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1), nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ECGBranch(nn.Module):
    """ECG branch (2-channel input) — same architecture as v2."""

    def __init__(self, in_channels=2):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 32, 11, padding=5), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.block1 = ResidualConvBlock(32, 32, kernel=7)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.2)

        self.block2 = ResidualConvBlock(32, 64, kernel=7)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.2)

        self.block3 = ResidualConvBlock(64, 128, kernel=5)
        self.pool3 = nn.MaxPool1d(2)
        self.drop3 = nn.Dropout(0.3)

        self.block4 = ResidualConvBlock(128, 128, kernel=3)

    def forward(self, x):
        x = self.initial(x)
        x = self.drop1(self.pool1(self.block1(x)))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.drop3(self.pool3(self.block3(x)))
        x = self.block4(x)
        return x  # (N, 128, T')


class SpO2Branch(nn.Module):
    """Lightweight SpO2 branch (1-channel input, fewer params)."""

    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(1, 16, 11, padding=5), nn.BatchNorm1d(16), nn.ReLU()
        )
        self.block1 = ResidualConvBlock(16, 32, kernel=7)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.15)

        self.block2 = ResidualConvBlock(32, 64, kernel=5)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.2)

        self.block3 = ResidualConvBlock(64, 64, kernel=3)

    def forward(self, x):
        x = self.initial(x)
        x = self.drop1(self.pool1(self.block1(x)))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.block3(x)
        return x  # (N, 64, T')


class SEMSCNN_v3(nn.Module):
    """
    SE-MSCNN v3: 4-branch fusion model (3 ECG + 1 SpO2).
    
    Architecture:
      ECG branches (×3): 2ch → 128ch  (5-min, 3-min, 1-min)
      SpO2 branch (×1):  1ch → 64ch   (aggregated multi-scale)
      Total channels: 128×3 + 64 = 448
      SE attention on 448 channels
      Classifier: 448 → 256 → 128 → 2
    """

    def __init__(self):
        super().__init__()
        # ECG branches (same as v2)
        self.ecg_branch1 = ECGBranch(2)   # 5-min
        self.ecg_branch2 = ECGBranch(2)   # 3-min
        self.ecg_branch3 = ECGBranch(2)   # 1-min

        # SpO2 branch — processes concatenated multi-scale SpO2
        self.spo2_branch = SpO2Branch()

        # SE Attention on fused features
        total_ch = 128 * 3 + 64  # 448
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_excite = nn.Sequential(
            nn.Linear(total_ch, total_ch // 4),
            nn.ReLU(),
            nn.Linear(total_ch // 4, total_ch),
            nn.Sigmoid(),
        )

        # Classification head (reduced dropout from v2)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_ch),
            nn.Linear(total_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.3),      # was 0.4 in v2
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.35),     # was 0.5 in v2
        )
        self.output_layer = nn.Linear(128, 2)

    def forward(self, x1, x2, x3, spo2, return_features=False):
        """
        Args:
            x1: (N, 2, 900) ECG 5-min
            x2: (N, 2, 540) ECG 3-min
            x3: (N, 2, 180) ECG 1-min
            spo2: (N, 1, 900) SpO2 5-min (we use 5-min for the branch)
        """
        # ECG branches
        e1 = self.ecg_branch1(x1)   # (N, 128, T1)
        e2 = self.ecg_branch2(x2)   # (N, 128, T2)
        e3 = self.ecg_branch3(x3)   # (N, 128, T3)

        # SpO2 branch
        s = self.spo2_branch(spo2)   # (N, 64, Ts)

        # Align temporal dims via adaptive pooling
        min_t = min(e1.shape[2], e2.shape[2], e3.shape[2], s.shape[2])
        e1 = F.adaptive_avg_pool1d(e1, min_t)
        e2 = F.adaptive_avg_pool1d(e2, min_t)
        e3 = F.adaptive_avg_pool1d(e3, min_t)
        s  = F.adaptive_avg_pool1d(s, min_t)

        # Concatenate all branches: (N, 448, T)
        concat = torch.cat([e1, e2, e3, s], dim=1)

        # SE attention
        squeeze = self.se_squeeze(concat).squeeze(-1)        # (N, 448)
        excitation = self.se_excite(squeeze).unsqueeze(-1)    # (N, 448, 1)
        scaled = concat * excitation                          # (N, 448, T)

        # Global average pooling
        pooled = scaled.mean(dim=2)   # (N, 448)

        # Classifier
        features = self.classifier(pooled)   # (N, 128)
        logits = self.output_layer(features) # (N, 2)

        if return_features:
            return logits, features
        return logits


# ======================== FOCAL LOSS WITH LABEL SMOOTHING ========================
class FocalLossSmoothed(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = 2
        # Label smoothing
        smooth_targets = torch.full_like(logits, self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1)

        # Focal weighting
        probs = torch.exp(log_probs)
        pt = (probs * F.one_hot(targets, n_classes).float()).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            loss = alpha_weight * focal_weight * loss
        else:
            loss = focal_weight * loss

        return loss.mean()


# ======================== MIXUP ========================
def mixup_data(x1, x2, x3, s1, y, alpha=0.2):
    """Perform mixup on a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index]
    mixed_s1 = lam * s1 + (1 - lam) * s1[index]

    return mixed_x1, mixed_x2, mixed_x3, mixed_s1, y, y[index], lam


def mixup_criterion(criterion_fn, logits, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion_fn(logits, y_a) + (1 - lam) * criterion_fn(logits, y_b)


# ======================== TRAINING ========================
def train_epoch(model, loader, criterion, optimizer, use_mixup=True, accum_steps=4):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    for batch_idx, (x1, x2, x3, s1, s2, s3, y) in enumerate(loader):
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        x3 = x3.to(DEVICE)
        s1 = s1.to(DEVICE)
        y  = y.to(DEVICE)

        if use_mixup and random.random() < 0.5:
            x1, x2, x3, s1, y_a, y_b, lam = mixup_data(x1, x2, x3, s1, y)
            logits = model(x1, x2, x3, s1)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / accum_steps
            # For accuracy tracking, use original labels
            preds = logits.argmax(1)
            correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        else:
            logits = model(x1, x2, x3, s1)
            loss = criterion(logits, y) / accum_steps
            correct += (logits.argmax(1) == y).sum().item()

        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * y.size(0)
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_preds, all_true = [], [], []

    with torch.no_grad():
        for x1, x2, x3, s1, s2, s3, y in loader:
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            x3 = x3.to(DEVICE)
            s1 = s1.to(DEVICE)
            y  = y.to(DEVICE)

            logits = model(x1, x2, x3, s1)
            loss = criterion(logits, y)

            probs = F.softmax(logits, dim=1)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

            all_probs.append(probs[:, 1].cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(y.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_true),
    )


def extract_features(model, loader):
    """Extract deep features from the penultimate layer."""
    model.eval()
    all_features, all_true = [], []

    with torch.no_grad():
        for x1, x2, x3, s1, s2, s3, y in loader:
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            x3 = x3.to(DEVICE)
            s1 = s1.to(DEVICE)

            _, features = model(x1, x2, x3, s1, return_features=True)
            all_features.append(features.cpu().numpy())
            all_true.append(y.numpy())

    return np.concatenate(all_features), np.concatenate(all_true)


def find_best_threshold(probs, labels):
    """Search for optimal classification threshold."""
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr, best_f1


# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="SE-MSCNN v3 Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--smoke-test", action="store_true", help="Run 3-epoch smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 3

    print("=" * 60)
    print("SE-MSCNN v3 + SpO2 Fusion — Target 95%+ Accuracy")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Load Data ---
    print("\n[1/6] Loading SpO2-augmented data...")
    if not os.path.exists(SPO2_CACHE):
        print(f"ERROR: {SPO2_CACHE} not found.")
        print("Run: python generate_spo2_dataset.py")
        sys.exit(1)

    with open(SPO2_CACHE, "rb") as f:
        data = pickle.load(f)

    n_train = len(data['y_train'])
    n_val = len(data['y_val'])
    n_test = len(data['y_test'])
    print(f"  Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")
    print(f"  Train apnea rate: {np.mean(data['y_train']):.2%}")

    # --- Create Datasets ---
    print("\n[2/6] Creating datasets...")
    train_ds = ApneaSpO2Dataset(
        data['ecg_train1'], data['ecg_train2'], data['ecg_train3'],
        data['spo2_train1'], data['spo2_train2'], data['spo2_train3'],
        data['y_train'], augment=True
    )
    val_ds = ApneaSpO2Dataset(
        data['ecg_val1'], data['ecg_val2'], data['ecg_val3'],
        data['spo2_val1'], data['spo2_val2'], data['spo2_val3'],
        data['y_val'], augment=False
    )
    test_ds = ApneaSpO2Dataset(
        data['ecg_test1'], data['ecg_test2'], data['ecg_test3'],
        data['spo2_test1'], data['spo2_test2'], data['spo2_test3'],
        data['y_test'], augment=False
    )

    # Free raw data from memory
    groups_test = data.get('groups_test', None)
    y_test_np = data['y_test'].copy()
    y_val_np = data['y_val'].copy()
    del data
    gc.collect()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Class weights ---
    n_apnea = np.sum(train_ds.y.numpy() == 1)
    n_normal = np.sum(train_ds.y.numpy() == 0)
    total_samples = len(train_ds)
    class_weights = torch.tensor(
        [total_samples / (2 * max(n_normal, 1)), total_samples / (2 * max(n_apnea, 1))],
        dtype=torch.float32
    ).to(DEVICE)
    print(f"  Class weights: Normal={class_weights[0]:.3f}, Apnea={class_weights[1]:.3f}")

    # --- Build Model ---
    print("\n[3/6] Building SE-MSCNN v3 model...")
    model = SEMSCNN_v3().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    criterion = FocalLossSmoothed(gamma=2.0, alpha=class_weights, smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    # SWA setup
    try:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=5e-4)
        has_swa = True
        print("  SWA enabled (starts at epoch {})".format(SWA_START))
    except ImportError:
        has_swa = False
        print("  SWA not available (PyTorch version too old)")

    # --- Train ---
    print(f"\n[4/6] Training model ({args.epochs} epochs)...")
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Use mixup after epoch 5 (let model warm up first)
        use_mixup = epoch > 5

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            use_mixup=use_mixup, accum_steps=ACCUM_STEPS
        )
        val_loss, val_acc, val_probs, _, val_true = evaluate(model, val_loader, criterion)

        # SWA after SWA_START
        if has_swa and epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_SAVE)
            patience_counter = 0
            if epoch % 5 != 0:
                print(f"    → New best val_acc: {val_acc:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and not args.smoke_test:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

    # --- Apply SWA if available ---
    if has_swa and not args.smoke_test:
        print("\n  Applying SWA batch norm update...")
        update_bn(train_loader, swa_model, device=DEVICE)
        # Save SWA model
        swa_weights_path = WEIGHTS_SAVE.replace('.pt', '_swa.pt')
        torch.save(swa_model.module.state_dict(), swa_weights_path)
        print(f"  SWA weights saved to {swa_weights_path}")

        # Compare SWA vs best checkpoint
        swa_model.to(DEVICE)

    # Load best checkpoint for evaluation
    model.load_state_dict(torch.load(WEIGHTS_SAVE, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    # --- Evaluate CNN ---
    print("\n[5/6] Evaluating CNN model on test set...")
    _, cnn_acc, cnn_probs, cnn_preds, y_true = evaluate(model, test_loader, criterion)
    print(f"  CNN Test Accuracy: {cnn_acc:.4f} ({cnn_acc:.2%})")

    # --- XGBoost Ensemble ---
    print("\n[6/6] Training XGBoost ensemble on deep features...")
    try:
        from xgboost import XGBClassifier

        # Extract features
        feat_train, y_tr = extract_features(model, train_loader)
        feat_val, y_v = extract_features(model, val_loader)
        feat_test, y_te = extract_features(model, test_loader)

        scale_pos = max(n_normal, 1) / max(n_apnea, 1)
        xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=SEED,
            verbosity=0,
            n_jobs=1,  # RAM-safe
        )
        xgb_model.fit(feat_train, y_tr, eval_set=[(feat_val, y_v)], verbose=False)

        xgb_probs = xgb_model.predict_proba(feat_test)[:, 1]
        xgb_preds = xgb_model.predict(feat_test)
        xgb_acc = np.mean(xgb_preds == y_te)
        print(f"  XGBoost Test Accuracy: {xgb_acc:.4f} ({xgb_acc:.2%})")

        # Find optimal ensemble weight
        xgb_val_probs = xgb_model.predict_proba(feat_val)[:, 1]
        _, _, cnn_val_probs, _, val_true_labels = evaluate(model, val_loader, criterion)

        best_alpha, best_alpha_acc = 0.5, 0
        for alpha in np.arange(0.1, 0.95, 0.05):
            ens_probs = alpha * cnn_val_probs + (1 - alpha) * xgb_val_probs
            ens_preds = (ens_probs > 0.5).astype(int)
            acc = np.mean(ens_preds == val_true_labels)
            if acc > best_alpha_acc:
                best_alpha, best_alpha_acc = alpha, acc

        print(f"  Optimal ensemble weight: CNN={best_alpha:.2f}, XGB={1-best_alpha:.2f}")

        # Apply optimal weight to test
        ensemble_probs = best_alpha * cnn_probs + (1 - best_alpha) * xgb_probs

        # Find optimal threshold
        ens_val_probs = best_alpha * cnn_val_probs + (1 - best_alpha) * xgb_val_probs
        best_thr, _ = find_best_threshold(ens_val_probs, val_true_labels)
        print(f"  Optimal threshold: {best_thr:.2f}")

        ensemble_preds = (ensemble_probs >= best_thr).astype(int)
        has_xgb = True

    except Exception as e:
        print(f"  XGBoost failed: {e}. Using CNN-only results.")
        ensemble_probs = cnn_probs
        ensemble_preds = cnn_preds
        xgb_probs = cnn_probs
        xgb_preds = cnn_preds
        has_xgb = False

    # --- Final Metrics ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS — SE-MSCNN v3 + SpO2")
    print("=" * 60)

    results = [("CNN Only", cnn_preds, cnn_probs)]
    if has_xgb:
        results.append(("XGBoost Only", xgb_preds, xgb_probs))
        results.append(("Ensemble (Optimized)", ensemble_preds, ensemble_probs))

    for name, preds, probs in results:
        C = confusion_matrix(y_true, preds, labels=[1, 0])
        TP, TN = C[0, 0], C[1, 1]
        FP, FN = C[1, 0], C[0, 1]
        acc = (TP + TN) / (TP + TN + FP + FN)
        sn = TP / (TP + FN) if (TP + FN) > 0 else 0
        sp = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = f1_score(y_true, preds)
        try:
            auc_val = roc_auc_score(y_true, probs)
        except Exception:
            auc_val = 0.0

        print(f"\n{name}:")
        print(f"  Accuracy:    {acc:.4f} ({acc:.2%})")
        print(f"  Sensitivity: {sn:.4f} ({sn:.2%})")
        print(f"  Specificity: {sp:.4f} ({sp:.2%})")
        print(f"  F1-score:    {f1:.4f}")
        print(f"  AUC-ROC:     {auc_val:.4f}")

    # Save predictions
    output = pd.DataFrame({
        "y_true": y_true,
        "y_pred": ensemble_preds,
        "y_score": ensemble_probs,
        "y_score_cnn": cnn_probs,
    })
    if has_xgb:
        output["y_score_xgb"] = xgb_probs
    if groups_test is not None:
        output["subject"] = groups_test
    output.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nPredictions saved to {PREDICTIONS_CSV}")
    print(f"Model weights saved to {WEIGHTS_SAVE}")

    # Final verdict
    ens_acc = np.mean(ensemble_preds == y_true)
    print(f"\n{'=' * 60}")
    print(f"BEST TEST ACCURACY: {ens_acc:.2%}")
    if ens_acc >= 0.95:
        print("🎯 TARGET 95% ACHIEVED!")
    else:
        print(f"Gap to 95% target: {0.95 - ens_acc:.2%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
