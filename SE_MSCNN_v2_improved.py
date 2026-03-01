"""
SE-MSCNN v2 Improved (PyTorch) — Target 95% Accuracy
=====================================================
Enhancements over baseline (89.85%):
  1. Deeper Conv1D branches with BatchNorm & residual connections
  2. Focal Loss for class imbalance
  3. Cosine Annealing LR with warm restarts
  4. Data augmentation (Gaussian noise + temporal jitter)
  5. Smaller batch size (32) for better generalization
  6. XGBoost on deep features + ensemble with CNN
  7. Early stopping with patience=20
"""

import os
import pickle
import numpy as np
import pandas as pd
import random
import math
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

# ======================== CONFIG ========================
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "apnea-ecg-database-1.0.0"
)
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
PICKLE_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "preprocessed_data.pkl"
)
WEIGHTS_SAVE = "weights.v2_improved.pt"
PREDICTIONS_CSV = "SE_MSCNN_v2_predictions.csv"

IR = 3
BEFORE = 2
AFTER = 2
TM = np.arange(0, (BEFORE + 1 + AFTER) * 60, step=1 / float(IR))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

scale_fn = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)


# ======================== DATA LOADING ========================
def load_data():
    """Load and preprocess the apnea-ecg dataset using wfdb."""
    import wfdb
    from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
    import biosppy.signals.tools as st
    from scipy.interpolate import splev, splrep

    fs = 100
    sample = fs * 60

    train_names = [
        "a01",
        "a02",
        "a03",
        "a04",
        "a05",
        "a06",
        "a07",
        "a08",
        "a09",
        "a10",
        "a11",
        "a12",
        "a13",
        "a14",
        "a15",
        "a16",
        "a17",
        "a18",
        "a19",
        "a20",
        "b01",
        "b02",
        "b03",
        "b04",
        "b05",
        "c01",
        "c02",
        "c03",
        "c04",
        "c05",
        "c06",
        "c07",
        "c08",
        "c09",
        "c10",
    ]

    test_names = [
        "x01",
        "x02",
        "x03",
        "x04",
        "x05",
        "x06",
        "x07",
        "x08",
        "x09",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
        "x16",
        "x17",
        "x18",
        "x19",
        "x20",
        "x21",
        "x22",
        "x23",
        "x24",
        "x25",
        "x26",
        "x27",
        "x28",
        "x29",
        "x30",
        "x31",
        "x32",
        "x33",
        "x34",
        "x35",
    ]

    # Load answers for test set
    answers = {}
    answers_path = os.path.join(DATASET_DIR, "event-2-answers.txt")
    if not os.path.exists(answers_path):
        answers_path = os.path.join(BASE_DIR, "event-2-answers.txt")
    if os.path.exists(answers_path):
        with open(answers_path, "r") as f:
            for answer in f.read().split("\n\n"):
                answer = answer.strip()
                if len(answer) >= 3:
                    answers[answer[:3]] = list("".join(answer.split()[2::2]))

    def process_subject(name, labels, data_dir):
        """Extract RRI and amplitude features for one subject."""
        X1, X2, X3, Y, G = [], [], [], [], []

        try:
            signals = wfdb.rdrecord(
                os.path.join(data_dir, name), channels=[0]
            ).p_signal[:, 0]
        except Exception as e:
            print(f"  Could not load {name}: {e}")
            return X1, X2, X3, Y, G

        for j in range(len(labels)):
            if j < BEFORE or (j + 1 + AFTER) > len(signals) / float(sample):
                continue

            signal = signals[int((j - BEFORE) * sample) : int((j + 1 + AFTER) * sample)]

            try:
                signal, _, _ = st.filter_signal(
                    signal,
                    ftype="FIR",
                    band="bandpass",
                    order=int(0.3 * fs),
                    frequency=[3, 45],
                    sampling_rate=fs,
                )

                (rpeaks,) = hamilton_segmenter(signal, sampling_rate=fs)
                (rpeaks,) = correct_rpeaks(
                    signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1
                )

                if len(rpeaks) < 2:
                    continue

                # RRI features
                rri = np.diff(rpeaks) / float(fs)
                rri_tm_vals = rpeaks[:-1] / float(fs)
                ampl = signal[rpeaks]
                ampl_tm_vals = rpeaks / float(fs)

                # Spline interpolation
                rri_interp = splev(TM, splrep(rri_tm_vals, scale_fn(rri), k=3), ext=1)
                ampl_interp = splev(
                    TM, splrep(ampl_tm_vals, scale_fn(ampl), k=3), ext=1
                )

                X1.append([rri_interp, ampl_interp])  # 5-min: (2, 900)
                X2.append(
                    [rri_interp[180:720], ampl_interp[180:720]]
                )  # 3-min: (2, 540)
                X3.append(
                    [rri_interp[360:540], ampl_interp[360:540]]
                )  # 1-min: (2, 180)
                Y.append(0.0 if labels[j] == "N" else 1.0)
                G.append(name)
            except Exception:
                continue

        return X1, X2, X3, Y, G

    # Process training data
    print("Processing training subjects...")
    o_train1, o_train2, o_train3 = [], [], []
    y_train, groups_train = [], []

    for name in train_names:
        try:
            labels = wfdb.rdann(os.path.join(BASE_DIR, name), extension="apn").symbol
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        x1, x2, x3, y, g = process_subject(name, labels, BASE_DIR)
        o_train1.extend(x1)
        o_train2.extend(x2)
        o_train3.extend(x3)
        y_train.extend(y)
        groups_train.extend(g)
        print(f"  {name}: {len(y)} segments")

    print(f"Total training segments: {len(y_train)}")

    # Process test data
    print("\nProcessing test subjects...")
    o_test1, o_test2, o_test3 = [], [], []
    y_test, groups_test = [], []

    for name in test_names:
        try:
            if name in answers:
                labels = answers[name]
            else:
                labels = wfdb.rdann(
                    os.path.join(BASE_DIR, name), extension="apn"
                ).symbol
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        x1, x2, x3, y, g = process_subject(name, labels, BASE_DIR)
        o_test1.extend(x1)
        o_test2.extend(x2)
        o_test3.extend(x3)
        y_test.extend(y)
        groups_test.extend(g)
        print(f"  {name}: {len(y)} segments")

    print(f"Total test segments: {len(y_test)}")

    # Convert to numpy arrays: (N, 2, T) -> (N, T, 2) for PyTorch conv1d (we'll use channels_last then permute)
    def to_arrays(x_list):
        return np.array(x_list, dtype="float32").transpose((0, 2, 1))  # (N, T, 2)

    x_all1 = to_arrays(o_train1)
    x_all2 = to_arrays(o_train2)
    x_all3 = to_arrays(o_train3)
    y_all = np.array(y_train, dtype="float32")

    # 80/20 train-val split
    indices = list(range(len(y_all)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    data = {
        "x_train1": x_all1[train_idx],
        "x_train2": x_all2[train_idx],
        "x_train3": x_all3[train_idx],
        "y_train": y_all[train_idx],
        "x_val1": x_all1[val_idx],
        "x_val2": x_all2[val_idx],
        "x_val3": x_all3[val_idx],
        "y_val": y_all[val_idx],
        "x_test1": to_arrays(o_test1),
        "x_test2": to_arrays(o_test2),
        "x_test3": to_arrays(o_test3),
        "y_test": np.array(y_test, dtype="float32"),
        "groups_test": groups_test,
    }

    print(
        f"\nTrain: {data['x_train1'].shape[0]}, Val: {data['x_val1'].shape[0]}, Test: {data['x_test1'].shape[0]}"
    )
    print(f"Train apnea rate: {np.mean(data['y_train']):.2%}")
    print(f"Test apnea rate: {np.mean(data['y_test']):.2%}")

    return data


# ======================== PYTORCH DATASET ========================
class ApneaDataset(Dataset):
    def __init__(self, x1, x2, x3, y, augment=False):
        # Store as (N, C, T) for PyTorch Conv1d
        self.x1 = torch.from_numpy(x1.transpose(0, 2, 1))  # (N, 2, 900)
        self.x2 = torch.from_numpy(x2.transpose(0, 2, 1))  # (N, 2, 540)
        self.x3 = torch.from_numpy(x3.transpose(0, 2, 1))  # (N, 2, 180)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1, x2, x3, y = self.x1[idx], self.x2[idx], self.x3[idx], self.y[idx]

        if self.augment:
            # Gaussian noise
            x1 = x1 + torch.randn_like(x1) * 0.02
            x2 = x2 + torch.randn_like(x2) * 0.02
            x3 = x3 + torch.randn_like(x3) * 0.02

            # Random temporal shift
            shift = random.randint(-5, 5)
            if shift != 0:
                x1 = torch.roll(x1, shift, dims=-1)
                x2 = torch.roll(x2, shift, dims=-1)
                x3 = torch.roll(x3, shift, dims=-1)

        return x1, x2, x3, y


# ======================== MODEL ========================
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


class Branch(nn.Module):
    """One scale branch with deep residual blocks."""

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

        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.block2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.block3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.block4(x)
        return x


class ImprovedSEMSCNN(nn.Module):
    """SE-MSCNN v2 with deeper residual branches and improved SE attention."""

    def __init__(self):
        super().__init__()
        self.branch1 = Branch(2)  # 5-min
        self.branch2 = Branch(2)  # 3-min
        self.branch3 = Branch(2)  # 1-min

        # SE Attention
        total_ch = 128 * 3  # 384
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_excite = nn.Sequential(
            nn.Linear(total_ch, total_ch // 4),
            nn.ReLU(),
            nn.Linear(total_ch // 4, total_ch),
            nn.Sigmoid(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_ch),
            nn.Linear(total_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.output_layer = nn.Linear(128, 2)

    def forward(self, x1, x2, x3, return_features=False):
        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        b3 = self.branch3(x3)

        # Align temporal dims before concat via adaptive pooling
        min_t = min(b1.shape[2], b2.shape[2], b3.shape[2])
        b1 = F.adaptive_avg_pool1d(b1, min_t)
        b2 = F.adaptive_avg_pool1d(b2, min_t)
        b3 = F.adaptive_avg_pool1d(b3, min_t)

        concat = torch.cat([b1, b2, b3], dim=1)  # (N, 384, T)

        # SE attention
        squeeze = self.se_squeeze(concat).squeeze(-1)  # (N, 384)
        excitation = self.se_excite(squeeze).unsqueeze(-1)  # (N, 384, 1)
        scaled = concat * excitation  # (N, 384, T)

        # Global average pooling
        pooled = scaled.mean(dim=2)  # (N, 384)

        # Classifier
        features = self.classifier(pooled)  # (N, 128)
        logits = self.output_layer(features)  # (N, 2)

        if return_features:
            return logits, features
        return logits


# ======================== FOCAL LOSS ========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of per-class weights

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=2).float()

        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        loss = focal_weight * ce
        return loss.mean()


# ======================== TRAINING ========================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x1, x2, x3, y in loader:
        x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x1, x2, x3)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_preds, all_true = [], [], []

    with torch.no_grad():
        for x1, x2, x3, y in loader:
            x1, x2, x3, y = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), y.to(DEVICE)
            logits = model(x1, x2, x3)
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
        for x1, x2, x3, y in loader:
            x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)
            _, features = model(x1, x2, x3, return_features=True)
            all_features.append(features.cpu().numpy())
            all_true.append(y.numpy())

    return np.concatenate(all_features), np.concatenate(all_true)


# ======================== MAIN ========================
if __name__ == "__main__":
    print("=" * 60)
    print("SE-MSCNN v2 Improved (PyTorch) — Target 95% Accuracy")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Load Data (with pickle cache) ---
    print("\n[1/5] Loading and preprocessing data...")
    if os.path.exists(PICKLE_CACHE):
        print(f"  Loading cached data from {PICKLE_CACHE}...")
        with open(PICKLE_CACHE, "rb") as f:
            data = pickle.load(f)
        print(
            f"  Train: {data['x_train1'].shape[0]}, Val: {data['x_val1'].shape[0]}, Test: {data['x_test1'].shape[0]}"
        )
    else:
        data = load_data()
        print(f"  Saving preprocessed data to {PICKLE_CACHE}...")
        with open(PICKLE_CACHE, "wb") as f:
            pickle.dump(data, f, protocol=4)
        print(f"  Cached! Next run will load instantly.")

    # Datasets and loaders
    train_ds = ApneaDataset(
        data["x_train1"],
        data["x_train2"],
        data["x_train3"],
        data["y_train"],
        augment=True,
    )
    val_ds = ApneaDataset(
        data["x_val1"], data["x_val2"], data["x_val3"], data["y_val"], augment=False
    )
    test_ds = ApneaDataset(
        data["x_test1"], data["x_test2"], data["x_test3"], data["y_test"], augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Class weights for focal loss
    n_apnea = np.sum(data["y_train"] == 1)
    n_normal = np.sum(data["y_train"] == 0)
    total = len(data["y_train"])
    class_weights = torch.tensor(
        [total / (2 * n_normal), total / (2 * n_apnea)], dtype=torch.float32
    ).to(DEVICE)
    print(f"Class weights: Normal={class_weights[0]:.3f}, Apnea={class_weights[1]:.3f}")

    # --- Build Model ---
    print("\n[2/5] Building improved SE-MSCNN v2 model...")
    model = ImprovedSEMSCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    # --- Train ---
    print("\n[3/5] Training model...")
    best_val_acc = 0
    patience_counter = 0
    patience = 20

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_SAVE)
            patience_counter = 0
            if epoch % 5 != 0:
                print(f"  -> New best val_acc: {val_acc:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load(WEIGHTS_SAVE, weights_only=True))
    model.to(DEVICE)

    # --- Evaluate CNN ---
    print("\n[4/5] Evaluating CNN model...")
    _, cnn_acc, cnn_probs, cnn_preds, y_true = evaluate(model, test_loader, criterion)
    print(f"CNN Test Accuracy: {cnn_acc:.4f}")

    # --- XGBoost Ensemble ---
    print("\n[5/5] Training XGBoost ensemble...")
    try:
        from xgboost import XGBClassifier

        # No augmentation for feature extraction
        train_noaug_ds = ApneaDataset(
            data["x_train1"],
            data["x_train2"],
            data["x_train3"],
            data["y_train"],
            augment=False,
        )
        train_noaug_loader = DataLoader(
            train_noaug_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        feat_train, y_tr = extract_features(model, train_noaug_loader)
        feat_val, y_v = extract_features(model, val_loader)
        feat_test, y_te = extract_features(model, test_loader)

        scale_pos = n_normal / max(n_apnea, 1)
        xgb = XGBClassifier(
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
        )

        xgb.fit(feat_train, y_tr, eval_set=[(feat_val, y_v)], verbose=False)

        xgb_probs = xgb.predict_proba(feat_test)[:, 1]
        xgb_preds = xgb.predict(feat_test)
        xgb_acc = np.mean(xgb_preds == y_te)
        print(f"XGBoost Accuracy: {xgb_acc:.4f}")

        # Ensemble
        ensemble_probs = 0.5 * cnn_probs + 0.5 * xgb_probs
        ensemble_preds = (ensemble_probs > 0.5).astype(int)

        has_xgb = True
    except Exception as e:
        print(f"XGBoost failed: {e}. Using CNN-only results.")
        xgb_probs = cnn_probs
        xgb_preds = cnn_preds
        ensemble_probs = cnn_probs
        ensemble_preds = cnn_preds
        has_xgb = False

    # --- Final Metrics ---
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    results = [("CNN Only", cnn_preds, cnn_probs)]
    if has_xgb:
        results.append(("XGBoost Only", xgb_preds, xgb_probs))
        results.append(("Ensemble (CNN+XGB)", ensemble_preds, ensemble_probs))

    for name, preds, probs in results:
        C = confusion_matrix(y_true, preds, labels=[1, 0])
        TP, TN = C[0, 0], C[1, 1]
        FP, FN = C[1, 0], C[0, 1]
        acc = (TP + TN) / (TP + TN + FP + FN)
        sn = TP / (TP + FN) if (TP + FN) > 0 else 0
        sp = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = f1_score(y_true, preds)
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.0

        print(f"\n{name}:")
        print(f"  Accuracy:    {acc:.4f} ({acc:.2%})")
        print(f"  Sensitivity: {sn:.4f} ({sn:.2%})")
        print(f"  Specificity: {sp:.4f} ({sp:.2%})")
        print(f"  F1-score:    {f1:.4f}")
        print(f"  AUC-ROC:     {auc:.4f}")

    # Save best predictions (ensemble)
    output = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": ensemble_preds,
            "y_score": ensemble_probs,
            "y_score_cnn": cnn_probs,
            "subject": data["groups_test"],
        }
    )
    if has_xgb:
        output["y_score_xgb"] = xgb_probs
    output.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nPredictions saved to {PREDICTIONS_CSV}")
    print(f"Model weights saved to {WEIGHTS_SAVE}")

    # Final verdict
    ens_acc = np.mean(ensemble_preds == y_true)
    print(f"\n{'=' * 60}")
    print(f"BEST ACCURACY: {ens_acc:.2%}")
    if ens_acc >= 0.95:
        print("TARGET 95% ACHIEVED!")
    else:
        print(f"Gap to target: {0.95 - ens_acc:.2%}")
    print(f"{'=' * 60}")
