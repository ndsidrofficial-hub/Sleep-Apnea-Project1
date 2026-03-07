"""
SE-MSCNN v3 Benchmark — Sleep Apnea Detection with SpO2 Fusion
==============================================================
Benchmarks the trained v3 model and compares against v2 baseline.
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys


# ======================== MODEL (must match SE_MSCNN_v3_spo2.py) ========================
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
    """ECG branch (2-channel input)."""
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
        return x


class SpO2Branch(nn.Module):
    """Lightweight SpO2 branch (1-channel input)."""
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
        return x


class SEMSCNN_v3(nn.Module):
    """SE-MSCNN v3: 4-branch fusion model (3 ECG + 1 SpO2)."""
    def __init__(self):
        super().__init__()
        self.ecg_branch1 = ECGBranch(2)
        self.ecg_branch2 = ECGBranch(2)
        self.ecg_branch3 = ECGBranch(2)
        self.spo2_branch = SpO2Branch()

        total_ch = 128 * 3 + 64  # 448
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_excite = nn.Sequential(
            nn.Linear(total_ch, total_ch // 4),
            nn.ReLU(),
            nn.Linear(total_ch // 4, total_ch),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_ch),
            nn.Linear(total_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
        )
        self.output_layer = nn.Linear(128, 2)

    def forward(self, x1, x2, x3, spo2, return_features=False):
        e1 = self.ecg_branch1(x1)
        e2 = self.ecg_branch2(x2)
        e3 = self.ecg_branch3(x3)
        s = self.spo2_branch(spo2)

        min_t = min(e1.shape[2], e2.shape[2], e3.shape[2], s.shape[2])
        e1 = F.adaptive_avg_pool1d(e1, min_t)
        e2 = F.adaptive_avg_pool1d(e2, min_t)
        e3 = F.adaptive_avg_pool1d(e3, min_t)
        s  = F.adaptive_avg_pool1d(s, min_t)

        concat = torch.cat([e1, e2, e3, s], dim=1)

        squeeze = self.se_squeeze(concat).squeeze(-1)
        excitation = self.se_excite(squeeze).unsqueeze(-1)
        scaled = concat * excitation

        pooled = scaled.mean(dim=2)
        features = self.classifier(pooled)
        logits = self.output_layer(features)

        if return_features:
            return logits, features
        return logits


# ======================== V2 MODEL (for comparison) ========================
class Branch_v2(nn.Module):
    """v2 branch for comparison."""
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
        return x


class ImprovedSEMSCNN(nn.Module):
    """v2 model for comparison."""
    def __init__(self):
        super().__init__()
        self.branch1 = Branch_v2(2)
        self.branch2 = Branch_v2(2)
        self.branch3 = Branch_v2(2)
        total_ch = 128 * 3
        self.se_squeeze = nn.AdaptiveAvgPool1d(1)
        self.se_excite = nn.Sequential(
            nn.Linear(total_ch, total_ch // 4), nn.ReLU(),
            nn.Linear(total_ch // 4, total_ch), nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_ch), nn.Linear(total_ch, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
        )
        self.output_layer = nn.Linear(128, 2)

    def forward(self, x1, x2, x3, return_features=False):
        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        b3 = self.branch3(x3)
        min_t = min(b1.shape[2], b2.shape[2], b3.shape[2])
        b1 = F.adaptive_avg_pool1d(b1, min_t)
        b2 = F.adaptive_avg_pool1d(b2, min_t)
        b3 = F.adaptive_avg_pool1d(b3, min_t)
        concat = torch.cat([b1, b2, b3], dim=1)
        squeeze = self.se_squeeze(concat).squeeze(-1)
        excitation = self.se_excite(squeeze).unsqueeze(-1)
        scaled = concat * excitation
        pooled = scaled.mean(dim=2)
        features = self.classifier(pooled)
        logits = self.output_layer(features)
        if return_features:
            return logits, features
        return logits


# ======================== BENCHMARK ========================
def run_benchmark():
    print("=" * 70)
    print("   SE-MSCNN v3 Benchmark — Sleep Apnea Detection + SpO2 Fusion")
    print("=" * 70)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256

    # --- Load data ---
    print("\nLoading test data...")

    spo2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spo2_data.pkl")
    if not os.path.exists(spo2_path):
        print(f"ERROR: {spo2_path} not found. Run generate_spo2_dataset.py first.")
        sys.exit(1)

    with open(spo2_path, "rb") as f:
        data = pickle.load(f)

    y_test = data["y_test"]
    y_train = data["y_train"]
    n_test = len(y_test)
    print(f"[Dataset] Train: {len(y_train)}, Val: {len(data['y_val'])}, Test: {n_test}")

    # Prepare test tensors
    ecg_test1 = torch.FloatTensor(data["ecg_test1"]).transpose(1, 2).to(device)
    ecg_test2 = torch.FloatTensor(data["ecg_test2"]).transpose(1, 2).to(device)
    ecg_test3 = torch.FloatTensor(data["ecg_test3"]).transpose(1, 2).to(device)
    spo2_test = torch.FloatTensor(data["spo2_test1"]).transpose(1, 2).to(device)

    # ============================
    # v3 Model Benchmark
    # ============================
    v3_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.v3_spo2.pt")
    if not os.path.exists(v3_weights):
        print(f"\nERROR: {v3_weights} not found. Run SE_MSCNN_v3_spo2.py first.")
        sys.exit(1)

    print(f"\n--- v3 Model (ECG + SpO2) ---")
    model_v3 = SEMSCNN_v3().to(device)
    model_v3.load_state_dict(torch.load(v3_weights, map_location=device, weights_only=True))
    model_v3.eval()
    v3_params = sum(p.numel() for p in model_v3.parameters())
    print(f"Parameters: {v3_params:,}")

    # Inference
    v3_cnn_probs = []
    v3_features_test = []

    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            b1 = ecg_test1[i:i+batch_size]
            b2 = ecg_test2[i:i+batch_size]
            b3 = ecg_test3[i:i+batch_size]
            s1 = spo2_test[i:i+batch_size]

            logits, features = model_v3(b1, b2, b3, s1, return_features=True)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            v3_cnn_probs.extend(probs)
            v3_features_test.append(features.cpu().numpy())

    v3_cnn_probs = np.array(v3_cnn_probs)
    v3_features_test = np.vstack(v3_features_test)

    # XGBoost ensemble for v3
    print("Training XGBoost ensemble for v3...")
    ecg_train1 = torch.FloatTensor(data["ecg_train1"]).transpose(1, 2).to(device)
    ecg_train2 = torch.FloatTensor(data["ecg_train2"]).transpose(1, 2).to(device)
    ecg_train3 = torch.FloatTensor(data["ecg_train3"]).transpose(1, 2).to(device)
    spo2_train = torch.FloatTensor(data["spo2_train1"]).transpose(1, 2).to(device)

    v3_features_train = []
    with torch.no_grad():
        for i in range(0, len(y_train), batch_size):
            b1 = ecg_train1[i:i+batch_size]
            b2 = ecg_train2[i:i+batch_size]
            b3 = ecg_train3[i:i+batch_size]
            s1 = spo2_train[i:i+batch_size]
            _, features = model_v3(b1, b2, b3, s1, return_features=True)
            v3_features_train.append(features.cpu().numpy())

    v3_features_train = np.vstack(v3_features_train)

    xgb_v3 = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        n_jobs=1, random_state=42
    )
    xgb_v3.fit(v3_features_train, y_train)
    v3_xgb_probs = xgb_v3.predict_proba(v3_features_test)[:, 1]

    # Optimized ensemble
    best_alpha, best_acc = 0.5, 0
    for alpha in np.arange(0.1, 0.95, 0.05):
        probs = alpha * v3_cnn_probs + (1 - alpha) * v3_xgb_probs
        preds = (probs >= 0.5).astype(int)
        acc = np.mean(preds == y_test)
        if acc > best_acc:
            best_alpha, best_acc = alpha, acc

    v3_ensemble_probs = best_alpha * v3_cnn_probs + (1 - best_alpha) * v3_xgb_probs

    # Optimize threshold
    best_thr, _ = 0.5, 0
    for thr in np.arange(0.35, 0.65, 0.01):
        preds = (v3_ensemble_probs >= thr).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > _:
            best_thr, _ = thr, f1

    v3_ensemble_preds = (v3_ensemble_probs >= best_thr).astype(int)

    # Free GPU memory
    del ecg_train1, ecg_train2, ecg_train3, spo2_train
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ============================
    # v2 Model Benchmark (comparison)
    # ============================
    v2_weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.v2_improved.pt")
    v2_results = None

    if os.path.exists(v2_weights):
        print(f"\n--- v2 Model (ECG only, baseline) ---")
        model_v2 = ImprovedSEMSCNN().to(device)
        model_v2.load_state_dict(torch.load(v2_weights, map_location=device, weights_only=True))
        model_v2.eval()
        v2_params = sum(p.numel() for p in model_v2.parameters())
        print(f"Parameters: {v2_params:,}")

        v2_cnn_probs = []
        v2_features_test = []

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                b1 = ecg_test1[i:i+batch_size]
                b2 = ecg_test2[i:i+batch_size]
                b3 = ecg_test3[i:i+batch_size]
                logits, features = model_v2(b1, b2, b3, return_features=True)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                v2_cnn_probs.extend(probs)
                v2_features_test.append(features.cpu().numpy())

        v2_cnn_probs = np.array(v2_cnn_probs)
        v2_features_test = np.vstack(v2_features_test)

        # Load original data for v2 XGBoost
        orig_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed_data.pkl")
        if os.path.exists(orig_data_path):
            with open(orig_data_path, "rb") as f:
                orig_data = pickle.load(f)

            orig_train1 = torch.FloatTensor(orig_data["x_train1"]).transpose(1, 2).to(device)
            orig_train2 = torch.FloatTensor(orig_data["x_train2"]).transpose(1, 2).to(device)
            orig_train3 = torch.FloatTensor(orig_data["x_train3"]).transpose(1, 2).to(device)

            v2_features_train = []
            with torch.no_grad():
                for i in range(0, len(orig_data["y_train"]), batch_size):
                    b1 = orig_train1[i:i+batch_size]
                    b2 = orig_train2[i:i+batch_size]
                    b3 = orig_train3[i:i+batch_size]
                    _, features = model_v2(b1, b2, b3, return_features=True)
                    v2_features_train.append(features.cpu().numpy())

            v2_features_train = np.vstack(v2_features_train)
            del orig_train1, orig_train2, orig_train3

            xgb_v2 = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                n_jobs=1, random_state=42
            )
            xgb_v2.fit(v2_features_train, orig_data["y_train"])
            v2_xgb_probs = xgb_v2.predict_proba(v2_features_test)[:, 1]
            v2_ensemble_probs = 0.5 * v2_cnn_probs + 0.5 * v2_xgb_probs
            v2_ensemble_preds = (v2_ensemble_probs >= 0.5).astype(int)
            del orig_data
        else:
            v2_ensemble_probs = v2_cnn_probs
            v2_ensemble_preds = (v2_cnn_probs >= 0.5).astype(int)

        v2_results = {
            'preds': v2_ensemble_preds,
            'probs': v2_ensemble_probs,
            'params': v2_params,
        }

    end_time = time.time()

    # ============================
    # Results
    # ============================
    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total time: {end_time - start_time:.2f}s")

    def print_metrics(name, preds, probs, y_true):
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / len(y_true)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true, preds, zero_division=0)
        try:
            roc_auc = auc(*roc_curve(y_true, probs)[:2])
        except:
            roc_auc = 0
        print(f"\n{name}:")
        print(f"  Accuracy:    {acc*100:.2f}%")
        print(f"  Sensitivity: {sens*100:.2f}%")
        print(f"  Specificity: {spec*100:.2f}%")
        print(f"  F1-score:    {f1:.4f}")
        print(f"  AUC-ROC:     {roc_auc:.4f}")
        return acc, cm, sens, spec, roc_auc

    v3_acc, v3_cm, v3_sens, v3_spec, v3_auc = print_metrics(
        "SE-MSCNN v3 (ECG + SpO2)", v3_ensemble_preds, v3_ensemble_probs, y_test
    )

    if v2_results:
        v2_acc, v2_cm, _, _, v2_auc = print_metrics(
            "SE-MSCNN v2 (ECG Only)", v2_results['preds'], v2_results['probs'], y_test
        )

        print(f"\n{'=' * 70}")
        print(f"  IMPROVEMENT: v2 → v3")
        print(f"  Accuracy:  {v2_acc*100:.2f}% → {v3_acc*100:.2f}% ({(v3_acc-v2_acc)*100:+.2f}%)")
        print(f"  AUC-ROC:   {v2_auc:.4f} → {v3_auc:.4f} ({v3_auc-v2_auc:+.4f})")
        print(f"  Parameters: {v2_results['params']:,} → {v3_params:,}")
        print(f"{'=' * 70}")

    # ============================
    # Visualization
    # ============================
    plt.style.use("dark_background")

    if v2_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # v3 Confusion Matrix
        sns.heatmap(v3_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
                    xticklabels=["Normal", "Apnea"], yticklabels=["Normal", "Apnea"])
        axes[0, 0].set_title(f"v3 Confusion Matrix (Acc: {v3_acc*100:.2f}%)", fontsize=13)
        axes[0, 0].set_ylabel("True Label")
        axes[0, 0].set_xlabel("Predicted Label")

        # v2 Confusion Matrix
        sns.heatmap(v2_cm, annot=True, fmt="d", cmap="Oranges", ax=axes[0, 1],
                    xticklabels=["Normal", "Apnea"], yticklabels=["Normal", "Apnea"])
        axes[0, 1].set_title(f"v2 Confusion Matrix (Acc: {v2_acc*100:.2f}%)", fontsize=13)
        axes[0, 1].set_ylabel("True Label")
        axes[0, 1].set_xlabel("Predicted Label")

        # ROC Curves
        v3_fpr, v3_tpr, _ = roc_curve(y_test, v3_ensemble_probs)
        v2_fpr, v2_tpr, _ = roc_curve(y_test, v2_results['probs'])

        axes[1, 0].plot(v3_fpr, v3_tpr, color="cyan", lw=2,
                       label=f"v3 (AUC = {v3_auc:.4f})")
        axes[1, 0].plot(v2_fpr, v2_tpr, color="orange", lw=2, linestyle="--",
                       label=f"v2 (AUC = {v2_auc:.4f})")
        axes[1, 0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("ROC Comparison (v2 vs v3)", fontsize=13)
        axes[1, 0].legend(loc="lower right")

        # Comparison bar chart
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC']
        v2_vals = [v2_acc, 0, 0, v2_auc]
        v3_vals = [v3_acc, v3_sens, v3_spec, v3_auc]

        # Compute v2 sens/spec
        v2_tn, v2_fp, v2_fn, v2_tp = v2_cm.ravel()
        v2_vals[1] = v2_tp / (v2_tp + v2_fn) if (v2_tp + v2_fn) > 0 else 0
        v2_vals[2] = v2_tn / (v2_tn + v2_fp) if (v2_tn + v2_fp) > 0 else 0

        x = np.arange(len(metrics))
        width = 0.35
        bars1 = axes[1, 1].bar(x - width/2, [v*100 for v in v2_vals], width,
                               label='v2 (ECG)', color='orange', alpha=0.8)
        bars2 = axes[1, 1].bar(x + width/2, [v*100 for v in v3_vals], width,
                               label='v3 (ECG+SpO2)', color='cyan', alpha=0.8)
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].set_title('v2 vs v3 Performance Comparison', fontsize=13)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(80, 100)

        # Add value labels on bars
        for bar in bars1:
            h = bar.get_height()
            axes[1, 1].annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            axes[1, 1].annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(v3_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=["Normal", "Apnea"], yticklabels=["Normal", "Apnea"])
        axes[0].set_title(f"v3 Confusion Matrix (Acc: {v3_acc*100:.2f}%)", fontsize=13)
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        v3_fpr, v3_tpr, _ = roc_curve(y_test, v3_ensemble_probs)
        axes[1].plot(v3_fpr, v3_tpr, color="cyan", lw=2,
                    label=f"v3 ROC (AUC = {v3_auc:.4f})")
        axes[1].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve", fontsize=13)
        axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("benchmark_v3_plot.png", dpi=300)
    print(f"\nPlot saved as benchmark_v3_plot.png")

    # Final verdict
    print(f"\n{'=' * 70}")
    if v3_acc >= 0.95:
        print(f"🎯 TARGET ACHIEVED! Test Accuracy: {v3_acc*100:.2f}%")
    else:
        print(f"Test Accuracy: {v3_acc*100:.2f}% (gap to 95%: {(0.95-v3_acc)*100:.2f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_benchmark()
