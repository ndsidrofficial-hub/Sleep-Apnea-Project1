import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

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
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch)
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
            nn.Conv1d(in_channels, 32, 11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU()
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
            nn.Sigmoid()
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

def run_benchmark():
    print("============================================================")
    print("   SE-MSCNN v2 Benchmark - Sleep Apnea Detection")
    print("============================================================")
    
    start_time = time.time()
    
    print("Loading test data from cache...")
    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"[Dataset] Train: {len(data['y_train'])}, Val: {len(data['y_val'])}, Test: {len(data['y_test'])}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedSEMSCNN().to(device)
    
    print("\nLoading weights.v2_improved.pt (94.75% val accuracy)...")
    model.load_state_dict(torch.load('weights.v2_improved.pt', map_location=device))
    model.eval()
    
    # 1. Prepare PyTorch tensors
    test_1 = torch.FloatTensor(data['x_test1']).transpose(1, 2).to(device)
    test_2 = torch.FloatTensor(data['x_test2']).transpose(1, 2).to(device)
    test_3 = torch.FloatTensor(data['x_test3']).transpose(1, 2).to(device)
    y_test = data['y_test']
    
    # 2. Extract Deep Features + CNN Predictions
    print("\nRunning inference and feature extraction on test set (17,075 segments)...")
    batch_size = 512
    all_features_train = []
    all_features_test = []
    all_cnn_preds = []
    
    with torch.no_grad():
        for i in range(0, len(y_test), batch_size):
            b1 = test_1[i:i+batch_size]
            b2 = test_2[i:i+batch_size]
            b3 = test_3[i:i+batch_size]
            
            logits, features = model(b1, b2, b3, return_features=True)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_cnn_preds.extend(probs)
            all_features_test.append(features.cpu().numpy())
            
    test_deep_features = np.vstack(all_features_test)
    cnn_test_probs = np.array(all_cnn_preds)
    
    print("\nTraining XGBoost ensemble on deep features...")
    # Get Train Features
    train_1 = torch.FloatTensor(data['x_train1']).transpose(1, 2).to(device)
    train_2 = torch.FloatTensor(data['x_train2']).transpose(1, 2).to(device)
    train_3 = torch.FloatTensor(data['x_train3']).transpose(1, 2).to(device)
    
    with torch.no_grad():
        for i in range(0, len(data['y_train']), batch_size):
            b1 = train_1[i:i+batch_size]
            b2 = train_2[i:i+batch_size]
            b3 = train_3[i:i+batch_size]
            _, features = model(b1, b2, b3, return_features=True)
            all_features_train.append(features.cpu().numpy())
            
    train_deep_features = np.vstack(all_features_train)
    
    xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
    xgb_model.fit(train_deep_features, data['y_train'])
    
    xgb_test_probs = xgb_model.predict_proba(test_deep_features)[:, 1]
    
    ensemble_probs = (cnn_test_probs + xgb_test_probs) / 2
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    end_time = time.time()
    
    print("============================================================")
    print("                    FINAL BENCHMARK")
    print("============================================================")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    cm = confusion_matrix(y_test, ensemble_preds)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / len(y_test)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nENSEMBLE MODEL PERFORMANCE:")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Sensitivity: {sens*100:.2f}%")
    print(f"Specificity: {spec*100:.2f}%")
    print(f"AUC-ROC:     {roc_auc:.4f}")
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, ensemble_preds, target_names=['Normal', 'Apnea']))
    
    # ---- Generation Visualization ----
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'])
    axes[0].set_title('Confusion Matrix (Ensemble)', fontsize=14)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    axes[1].plot(fpr, tpr, color='cyan', lw=2, label=f'Ensemble ROC (AUC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('benchmark_plot.png', dpi=300)
    print("\nPlot saved as benchmark_plot.png. Ready for your screenshot!")

if __name__ == '__main__':
    run_benchmark()
