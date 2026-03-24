# Ensemble Logic: SE-MSCNN + XGBoost

## Overview

The SE-MSCNN v2/v3 pipeline uses a **hybrid ensemble** combining:
1. The CNN's raw probability output (soft score)
2. XGBoost trained on the CNN's **penultimate-layer deep features**

This two-stage approach consistently provides a **1–2% gain** in both AUC-ROC and accuracy over CNN-alone inference.

---

## Stage 1: Penultimate Layer Feature Extraction

The SE-MSCNN exposes a 128-dimensional latent vector from its classification head. This vector captures the global multi-scale temporal context fused from all ECG and SpO2 branches.

```python
def extract_features(model, loader):
    model.eval()
    all_features, all_true = [], []
    with torch.no_grad():
        for x1, x2, x3, s1, s2, s3, y in loader:
            x1, x2, x3, s1 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE), s1.to(DEVICE)
            # return_features=True returns (logits, 128-d feature vector)
            _, features = model(x1, x2, x3, s1, return_features=True)
            all_features.append(features.cpu().numpy())
            all_true.append(y.numpy())
    return np.concatenate(all_features), np.concatenate(all_true)
```

---

## Stage 2: XGBoost on Deep Features

The XGBoost tree ensemble acts as a **secondary classifier** on top of the CNN's learned representations. It is particularly effective at capturing non-linear class boundaries in the latent space that the linear output head may miss.

```python
from xgboost import XGBClassifier

# Balance the class contribution using the training set ratio
scale_pos = n_normal / max(n_apnea, 1)

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,  # compensates for apnea class imbalance
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
    n_jobs=1,  # RAM-safe on consumer hardware
)
xgb_model.fit(feat_train, y_tr, eval_set=[(feat_val, y_v)], verbose=False)
```

---

## Stage 3: Weighted Probability Ensemble

The final prediction blends the CNN and XGBoost probability scores using a weighted average. The optimal `alpha` weight is grid-searched over the **validation set** to maximise accuracy before being applied to the held-out test set.

```python
# --- Find optimal alpha on validation set ---
best_alpha, best_alpha_acc = 0.5, 0
for alpha in np.arange(0.1, 0.95, 0.05):
    ens_probs = alpha * cnn_val_probs + (1 - alpha) * xgb_val_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    acc = np.mean(ens_preds == val_true_labels)
    if acc > best_alpha_acc:
        best_alpha, best_alpha_acc = alpha, acc

# --- Apply to test set ---
ensemble_probs = best_alpha * cnn_probs + (1 - best_alpha) * xgb_probs
```

---

## Stage 4: Optimal Threshold Search

Instead of using a fixed 0.5 threshold, the final decision boundary is tuned to maximise **F1-score** on the validation set — important for unbalanced clinical data.

```python
def find_best_threshold(probs, labels):
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.3, 0.7, 0.01):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr, best_f1

best_thr, _ = find_best_threshold(ens_val_probs, val_true_labels)
ensemble_preds = (ensemble_probs >= best_thr).astype(int)
```

---

## Clinical Trade-offs

| Tuning Goal | Strategy |
|---|---|
| High clinical **sensitivity** (catch all apnea) | Lower the threshold (e.g., 0.35–0.40) |
| High **specificity** (fewer false alarms) | Raise the threshold (e.g., 0.55–0.60) |
| Balanced **F1** | Use `find_best_threshold()` on validation |

---

## Performance Notes

- Ensemble outperforms CNN-only by **~1–2% AUC-ROC** consistently
- XGBoost's `early_stopping_rounds=30` prevents overfitting on small val sets
- `scale_pos_weight` is critical when the apnea class is underrepresented (~40%)
