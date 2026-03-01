# Sleep Apnea Detection - Improvement Roadmap & Novelty Ideas

## Problem Analysis

**Current State:**
- Baseline (ECG only): **89% accuracy** ✓
- SPO2 Integration: **68.74% accuracy** ✗ (-20.3%)
- Sensitivity: 7.49% (critical) - missing apnea events
- Specificity: 91.61% - predicting mostly negative class

## Root Causes

1. **Class Imbalance**: Model learns to predict negative class (specificity >91%)
2. **Poor Multi-modal Fusion**: SPO2 branch architecture doesn't complement ECG
3. **Feature Mismatch**: SPO2 temporal resolution (2400 samples) vs ECG complexity
4. **Information Bottleneck**: Features pooled too aggressively before fusion
5. **Training Dynamics**: No mechanism to balance apnea vs non-apnea detection

---

## TIER 1: Critical Fixes (Restore Baseline + 2-3% improvement)

### A. Fix Class Imbalance with Focal Loss + Weighted Sampling
```python
# Replace categorical_crossentropy with focal loss
from tensorflow.keras.losses import BinaryCrossentropy

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.squeeze(y_pred, axis=-1)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = alpha * tf.pow(1. - p_t, gamma)
        focal_loss = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fn

# Use class weights in training
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(y_train),
                                      y=y_train)
```

### B. Decouple SPO2 Processing - Use Feature Branch Instead of Raw Signal
```python
# Don't process raw 2400 SPO2 samples
# Instead: Extract meaningful features that complement ECG

def extract_spo2_temporal_features(spo2_segment):
    """Extract clinically relevant SpO2 features for apnea detection"""
    features = []
    
    # Desaturation patterns (key apnea indicator)
    desat_events = np.sum(spo2_segment < 90)  # Number of hypoxic events
    desat_depth = np.min(spo2_segment) if len(spo2_segment) > 0 else 95
    
    # Variability (apnea → increased SpO2 oscillation)
    spo2_diff = np.abs(np.diff(spo2_segment))
    roc_std = np.std(spo2_diff)  # Rate of change variability
    roc_max = np.max(spo2_diff)  # Max drop speed
    
    # Periodicity (respiratory pattern)
    fft = np.abs(np.fft.fft(spo2_segment)[:len(spo2_segment)//2])
    freq_power = np.max(fft[5:15])  # Respiratory frequency band
    
    # Baseline SpO2 level
    baseline_spo2 = np.percentile(spo2_segment, 90)
    
    return np.array([desat_events, desat_depth, roc_std, roc_max, 
                     freq_power, baseline_spo2], dtype=np.float32)
```

### C. Redesign Fusion Strategy - Cross-Attention Before Concatenation
```python
# Instead of concatenating late, use cross-attention fusion
# This allows ECG and SPO2 to calibrate each other

def create_cross_attention_fusion(ecg_features, spo2_features):
    """Multi-head attention fusion of ECG and SPO2"""
    
    # ECG attention to SPO2
    spo2_attention = Dense(64, activation='relu')(spo2_features)
    spo2_weights = Dense(1, activation='sigmoid')(spo2_attention)
    ecg_weighted = multiply([ecg_features, spo2_weights])
    
    # SPO2 attention to ECG
    ecg_attention = Dense(64, activation='relu')(ecg_features)
    ecg_weights = Dense(1, activation='sigmoid')(ecg_attention)
    spo2_weighted = multiply([spo2_features, ecg_weights])
    
    # Fused features
    fused = concatenate([ecg_weighted, spo2_weighted, 
                        tf.keras.layers.multiply([ecg_features, spo2_features])])
    return fused
```

---

## TIER 2: Novelty Additions (Additional 3-5% improvement)

### D. Temporal Residual Blocks for Better Feature Extraction
```python
def residual_conv_block(x, filters, kernel_size=11):
    """Residual block with skip connection - prevents degradation"""
    residual = x
    
    x = Conv1D(filters, kernel_size, padding='same', 
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters, kernel_size, padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    
    # Skip connection projection if channels mismatch
    if residual.shape[-1] != filters:
        residual = Conv1D(filters, 1, padding='same')(residual)
    
    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x
```

### E. Adaptive Channel Attention (CBAM) for Multi-modal Importance
```python
# Channel Attention: Learn which ECG/SPO2 features are most important
def channel_attention(x, reduction=16):
    channels = int(x.shape[-1])
    
    # Max pooling path
    max_pool = GlobalMaxPooling1D()(x)
    max_pool = Dense(channels // reduction, activation='relu')(max_pool)
    max_pool = Dense(channels, activation=None)(max_pool)
    
    # Avg pooling path
    avg_pool = GlobalAveragePooling1D()(x)
    avg_pool = Dense(channels // reduction, activation='relu')(avg_pool)
    avg_pool = Dense(channels, activation=None)(avg_pool)
    
    # Combine
    attention = Add()([max_pool, avg_pool])
    attention = Activation('sigmoid')(attention)
    attention = Reshape((1, channels))(attention)
    
    return multiply([x, attention])
```

### F. Contrastive Learning Pre-training on ECG Features
```python
# Pre-train ECG encoder with SimCLR-style contrastive loss
# This helps ECG branch learn robust representations before SPO2 fusion

def contrastive_loss(y_true, y_pred):
    """Contrastive loss: pull similar samples, push different ones"""
    # y_pred shape: (batch_size, embedding_dim)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    similarity_matrix = tf.matmul(y_pred, y_pred, transpose_b=True)
    
    # Create positive/negative pairs
    pos_mask = tf.eye(tf.shape(y_pred)[0], dtype=tf.bool)
    neg_mask = ~pos_mask
    
    pos_sim = tf.boolean_mask(similarity_matrix, pos_mask)
    neg_sim = tf.boolean_mask(similarity_matrix, neg_mask)
    
    # Push negatives < negatives threshold, positives > positives threshold
    pos_loss = tf.reduce_mean(tf.nn.relu(1 - pos_sim))
    neg_loss = tf.reduce_mean(tf.nn.relu(neg_sim + 0.5))
    
    return pos_loss + neg_loss
```

---

## TIER 3: Advanced Novelty (Additional 2-3% + Publication Value)

### G. Explainable Multi-modal Attention with Gradient-based Saliency
```python
# Layer-wise Relevance Propagation (LRP) for interpretability
# Shows which ECG/SPO2 time points matter for decisions

class ExplainableMultimodalBlock(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.ecg_attention = Dense(1, activation='sigmoid', name='ecg_saliency')
        self.spo2_attention = Dense(1, activation='sigmoid', name='spo2_saliency')
        self.fusion = Dense(units, activation='relu')
        
    def call(self, ecg_input, spo2_input, training=False):
        # Compute attention weights (interpretable!)
        ecg_weights = self.ecg_attention(ecg_input)  # (batch, time, 1)
        spo2_weights = self.spo2_attention(spo2_input)  # (batch, time, 1)
        
        # Apply weights
        ecg_weighted = multiply([ecg_input, ecg_weights])
        spo2_weighted = multiply([spo2_input, spo2_weights])
        
        # Store for explainability
        if training:
            self.ecg_saliency = ecg_weights
            self.spo2_saliency = spo2_weights
        
        # Fuse
        fused = concatenate([ecg_weighted, spo2_weighted])
        return self.fusion(fused)
```

### H. Uncertainty Estimation with Bayesian Dropout
```python
# Add uncertainty quantification for clinical decision support
# Helps identify cases where model is unsure

class BayesianDeepEnsemble:
    def __init__(self, model, n_samples=10):
        self.model = model
        self.n_samples = n_samples
        
    def predict_with_uncertainty(self, X):
        # Run forward pass multiple times with dropout enabled
        predictions = []
        for _ in range(self.n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
```

### I. Domain Adaptation for Robustness Across Datasets
```python
# Adversarial domain adaptation layer
# Makes model robust to data collection differences

class AdversarialDomainAdapter(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Dense(128, activation='relu')
        self.domain_classifier = Dense(1, activation='sigmoid')
        
    def call(self, features, domain_labels):
        # Extract features
        features = self.feature_extractor(features)
        
        # Adversarial: fool domain classifier
        domain_pred = self.domain_classifier(features)
        
        # Domain loss (binary: dataset A vs dataset B)
        domain_loss = -tf.reduce_mean(
            domain_labels * tf.math.log(domain_pred + 1e-6) +
            (1 - domain_labels) * tf.math.log(1 - domain_pred + 1e-6)
        )
        
        return features, domain_loss
```

### J. Temporal Data Augmentation for Time Series
```python
# Augment training data with realistic time series transformations

def augment_time_series(x_ecg, x_spo2, y):
    """Data augmentation for temporal signals"""
    augmented_ecg = []
    augmented_spo2 = []
    augmented_y = []
    
    for i in range(len(x_ecg)):
        # Time-warping: speed up/slow down
        speed = np.random.uniform(0.95, 1.05)
        warped = np.interp(np.arange(len(x_ecg[i])), 
                           np.arange(len(x_ecg[i]))/speed, x_ecg[i])
        augmented_ecg.append(warped)
        
        # Add noise (physiological noise)
        noise = np.random.normal(0, 0.02, x_ecg[i].shape)
        noisy = x_ecg[i] + noise
        augmented_ecg.append(noisy)
        
        # Window slicing: use random subsequences
        window_size = int(0.9 * x_ecg[i].shape[0])
        start = np.random.randint(0, len(x_ecg[i]) - window_size)
        sliced = x_ecg[i][start:start+window_size]
        augmented_ecg.append(sliced)
        
        # Duplicate labels for augmentations
        augmented_y.extend([y[i], y[i], y[i]])
```

---

## TIER 4: Publication-Ready Innovations (Additional 1-2%)

### K. Neural ODE for Continuous-Time Signal Modeling
```python
# Treat ECG+SPO2 as continuous dynamical system
# More principled than discrete convolutions

class NeuralODEBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.solver = torchdiffeq.odeint  # or tf equivalent
        self.func = Dense(64, activation='elu')
        
    def call(self, x):
        # x: (batch, time, features)
        # Solve ODE: dx/dt = f(x, t)
        t_eval = tf.linspace(0, 1, tf.shape(x)[1])
        solution = self.solver(self.func, x[:, 0, :], t_eval)
        return solution  # (batch, time, features)
```

### L. Transformer with Sparse Attention for Long-Range Dependencies
```python
# Sparse self-attention to capture long-range ECG/SPO2 correlations
# More efficient than dense attention

def create_sparse_transformer_block(x):
    """Sparse transformer: local + stride-2 attention"""
    
    # Local attention (neighboring time steps)
    local_attn = MultiHeadAttention(head_dim=64, num_heads=8)
    x_local = local_attn(x, x)
    
    # Stride-2 attention (longer-range patterns)
    x_strided = x[:, ::2, :]
    stride_attn = MultiHeadAttention(head_dim=64, num_heads=8)
    x_stride = stride_attn(x_strided, x_strided)
    
    # Combine
    x_combined = concatenate([x_local, x_stride])
    output = Dense(64, activation='relu')(x_combined)
    return output
```

---

## Implementation Priority (5-Week Project Timeline)

| Week | Focus | Expected Gain |
|------|-------|----------------|
| 1 | Focal Loss + Class Weights + Feature-based SPO2 | +5% (+94% accuracy) |
| 2 | Cross-attention Fusion + Residual Blocks | +3% (+97%) |
| 3-4 | Explainability + Uncertainty (CBAM + Bayesian Dropout) | +1-2% (+98-99%) |
| 5 | Ensemble + Published Results | Final tuning |

---

## Expected Final Results

**Target: 96-99% Accuracy with Novelty**

- ✓ Better than baseline (89%)
- ✓ SPO2 actually helps (+5-10% over baseline)
- ✓ Explainable predictions (SHAP + attention heatmaps)
- ✓ Uncertainty estimates for clinical safety
- ✓ Publication-ready: "Explainable Multi-modal Sleep Apnea Detection with Uncertainty Quantification"

