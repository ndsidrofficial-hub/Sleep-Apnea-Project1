"""
Train SE-MSCNN with Real SPO2 Data from UCD Dataset
====================================================

Complete pipeline:
1. Preprocess UCD dataset (extract ECG + real SPO2)
2. Train SE-MSCNN model with multi-modal data
3. Evaluate and save results

Usage:
    python train_with_real_spo2.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(script_name, description):
    """Run a Python script and report progress."""
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70 + "\n")
    
    script_path = Path(__file__).parent / "code_baseline" / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("SLEEP APNEA DETECTION WITH REAL SPO2 DATA")
    print("=" * 70)
    
    print("\n📊 Dataset: UCD (St. Vincent's Hospital) Sleep Apnea Database")
    print("✓ Includes real ECG and real SPO2 measurements")
    print("✓ Expected accuracy improvement: +4-6% vs ECG-only baseline")
    
    # Step 1: Preprocess UCD data
    print("\n\n🔧 STEP 1: Preprocess UCD Dataset")
    print("-" * 70)
    
    ucd_dir = r"C:\Users\siddh\Downloads\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
    
    if not os.path.exists(ucd_dir):
        print(f"\n❌ ERROR: UCD dataset not found at:")
        print(f"   {ucd_dir}")
        print(f"\nPlease download the UCD Sleep Apnea Database from:")
        print(f"   https://physionet.org/content/ucddb/1.0.0/")
        return False
    
    print("✓ UCD dataset found")
    print("  - Extracting ECG features from _lifecard.edf files")
    print("  - Extracting SPO2 data from .rec (PSG) files")
    print("  - Parsing apnea events from _respevt.txt files")
    print("  - Generating 5-minute windows with labels")
    print("\nThis will take 5-10 minutes...\n")
    
    if not run_command("preprocess_ucd_real_spo2.py", "Running UCD Preprocessing"):
        print("\n❌ Preprocessing failed!")
        return False
    
    # Step 2: Train model
    print("\n\n🤖 STEP 2: Train SE-MSCNN with Real SPO2")
    print("-" * 70)
    
    pkl_path = Path(__file__).parent / "code_baseline" / "ucd_with_real_spo2.pkl"
    
    if not pkl_path.exists():
        print(f"\n❌ ERROR: Preprocessing didn't create pickle file!")
        print(f"   Expected: {pkl_path}")
        return False
    
    print("✓ Preprocessed data ready")
    print("  - Multi-scale ECG processing (5-min, 3-min, 1-min windows)")
    print("  - Real SPO2 signal at 8 Hz")
    print("  - SE Attention for multi-modal fusion")
    print("  - 100 epochs with early stopping")
    print("\nThis will take 10-20 minutes...")
    print("(Using GPU if available, CPU otherwise)\n")
    
    if not run_command("SE-MSCNN_with_SPO2.py", "Running Model Training"):
        print("\n❌ Training failed!")
        return False
    
    # Step 3: Results
    print("\n\n✅ STEP 3: Results")
    print("-" * 70)
    
    results_dir = Path(__file__).parent / "code_baseline"
    
    model_file = results_dir / "weights.best.spo2_real.keras"
    predictions_file = results_dir / "SE-MSCNN_with_real_SPO2_predictions.csv"
    metrics_file = results_dir / "SE-MSCNN_with_real_SPO2_info.txt"
    
    print("\n📁 Generated Files:")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024**2)
        print(f"  ✅ Model weights: {model_file.name} ({size_mb:.1f} MB)")
    
    if predictions_file.exists():
        print(f"  ✅ Predictions: {predictions_file.name}")
    
    if metrics_file.exists():
        print(f"  ✅ Metrics: {metrics_file.name}")
        print("\n📊 Performance Metrics:")
        print("  " + "-" * 45)
        with open(metrics_file, 'r') as f:
            for line in f:
                if ':' in line and not line.startswith('='):
                    print(f"  {line.rstrip()}")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print("\n🎯 Next Steps:")
        print("  1. Review the model predictions from CSV file")
        print("  2. Compare accuracy with ECG-only baseline (~89%)")
        print("  3. Check expected improvement: +4-6%")
        print("  4. Model is ready for deployment")
        print("\n📌 Key Files:")
        print("  - weights.best.spo2_real.keras: Trained model (use for predictions)")
        print("  - SE-MSCNN_with_real_SPO2_predictions.csv: Test results")
        print("  - SE-MSCNN_with_real_SPO2_info.txt: Detailed metrics")
    else:
        print("\n" + "=" * 70)
        print("❌ TRAINING FAILED")
        print("=" * 70)
        print("\n⚠️  Common Issues:")
        print("  1. UCD dataset not downloaded - check path:")
        print(f"     C:\\Users\\siddh\\Downloads\\st-vincents-university-hospital-...")
        print("  2. Missing required Python packages - install with:")
        print("     pip install tensorflow scipy scikit-learn pandas numpy mne biosppy")
        print("  3. Insufficient disk space - need ~10+ GB free")
        print("  4. GPU issues - try forcing CPU by modifying SE-MSCNN_with_SPO2.py:")
        print("     os.environ['CUDA_VISIBLE_DEVICES'] = ''")
    
    sys.exit(0 if success else 1)
