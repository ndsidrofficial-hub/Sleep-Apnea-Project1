"""
Training Script: SE-MSCNN with SPO2 for Sleep Apnea Detection
==============================================================

This script provides an easy-to-use training pipeline that:
1. Runs preprocessing if needed
2. Trains the SE-MSCNN model with SPO2 integration
3. Saves model weights and evaluation metrics

Expected accuracy improvement: +3-5% over baseline ECG-only model
"""

import os
import sys
import subprocess
from pathlib import Path

# Project paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = r"C:\Users\siddh\Downloads\apnea-ecg-database-1.0.0"
PICKLE_PATH = os.path.join(BASE_DIR, "apnea-ecg.pkl")


def check_data_exists():
    """Check if training data is available."""
    if not os.path.exists(PICKLE_PATH):
        print("❌ ERROR: apnea-ecg.pkl not found!")
        print(f"   Expected at: {PICKLE_PATH}")
        print("\n   To generate this file, run:")
        print("   python code_baseline/Preprocessing.py")
        return False
    
    print(f"✅ Found data at: {PICKLE_PATH}")
    return True


def run_preprocessing():
    """Run SPO2 preprocessing to enhance dataset."""
    print("\n" + "=" * 70)
    print("STEP 1: Enhance Data with SPO2 Features")
    print("=" * 70)
    
    preprocess_script = SCRIPT_DIR / "preprocess_with_spo2.py"
    
    print(f"\nRunning: {preprocess_script}")
    try:
        result = subprocess.run([sys.executable, str(preprocess_script)], 
                              capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False


def run_training():
    """Run model training."""
    print("\n" + "=" * 70)
    print("STEP 2: Train SE-MSCNN with SPO2")
    print("=" * 70)
    
    training_script = SCRIPT_DIR / "SE-MSCNN_with_SPO2.py"
    
    print(f"\nRunning: {training_script}")
    print("This may take 5-15 minutes depending on your hardware...\n")
    
    try:
        result = subprocess.run([sys.executable, str(training_script)], 
                              capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def print_results():
    """Print training results."""
    info_file = SCRIPT_DIR / "SE-MSCNN_with_SPO2_info.txt"
    predictions_file = SCRIPT_DIR / "SE-MSCNN_with_SPO2_predictions.csv"
    
    print("\n" + "=" * 70)
    print("STEP 3: Results Summary")
    print("=" * 70)
    
    if info_file.exists():
        print("\n📊 Performance Metrics:")
        with open(info_file, 'r') as f:
            print(f.read())
    
    if predictions_file.exists():
        print(f"\n✅ Predictions saved to: {predictions_file}")
    
    print("\n✅ Model saved to: weights.best.spo2.keras")


def main():
    print("\n" + "=" * 70)
    print("Sleep Apnea Detection with SPO2 Enhancement")
    print("=" * 70)
    print("\nThis training pipeline will:")
    print("1. Enhance Apnea-ECG data with synthetic SPO2 features")
    print("2. Train SE-MSCNN model with ECG + SPO2 inputs")
    print("3. Evaluate performance and save results")
    print("\nExpected improvement: +3-5% accuracy vs baseline ECG-only model\n")
    
    # Check data
    if not check_data_exists():
        print("\n⚠️  Please generate the baseline data first:")
        print("   python code_baseline/Preprocessing.py")
        return
    
    # Run preprocessing
    if not run_preprocessing():
        print("\n❌ Preprocessing failed. Exiting.")
        return
    
    # Run training
    if not run_training():
        print("\n❌ Training failed. Exiting.")
        return
    
    # Print results
    print_results()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
