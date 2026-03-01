from docx import Document

document = Document()

document.add_heading('Model Evaluation Report', 0)

document.add_paragraph('Based on the analysis scripts and model weights provided in this repository, here is the performance breakdown of all the evaluated Sleep Apnea detection models.')

# ---- Original Models ----
document.add_heading('1. Baseline SE-MSCNN (Original)', level=2)
document.add_paragraph('Model Reference: SE-MSCNN_predictions.csv', style='List Bullet')
document.add_paragraph('Accuracy: 89.85%', style='List Bullet')
document.add_paragraph('Sensitivity: 86.02%', style='List Bullet')
document.add_paragraph('Specificity: 92.22%', style='List Bullet')
document.add_paragraph('Description: The baseline ECG-only multi-scale CNN model with SE attention.', style='List Bullet')

document.add_heading('2. SPO2 Integration (Initial/Broken)', level=2)
document.add_paragraph('Model Reference: SE-MSCNN_robust_v2_predictions.csv', style='List Bullet')
document.add_paragraph('Accuracy: 68.74%', style='List Bullet')
document.add_paragraph('Sensitivity: 7.49%', style='List Bullet')
document.add_paragraph('Specificity: 91.61%', style='List Bullet')
document.add_paragraph('Description: This model attempted to add SPO2 features but suffered from terrible sensitivity.', style='List Bullet')

document.add_heading('3. Improved Baseline Version', level=2)
document.add_paragraph('Model Reference: SE-MSCNN_improved_baseline.csv (50 sample subset)', style='List Bullet')
document.add_paragraph('Accuracy: 100.0%', style='List Bullet')
document.add_paragraph('Sensitivity: 100.0%', style='List Bullet')
document.add_paragraph('Specificity: 100.0%', style='List Bullet')
document.add_paragraph('Description: Improved model on a very small diagnostic subset of 50 samples.', style='List Bullet')

document.add_heading('4. Final Aggressive Configuration (Real SPO2)', level=2)
document.add_paragraph('Model Reference: weights.final_aggressive.keras', style='List Bullet')
document.add_paragraph('Performance at Threshold 0.27 (Recommended):', style='List Bullet')
document.add_paragraph('  - Accuracy: 54.49%  |  Sensitivity: 85.46%  |  Specificity: 42.93%', style='List Bullet')

# ---- NEW Improved Model ----
document.add_heading('5. SE-MSCNN v2 Improved (PyTorch + XGBoost Ensemble) [NEW]', level=2)
document.add_paragraph('Model Reference: SE_MSCNN_v2_predictions.csv / weights.v2_improved.pt', style='List Bullet')
document.add_paragraph('Framework: PyTorch 2.10 + XGBoost', style='List Bullet')
document.add_paragraph('Architecture: Deeper residual Conv1D branches + BatchNorm + SE Attention + Focal Loss + Cosine Annealing LR + Data Augmentation', style='List Bullet')
document.add_paragraph('Training: 10 epochs, batch size 32, AdamW optimizer', style='List Bullet')
document.add_paragraph('Dataset: 13,494 train / 3,374 val / 17,075 test segments (PhysioNet Apnea-ECG)', style='List Bullet')

document.add_heading('CNN Only Results:', level=3)
document.add_paragraph('Accuracy: 87.02%', style='List Bullet')
document.add_paragraph('Sensitivity: 78.34%', style='List Bullet')
document.add_paragraph('Specificity: 92.38%', style='List Bullet')
document.add_paragraph('F1-score: 0.8217', style='List Bullet')
document.add_paragraph('AUC-ROC: 0.9430', style='List Bullet')

document.add_heading('XGBoost Only Results:', level=3)
document.add_paragraph('Accuracy: 87.81%', style='List Bullet')
document.add_paragraph('Sensitivity: 86.87%', style='List Bullet')
document.add_paragraph('Specificity: 88.39%', style='List Bullet')
document.add_paragraph('F1-score: 0.8447', style='List Bullet')
document.add_paragraph('AUC-ROC: 0.9512', style='List Bullet')

document.add_heading('Ensemble (CNN + XGBoost) Results:', level=3)
document.add_paragraph('Accuracy: 87.84%', style='List Bullet')
document.add_paragraph('Sensitivity: 85.62%', style='List Bullet')
document.add_paragraph('Specificity: 89.20%', style='List Bullet')
document.add_paragraph('F1-score: 0.8431', style='List Bullet')
document.add_paragraph('AUC-ROC: 0.9503', style='List Bullet')

document.add_heading('Notes on Reaching 95% Target', level=2)
document.add_paragraph(
    'The model was trained for only 10 epochs to save time. '
    'Validation accuracy was still climbing at 91.97% when training ended. '
    'To reach the 95% target, increase EPOCHS to 50-100 in SE_MSCNN_v2_improved.py and re-run. '
    'The preprocessed data is cached in preprocessed_data.pkl, so re-running will skip the 1.5-hour preprocessing phase and start training immediately.'
)

document.save('model_evaluation_report.docx')
print("Successfully generated model_evaluation_report.docx")
