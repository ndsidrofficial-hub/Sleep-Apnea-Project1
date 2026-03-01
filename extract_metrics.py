import pandas as pd
from compare_models import ResultsComparer

comparer = ResultsComparer()

try:
    comparer.load_results('Baseline', 'SE-MSCNN_predictions.csv')
except:
    pass
try:
    comparer.load_results('SPO2_Broken', 'SE-MSCNN_robust_v2_predictions.csv')
except:
    pass
try:
    comparer.load_results('Improved', 'SE-MSCNN_improved_baseline.csv')
except:
    pass

df = comparer.generate_comparison_table()
df.to_csv('metrics_output.csv', index=False)
print('DONE')
