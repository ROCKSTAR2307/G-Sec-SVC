import sys
sys.path.append('/Users/tshankarsai/FINAL-YEAR/G-Sec-Backend-SVC')
from backend.ml_engine import MLEngine
engine = MLEngine('/Users/tshankarsai/FINAL-YEAR/G-Sec-Backend-SVC/p_merged_data_3.csv')
engine.load_data('3yr')
metrics, preds = engine.train_lstm()
print(metrics)
