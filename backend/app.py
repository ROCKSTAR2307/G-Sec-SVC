from flask import Flask, request, jsonify
from flask_cors import CORS
from .config import Config
from .ml_engine import MLEngine
import os

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

VALID_MODELS = ['linear_regression', 'xgboost', 'arima', 'lstm']
VALID_BOND_TYPES = ['3yr', '10yr']

MODEL_TRAIN_MAP = {
    'linear_regression': 'train_linear_regression',
    'xgboost':           'train_xgboost',
    'arima':             'train_arima',
    'lstm':              'train_lstm',
}


def make_fresh_engine(bond_type):
    """
    Creates a brand new MLEngine every single call.
    No caching — always loads CSV fresh and re-trains from scratch.
    """
    csv_filename = 'p_merged_data_10.csv' if bond_type == '10yr' else 'p_merged_data_3.csv'
    data_path = os.path.join(os.path.dirname(__file__), '..', csv_filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    engine = MLEngine(data_path)
    engine.load_data(bond_type)
    return engine


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


# ---------------------------------------------------------------------------
# /api/compute — Primary endpoint used by the frontend Prediction screen
#
# Every call is fully stateless:
#   1. Loads CSV fresh
#   2. Trains the requested model
#   3. Returns metrics + raw chart data (actual vs predicted + dates)
#   4. Discards everything — nothing stored in memory
#
# POST body:
#   { "model": "linear_regression", "bond_type": "3yr" }
#
# Response:
#   {
#     "model": "linear_regression",
#     "bond_type": "3yr",
#     "metrics": { "mape": 0.011, "mae": 0.08, "mse": 0.007, "r2": 0.77 },
#     "chart_data": {
#       "dates":     ["2022-01-01", ...],
#       "actual":    [98.2, 98.5, ...],
#       "predicted": [98.1, 98.6, ...]
#     }
#   }
# ---------------------------------------------------------------------------
@app.route('/api/compute', methods=['POST'])
def compute():
    try:
        body = request.json or {}
        model_name = body.get('model', '').strip()
        bond_type  = body.get('bond_type', '').strip()

        # --- Validate ---
        if model_name not in VALID_MODELS:
            return jsonify({'error': f'Invalid model. Choose one of: {VALID_MODELS}'}), 400
        if bond_type not in VALID_BOND_TYPES:
            return jsonify({'error': f'Invalid bond_type. Choose one of: {VALID_BOND_TYPES}'}), 400

        # --- Fresh engine every time (no cache) ---
        engine = make_fresh_engine(bond_type)

        # --- Train the requested model ---
        train_fn = getattr(engine, MODEL_TRAIN_MAP[model_name])
        train_fn()

        # --- Extract results ---
        model_data = engine.models[model_name]
        metrics    = model_data['metrics']

        return jsonify({
            'model':     model_name,
            'bond_type': bond_type,
            'metrics': {
                'mape': round(float(metrics['mape']), 6),
                'mae':  round(float(metrics['mae']),  6),
                'mse':  round(float(metrics['mse']),  6),
                'r2':   round(float(metrics['r2']),   6)
            },
            'chart_data': {
                'dates':     [str(d) for d in model_data['dates']],
                'actual':    model_data['y_test'].tolist(),
                'predicted': model_data['y_pred'].tolist()
            }
        }), 200

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
