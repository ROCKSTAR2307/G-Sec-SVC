from flask import Flask, request, jsonify
from flask_cors import CORS
from .config import Config
import os
import threading
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

VALID_MODELS     = ['linear_regression', 'xgboost', 'arima', 'lstm']
VALID_BOND_TYPES = ['3yr', '10yr']

MODEL_TRAIN_MAP = {
    'linear_regression': 'train_linear_regression',
    'xgboost':           'train_xgboost',
    'arima':             'train_arima',
    'lstm':              'train_lstm',
}

# ---------------------------------------------------------------------------
# ONE-TIME MODEL CACHE  (trained once per Gunicorn worker at startup)
# Key: (bond_type, model_name) → engine with that model already trained
# ---------------------------------------------------------------------------
_cache = {}               # { (bond_type, model_name): engine }
_cache_lock = threading.Lock()
_inflight = {}            # { (bond_type, model_name): threading.Event }
_startup_warm_started = False

def _csv_path(bond_type):
    fn = 'p_merged_data_10.csv' if bond_type == '10yr' else 'p_merged_data_3.csv'
    return os.path.join(os.path.dirname(__file__), '..', fn)

def get_engine(bond_type, model_name):
    """
    Returns a cached MLEngine that has already trained `model_name`.
    Trains lazily on first call; subsequent calls return cached result instantly.
    """
    key = (bond_type, model_name)
    wait_event = None
    should_train = False

    with _cache_lock:
        if key in _cache:
            return _cache[key]
        if key in _inflight:
            wait_event = _inflight[key]
        else:
            wait_event = threading.Event()
            _inflight[key] = wait_event
            should_train = True

    if not should_train:
        wait_event.wait()
        with _cache_lock:
            if key in _cache:
                return _cache[key]
        raise RuntimeError(f"Model warm-up failed for {bond_type}/{model_name}")

    try:
        path = _csv_path(bond_type)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        # Lazy import keeps startup fast so the server can bind PORT early.
        from .ml_engine import MLEngine
        engine = MLEngine(path)
        engine.load_data(bond_type)
        train_fn = getattr(engine, MODEL_TRAIN_MAP[model_name])
        train_fn()

        with _cache_lock:
            _cache[key] = engine
            done = _inflight.pop(key, None)
            if done:
                done.set()
        return engine
    except Exception:
        with _cache_lock:
            done = _inflight.pop(key, None)
            if done:
                done.set()
        raise


def warm_all_models():
    """Warm all 8 model/bond combinations once per worker process."""
    for bond_type in VALID_BOND_TYPES:
        for model_name in VALID_MODELS:
            try:
                get_engine(bond_type, model_name)
            except Exception as exc:
                app.logger.warning("Warm-up failed for %s/%s: %s", bond_type, model_name, exc)


def _ensure_startup_warm():
    """Start one background warm-up pass once per worker."""
    global _startup_warm_started
    with _cache_lock:
        if _startup_warm_started:
            return
        _startup_warm_started = True

    threading.Thread(target=warm_all_models, daemon=True).start()


# Trigger warm-up as soon as this worker imports the app module.
_ensure_startup_warm()


def get_raw_df(bond_type):
    """Load the CSV and parse dates — used for feature lookup."""
    path = _csv_path(bond_type)
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace('\n', '').str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    _ensure_startup_warm()
    return jsonify({'status': 'healthy'}), 200


# ---------------------------------------------------------------------------
# /api/compute — Same front-end endpoint, now uses cache
#   POST { "model": "linear_regression", "bond_type": "3yr" }
# ---------------------------------------------------------------------------
@app.route('/api/compute', methods=['POST'])
def compute():
    try:
        _ensure_startup_warm()
        body       = request.json or {}
        model_name = body.get('model', '').strip()
        bond_type  = body.get('bond_type', '').strip()

        if model_name not in VALID_MODELS:
            return jsonify({'error': f'Invalid model. Choose one of: {VALID_MODELS}'}), 400
        if bond_type not in VALID_BOND_TYPES:
            return jsonify({'error': f'Invalid bond_type. Choose one of: {VALID_BOND_TYPES}'}), 400

        engine     = get_engine(bond_type, model_name)
        model_data = engine.models[model_name]
        metrics    = model_data['metrics']

        return jsonify({
            'model':      model_name,
            'bond_type':  bond_type,
            'metrics': {
                'mape': round(float(metrics['mape']), 6),
                'mae':  round(float(metrics['mae']),  6),
                'mse':  round(float(metrics['mse']),  6),
                'r2':   round(float(metrics['r2']),   6),
            },
            'chart_data': {
                'dates':     [str(d) for d in model_data['dates']],
                'actual':    model_data['y_test'].tolist(),
                'predicted': model_data['y_pred'].tolist(),
            }
        }), 200

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# /api/dates/<bond_type>
#   Returns the min/max dates in the TEST split (last 20%) of the dataset.
# ---------------------------------------------------------------------------
@app.route('/api/dates/<bond_type>', methods=['GET'])
@app.route('/api/date-range', methods=['GET'])  # backward-compatible alias
def date_range(bond_type=None):
    try:
        _ensure_startup_warm()
        if bond_type is None:
            bond_type = request.args.get('bond_type', '3yr').strip()
        else:
            bond_type = bond_type.strip()
        if bond_type not in VALID_BOND_TYPES:
            return jsonify({'error': 'Invalid bond_type'}), 400

        df = get_raw_df(bond_type)
        test_start = int(len(df) * 0.8)
        test_df    = df.iloc[test_start:]

        return jsonify({
            'bond_type': bond_type,
            'min_date':  test_df['Date'].min().strftime('%Y-%m-%d'),
            'max_date':  test_df['Date'].max().strftime('%Y-%m-%d'),
            'dates':     [d.strftime('%Y-%m-%d') for d in test_df['Date'].dropna()],
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# /api/features?date=YYYY-MM-DD&bond_type=3yr
#   Returns all numeric feature columns for the requested date row.
# ---------------------------------------------------------------------------
@app.route('/api/features', methods=['GET'])
def features():
    try:
        _ensure_startup_warm()
        date_str  = request.args.get('date', '').strip()
        bond_type = request.args.get('bond_type', '3yr').strip()

        if not date_str:
            return jsonify({'error': 'date param required'}), 400
        if bond_type not in VALID_BOND_TYPES:
            return jsonify({'error': 'Invalid bond_type'}), 400

        target_date = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(target_date):
            return jsonify({'error': 'Invalid date format'}), 400

        df = get_raw_df(bond_type)

        # Exact match first, then nearest available date
        exact = df[df['Date'] == target_date]
        if exact.empty:
            idx = (df['Date'] - target_date).abs().idxmin()
            row = df.iloc[idx]
        else:
            row = exact.iloc[0]

        # Return numeric columns only (exclude Date + target)
        target_col = 'Price' if bond_type == '10yr' else 'Close'
        excluded   = {'Date', target_col}
        result     = {}
        for col in df.columns:
            if col in excluded:
                continue
            val = row[col]
            try:
                val = float(str(val).replace('%', '').replace(',', ''))
                result[col] = round(val, 6)
            except Exception:
                pass   # skip non-numeric columns

        return jsonify({
            'date':       row['Date'].strftime('%Y-%m-%d'),
            'bond_type':  bond_type,
            'target_col': target_col,
            'actual':     float(str(row[target_col]).replace('%', '').replace(',', '')),
            'features':   result,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# /api/predict-single
#   Uses the cached trained model to predict for a single date's feature row.
#   POST { "date": "2023-05-10", "bond_type": "3yr", "model": "xgboost" }
# ---------------------------------------------------------------------------
@app.route('/api/predict-single', methods=['POST'])
def predict_single():
    try:
        _ensure_startup_warm()
        body       = request.json or {}
        date_str   = body.get('date', '').strip()
        bond_type  = body.get('bond_type', '3yr').strip()
        model_name = body.get('model', 'xgboost').strip()

        if model_name not in VALID_MODELS:
            return jsonify({'error': f'Invalid model'}), 400
        if bond_type not in VALID_BOND_TYPES:
            return jsonify({'error': 'Invalid bond_type'}), 400

        target_date = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(target_date):
            return jsonify({'error': 'Invalid date format'}), 400

        key = (bond_type, model_name)
        with _cache_lock:
            pretrained_before = key in _cache

        # Get cached engine (trains lazily the first time)
        engine = get_engine(bond_type, model_name)

        # Find this date in the test split predictions
        model_data = engine.models[model_name]
        dates      = pd.to_datetime(model_data['dates'])
        y_test     = model_data['y_test']
        y_pred     = model_data['y_pred']

        # Find closest date in the test set
        diffs = abs(dates - target_date)
        idx   = diffs.argmin()
        matched_date = dates[idx].strftime('%Y-%m-%d')

        actual    = float(y_test[idx])
        predicted = float(y_pred[idx])
        error     = abs(actual - predicted)
        pct_error = error / actual * 100 if actual != 0 else 0
        metrics   = model_data.get('metrics', {})
        with _cache_lock:
            pretrained_after = key in _cache

        return jsonify({
            'date':       matched_date,
            'bond_type':  bond_type,
            'model':      model_name,
            'actual':     round(actual, 6),
            'predicted':  round(predicted, 6),
            'error':      round(error, 6),
            'pct_error':  round(pct_error, 4),
            'pretrained_before': pretrained_before,
            'pretrained_after': pretrained_after,
            'metrics': {
                'mape': round(float(metrics.get('mape', 0.0)), 6),
                'mae':  round(float(metrics.get('mae', 0.0)),  6),
                'mse':  round(float(metrics.get('mse', 0.0)),  6),
                'r2':   round(float(metrics.get('r2', 0.0)),   6),
            },
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
