from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from .config import Config
from .ml_engine import MLEngine
import os
import io

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

ml_engines = {}


def get_ml_engine(bond_type):
    if bond_type not in ml_engines:
        csv_filename = 'p_merged_data_10.csv' if bond_type == '10yr' else 'p_merged_data_3.csv'
        data_path = os.path.join(os.path.dirname(__file__), '..', csv_filename)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        engine = MLEngine(data_path)
        engine.load_data(bond_type)
        ml_engines[bond_type] = engine

    return ml_engines[bond_type]


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/train', methods=['POST'])
def train_models():
    try:
        data = request.json
        bond_type = data.get('bond_type', '10yr')

        if bond_type not in ['3yr', '10yr']:
            return jsonify({'error': 'Invalid bond type'}), 400

        engine = get_ml_engine(bond_type)

        results = {}

        try:
            lr_metrics, _ = engine.train_linear_regression()
            results['linear_regression'] = lr_metrics
        except Exception as e:
            results['linear_regression'] = {'error': str(e)}

        try:
            xgb_metrics, _ = engine.train_xgboost()
            results['xgboost'] = xgb_metrics
        except Exception as e:
            results['xgboost'] = {'error': str(e)}

        try:
            arima_metrics, _ = engine.train_arima()
            results['arima'] = arima_metrics
        except Exception as e:
            results['arima'] = {'error': str(e)}

        try:
            lstm_metrics, _ = engine.train_lstm()
            results['lstm'] = lstm_metrics
        except Exception as e:
            results['lstm'] = {'error': str(e)}

        return jsonify({
            'status': 'success',
            'bond_type': bond_type,
            'models_trained': results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/<bond_type>/<model_name>', methods=['GET'])
def get_predictions(bond_type, model_name):
    try:
        if bond_type not in ['3yr', '10yr']:
            return jsonify({'error': 'Invalid bond type'}), 400

        if model_name not in ['linear_regression', 'xgboost', 'arima', 'lstm']:
            return jsonify({'error': 'Invalid model name'}), 400

        engine = get_ml_engine(bond_type)

        if model_name not in engine.models:
            return jsonify({'error': f'Model {model_name} not trained yet'}), 400

        model_data = engine.models[model_name]
        metrics = model_data['metrics']

        response = {
            'bond_type': bond_type,
            'model': model_name,
            'metrics': {
                'mape': round(metrics['mape'], 4),
                'mae': round(metrics['mae'], 4),
                'mse': round(metrics['mse'], 4),
                'r2': round(metrics['r2'], 4)
            },
            'predictions': {
                'actual': model_data['y_test'].tolist(),
                'predicted': model_data['y_pred'].tolist(),
                'dates': [str(d) for d in model_data['dates']]
            }
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chart/<bond_type>/<model_name>', methods=['GET'])
def get_chart(bond_type, model_name):
    try:
        if bond_type not in ['3yr', '10yr']:
            return jsonify({'error': 'Invalid bond type'}), 400

        engine = get_ml_engine(bond_type)

        if model_name not in engine.models:
            return jsonify({'error': f'Model {model_name} not trained yet'}), 400

        chart_data = engine.generate_prediction_chart(model_name, output_format='base64')

        return jsonify({
            'bond_type': bond_type,
            'model': model_name,
            'chart': chart_data
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<bond_type>/<model_name>', methods=['GET'])
def export_csv(bond_type, model_name):
    try:
        if bond_type not in ['3yr', '10yr']:
            return jsonify({'error': 'Invalid bond type'}), 400

        engine = get_ml_engine(bond_type)

        if model_name not in engine.models:
            return jsonify({'error': f'Model {model_name} not trained yet'}), 400

        csv_data = engine.export_predictions_csv(model_name)

        return send_file(
            io.BytesIO(csv_data.encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{bond_type}_{model_name}_predictions.csv'
        ), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary/<bond_type>', methods=['GET'])
def get_summary(bond_type):
    try:
        if bond_type not in ['3yr', '10yr']:
            return jsonify({'error': 'Invalid bond type'}), 400

        engine = get_ml_engine(bond_type)
        summary = engine.get_summary_metrics()

        return jsonify(summary), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
