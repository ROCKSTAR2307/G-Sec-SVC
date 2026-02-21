import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .data_processor import DataProcessor
from .models import LinearRegressionModel, ARIMAModel, XGBoostModel, LSTMModel


class MLEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.bond_type = None
        self.models = {}
        self.predictions = {}

    def load_data(self, bond_type='10yr'):
        self.bond_type = bond_type
        self.data = DataProcessor.load_and_clean_data(self.data_path)
        return self.data

    def train_linear_regression(self):
        X, y, dates, features = DataProcessor.prepare_features(self.data, self.bond_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = LinearRegressionModel()
        model.train(X_train, y_train)
        y_pred, metrics = model.evaluate(X_test, y_test)

        self.models['linear_regression'] = {
            'model': model,
            'metrics': metrics,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'dates': dates[len(X_train):]
        }

        return metrics, y_pred

    def train_xgboost(self):
        X, y, dates, features = DataProcessor.prepare_features(self.data, self.bond_type)
        X_scaled, y_scaled, feature_scaler, target_scaler = DataProcessor.scale_features(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

        model = XGBoostModel()
        model.model.fit(X_train, y_train)
        y_pred_scaled = model.predict(X_test)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        model.metrics = {
            'mape': np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)),
            'mae': np.mean(np.abs(y_test_orig - y_pred)),
            'mse': np.mean((y_test_orig - y_pred) ** 2),
            'r2': 1 - np.sum((y_test_orig - y_pred) ** 2) / np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
        }

        self.models['xgboost'] = {
            'model': model,
            'metrics': model.metrics,
            'X_test': X_test,
            'y_test': y_test_orig,
            'y_pred': y_pred,
            'dates': dates[len(X_train):],
            'target_scaler': target_scaler
        }

        return model.metrics, y_pred

    def train_arima(self):
        X, y, dates, features = DataProcessor.prepare_features(self.data, self.bond_type)
        train_size = int(len(y) * 0.8)
        y_train, y_test = y[:train_size], y[train_size:]

        model = ARIMAModel(order=(1, 1, 1))
        model.train(y_train)
        y_pred, metrics = model.evaluate(y_train, y_test)

        self.models['arima'] = {
            'model': model,
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'dates': dates[train_size:]
        }

        return metrics, y_pred

    def train_lstm(self):
        X, y, dates, features = DataProcessor.prepare_features(self.data, self.bond_type)

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        scaled_data = np.hstack((y_scaled.reshape(-1, 1), X_scaled))

        seq_length = 90
        X_seq, y_seq = self._create_sequences(scaled_data, seq_length)

        split_idx = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        model = LSTMModel(seq_length=seq_length)
        model.feature_scaler = feature_scaler
        model.target_scaler = target_scaler
        model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        model.metrics = {
            'mape': np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)),
            'mae': np.mean(np.abs(y_test_orig - y_pred)),
            'mse': np.mean((y_test_orig - y_pred) ** 2),
            'r2': 1 - np.sum((y_test_orig - y_pred) ** 2) / np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)
        }

        test_dates_start = seq_length + split_idx
        test_dates = dates[test_dates_start:test_dates_start + len(y_pred)]

        self.models['lstm'] = {
            'model': model,
            'metrics': model.metrics,
            'y_test': y_test_orig,
            'y_pred': y_pred,
            'dates': test_dates,
            'target_scaler': target_scaler
        }

        return model.metrics, y_pred

    def _create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, 1:])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)

    def get_model_comparison(self):
        comparison = {}
        for model_name, model_data in self.models.items():
            metrics = model_data['metrics']
            comparison[model_name] = {
                'mape': round(metrics['mape'], 4),
                'mae': round(metrics['mae'], 4),
                'mse': round(metrics['mse'], 4),
                'r2': round(metrics['r2'], 4)
            }
        return comparison

    def generate_prediction_chart(self, model_name, output_format='png'):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model_data = self.models[model_name]
        dates = model_data['dates']
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']

        plt.figure(figsize=(14, 6))
        plt.plot(dates, y_test, label='Actual', color='blue', linewidth=2)
        plt.plot(dates, y_pred, label='Predicted', color='red', linestyle='--', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Bond Price')
        plt.title(f'{model_name.upper()}: Actual vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_format == 'png':
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100)
            img_buffer.seek(0)
            plt.close()
            return img_buffer.getvalue()
        elif output_format == 'base64':
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"

    def export_predictions_csv(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model_data = self.models[model_name]
        df = pd.DataFrame({
            'Date': model_data['dates'],
            'Actual': model_data['y_test'],
            'Predicted': model_data['y_pred'],
            'Error': model_data['y_test'] - model_data['y_pred']
        })

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

    def get_summary_metrics(self):
        summary = {
            'bond_type': self.bond_type,
            'data_points': len(self.data),
            'model_metrics': self.get_model_comparison()
        }
        return summary
