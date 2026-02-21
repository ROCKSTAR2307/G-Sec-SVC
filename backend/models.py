import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.metrics = {}

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        self.metrics = {
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        return y_pred, self.metrics


class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.metrics = {}

    def train(self, series):
        if isinstance(series, pd.Series):
            series = series.values
        self.model = ARIMA(series, order=self.order).fit()

    def predict(self, steps=20):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.forecast(steps=steps)

    def evaluate(self, train_series, test_series):
        if isinstance(train_series, pd.Series):
            train_series = train_series.values
        if isinstance(test_series, pd.Series):
            test_series = test_series.values

        model = ARIMA(train_series, order=self.order).fit()
        y_pred = model.forecast(steps=len(test_series))

        self.metrics = {
            'mape': mean_absolute_percentage_error(test_series, y_pred),
            'mae': mean_absolute_error(test_series, y_pred),
            'mse': mean_squared_error(test_series, y_pred),
            'r2': r2_score(test_series, y_pred)
        }
        return y_pred, self.metrics


class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42
        )
        self.metrics = {}
        self.target_scaler = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        self.metrics = {
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        return y_pred, self.metrics


class LSTMModel:
    def __init__(self, seq_length=90):
        self.seq_length = seq_length
        self.model = None
        self.metrics = {}
        self.feature_scaler = None
        self.target_scaler = None

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i+self.seq_length, 1:])
            y.append(data[i+self.seq_length, 0])
        return np.array(X), np.array(y)

    def train(self, X_train, y_train, validation_data=None, epochs=100):
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(128, return_sequences=False),
            Dense(1)
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        kwargs = {
            'epochs': epochs,
            'batch_size': 32,
            'callbacks': [early_stop],
            'verbose': 0
        }

        if validation_data is not None:
            kwargs['validation_data'] = validation_data

        self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        self.metrics = {
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        return y_pred, self.metrics
