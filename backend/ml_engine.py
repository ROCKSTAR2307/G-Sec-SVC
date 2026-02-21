import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# Fix seeds for reproducibility across runs
np.random.seed(42)
tf.random.set_seed(42)


class MLEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.bond_type = None
        self.models = {}

    def load_data(self, bond_type='10yr'):
        self.bond_type = bond_type
        data = pd.read_csv(self.data_path)
        data.columns = data.columns.str.replace('\n', '').str.strip()
        self.raw_data = data
        return data

    # =========================================================================
    # LINEAR REGRESSION
    # Exactly matches code.md:
    #   - Strips % and divides by 100 for percentage columns
    #   - Applies pd.to_numeric on everything (drops Date implicitly)
    #   - 10yr: target=Price, 3yr: target=Close (different feature lists)
    # =========================================================================
    def train_linear_regression(self):
        data = self.raw_data.copy()

        PERCENTAGE_COLS = [
            'Change %', 'Daily Return (%)',
            'Forward Premia of US$ 1-month (%)',
            'Forward Premia of US$ 3-month (%)',
            'Forward Premia of US$ 6-month (%)',
        ]

        # Exact conversion from code.md: strip %, divide by 100
        def convert_percentage_to_float(df, col):
            df[col] = df[col].replace('%', '', regex=True).astype(str)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
            return df

        for col in PERCENTAGE_COLS:
            if col in data.columns:
                data = convert_percentage_to_float(data, col)

        # Convert everything to numeric (Date becomes NaN — not used in features)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.fillna(0)

        if self.bond_type == '10yr':
            target_col = 'Price'
            feature_cols = [
                'Change %', 'Daily Return (%)', 'Volatility (7D)', 'SMA (7D)',
                'Forward Premia of US$ 1-month (%)', 'Forward Premia of US$ 3-month (%)',
                'Forward Premia of US$ 6-month (%)',
                'Reverse Repo Rate (%)', 'Marginal Standing Facility (MSF) Rate (%)',
                'Bank Rate (%)', '91-Day Treasury Bill (Primary) Yield (%)',
                '182-Day Treasury Bill (Primary) Yield (%)',
                '364-Day Treasury Bill (Primary) Yield (%)',
                'Cash Reserve Ratio (%)', 'Statutory Liquidity Ratio (%)',
                'Policy Repo Rate (%)', 'Foreign Exchange Reserves (US $ Million)',
            ]
        else:  # 3yr
            target_col = 'Close'
            feature_cols = [
                'Change %', 'Daily Return (%)', 'Volatility (7D)', 'SMA (7D)', 'SMA (30D)',
                'Forward Premia of US$ 1-month (%)', 'Forward Premia of US$ 3-month (%)',
                'Forward Premia of US$ 6-month (%)',
                'Reverse Repo Rate (%)', 'Marginal Standing Facility (MSF) Rate (%)',
                'Bank Rate (%)', '91-Day Treasury Bill (Primary) Yield (%)',
                '182-Day Treasury Bill (Primary) Yield (%)',
                'Cash Reserve Ratio (%)', 'Statutory Liquidity Ratio (%)',
                'Policy Repo Rate (%)', 'Foreign Exchange Reserves (US $ Million)',
            ]

        available = [c for c in feature_cols if c in data.columns]
        X = data[available].values
        y = data[target_col].values

        # Get dates for chart_data (from raw before numeric conversion)
        raw = self.raw_data.copy()
        raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')
        raw = raw.sort_values('Date').reset_index(drop=True)
        all_dates = raw['Date'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'mape': float(mean_absolute_percentage_error(y_test, y_pred)),
            'mae':  float(mean_absolute_error(y_test, y_pred)),
            'mse':  float(mean_squared_error(y_test, y_pred)),
            'r2':   float(r2_score(y_test, y_pred)),
        }

        self.models['linear_regression'] = {
            'metrics': metrics,
            'y_test':  y_test,
            'y_pred':  y_pred,
            'dates':   all_dates[len(X_train):],
        }
        return metrics, y_pred

    # =========================================================================
    # ARIMA
    # Exactly matches code.md:
    #   - Checks stationarity; differences if p-value > 0.05
    #   - 10yr: uses auto_arima to find (p,d,q) → fits ARIMA(train, (p,d,q))
    #   - 3yr:  hardcoded ARIMA(train, order=(3,1,4)) — as in code.md Step 6
    #   - Evaluation: 80% train, forecast 20% test
    # =========================================================================
    def train_arima(self):
        raw = self.raw_data.copy()
        raw['Date'] = pd.to_datetime(raw['Date'], dayfirst=True, errors='coerce')
        raw = raw.sort_values('Date').reset_index(drop=True)

        target_col = 'Price' if self.bond_type == '10yr' else 'Close'
        series = pd.to_numeric(raw[target_col], errors='coerce').ffill().fillna(0)
        dates  = raw['Date'].values

        # Check stationarity (ADF test)
        def is_stationary(s):
            return adfuller(s.dropna())[1] <= 0.05

        if not is_stationary(series):
            series_diff = series.diff().dropna()
        else:
            series_diff = series

        train_size = int(len(series) * 0.8)
        train = series[:train_size]
        test  = series[train_size:]

        if self.bond_type == '10yr':
            # Use auto_arima to find best ARIMA params (same as code.md)
            try:
                from pmdarima import auto_arima as _auto_arima
                stepwise = _auto_arima(
                    series_diff,
                    start_p=1, start_q=1, max_p=5, max_q=5,
                    m=1, seasonal=False, trace=False,
                    stepwise=True, suppress_warnings=True,
                )
                p, d, q = stepwise.order
            except Exception:
                p, d, q = 1, 1, 1  # safe fallback
            model_fit = ARIMA(train, order=(p, d, q)).fit()
        else:
            # 3yr: hardcoded (3, 1, 4) as written in code.md evaluation step
            model_fit = ARIMA(train, order=(3, 1, 4)).fit()

        predictions = model_fit.forecast(steps=len(test))

        metrics = {
            'mape': float(mean_absolute_percentage_error(test, predictions)),
            'mae':  float(mean_absolute_error(test, predictions)),
            'mse':  float(mean_squared_error(test, predictions)),
            'r2':   float(r2_score(test, predictions)),
        }

        self.models['arima'] = {
            'metrics': metrics,
            'y_test':  test.values,
            'y_pred':  np.array(predictions),
            'dates':   dates[train_size:],
        }
        return metrics, predictions

    # =========================================================================
    # XGBOOST
    # Exactly matches code.md:
    #   - ffill missing values
    #   - Scale ALL features (except Date & target) with MinMaxScaler
    #   - y = target column UNSCALED (model trains on scaled X, unscaled y)
    #   - Separate target_scaler for inverse-transforming predictions AND actuals
    # =========================================================================
    def train_xgboost(self):
        data = self.raw_data.copy()
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
        data = data.sort_values('Date').reset_index(drop=True)
        data = data.ffill()

        target_col = 'Price' if self.bond_type == '10yr' else 'Close'
        dates = data['Date'].values

        PERCENTAGE_COLS = [
            'Change %', 'Daily Return (%)',
            'Forward Premia of US$ 1-month (%)',
            'Forward Premia of US$ 3-month (%)',
            'Forward Premia of US$ 6-month (%)',
        ]

        for col in PERCENTAGE_COLS:
            if col in data.columns:
                data[col] = data[col].replace('-', '0')
                data[col] = data[col].astype(str).str.replace('%', '')
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(0)

        # Convert all other non-Date, non-target columns to numeric
        for col in data.columns:
            if col != 'Date' and col not in PERCENTAGE_COLS and col != target_col:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].ffill()

        # Scale ALL features (drop Date + target)
        feature_df = data.drop(columns=['Date', target_col], errors='ignore')
        feature_df = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(feature_df)
        y = data[target_col].values  # UNSCALED target — exactly as in code.md

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200, learning_rate=0.1,
            max_depth=7, subsample=0.8,
            colsample_bytree=0.9, random_state=42,
        )
        model.fit(X_train, y_train)
        predicted_yields = model.predict(X_test)

        # Separate target_scaler — inverse transform both predicted AND actual
        # (exactly as in code.md)
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(data[[target_col]])

        y_pred_rescaled = target_scaler.inverse_transform(
            predicted_yields.reshape(-1, 1)
        ).flatten()
        y_test_rescaled = target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        metrics = {
            'mape': float(mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)),
            'mae':  float(mean_absolute_error(y_test_rescaled, y_pred_rescaled)),
            'mse':  float(mean_squared_error(y_test_rescaled, y_pred_rescaled)),
            'r2':   float(r2_score(y_test_rescaled, y_pred_rescaled)),
        }

        self.models['xgboost'] = {
            'metrics': metrics,
            'y_test':  y_test_rescaled,
            'y_pred':  y_pred_rescaled,
            'dates':   dates[len(X_train):],
        }
        return metrics, y_pred_rescaled

    # =========================================================================
    # LSTM
    # Exactly matches code.md:
    #   - Adds lag features (Lag_1, Lag_7, Lag_30) and rolling stats (Mean_7, Std_7)
    #   - Separate feature_scaler and target_scaler
    #   - SEQ_LENGTH = 90, epochs = 200, patience = 15, batch = 32, lr = 0.0003
    #   - Returns test-set predictions (inverse scaled)
    # =========================================================================
    def train_lstm(self):
        data = self.raw_data.copy()
        target_col = 'Price' if self.bond_type == '10yr' else 'Close'

        # Parse date — try both formats used in code.md
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
        if data['Date'].isna().all():
            data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

        data = data.sort_values(by='Date').reset_index(drop=True)
        dates_series = data['Date'].copy()  # save BEFORE numeric conversion

        # Handle '-' and convert to numeric (code.md does this to whole df)
        data_num = data.drop(columns=['Date'])
        data_num = data_num.replace('-', np.nan)
        data_num = data_num.apply(pd.to_numeric, errors='coerce')
        data_num = data_num.ffill()

        # Lag features — exactly as code.md
        for lag in [1, 7, 30]:
            data_num[f'{target_col}_Lag_{lag}'] = data_num[target_col].shift(lag)

        # Rolling stats — exactly as code.md
        data_num[f'{target_col}_Rolling_Mean_7'] = data_num[target_col].rolling(window=7).mean()
        data_num[f'{target_col}_Rolling_Std_7']  = data_num[target_col].rolling(window=7).std()

        # Re-attach dates and drop NaN rows introduced by lag/rolling
        data_num['Date'] = dates_series.values
        data_num = data_num.dropna().reset_index(drop=True)
        dates_clean = data_num['Date'].values

        # Feature selection: all columns except Date and target
        selected_features = [
            col for col in data_num.columns
            if col != 'Date' and col != target_col
        ]

        feature_vals = data_num[selected_features].values
        target_vals  = data_num[target_col].values

        feature_scaler = MinMaxScaler()
        target_scaler  = MinMaxScaler()

        scaled_features = feature_scaler.fit_transform(feature_vals)
        scaled_target   = target_scaler.fit_transform(target_vals.reshape(-1, 1))

        # Combine: first column = scaled target, rest = scaled features
        scaled_data = np.hstack((scaled_target, scaled_features))

        SEQ_LENGTH = 90

        def create_sequences(d, seq_len):
            Xs, ys = [], []
            for i in range(len(d) - seq_len):
                Xs.append(d[i:i + seq_len, 1:])   # all features
                ys.append(d[i + seq_len, 0])        # target
            return np.array(Xs), np.array(ys)

        X, y = create_sequences(scaled_data, SEQ_LENGTH)

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # LSTM architecture — exact from code.md
        model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(128, return_sequences=False),
            Dense(1),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.0003),
            loss='mean_squared_error'
        )
        early_stop = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200, batch_size=32,
            callbacks=[early_stop], verbose=0,
        )

        test_preds = model.predict(X_test, verbose=0)

        # Inverse scale — exactly as code.md
        test_preds_inv = target_scaler.inverse_transform(test_preds).flatten()
        y_test_inv     = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        metrics = {
            'mape': float(mean_absolute_percentage_error(y_test_inv, test_preds_inv)),
            'mae':  float(mean_absolute_error(y_test_inv, test_preds_inv)),
            'mse':  float(mean_squared_error(y_test_inv, test_preds_inv)),
            'r2':   float(r2_score(y_test_inv, test_preds_inv)),
        }

        # Test dates: data rows from (split_idx + SEQ_LENGTH) onwards
        test_dates = dates_clean[
            split_idx + SEQ_LENGTH: split_idx + SEQ_LENGTH + len(y_test_inv)
        ]

        self.models['lstm'] = {
            'metrics': metrics,
            'y_test':  y_test_inv,
            'y_pred':  test_preds_inv,
            'dates':   test_dates,
        }
        return metrics, test_preds_inv
