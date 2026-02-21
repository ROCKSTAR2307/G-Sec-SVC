import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    @staticmethod
    def load_and_clean_data(file_path):
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.replace('\n', '').str.strip()

        percentage_columns = [
            'Change %', 'Daily Return (%)', 'Forward Premia of US$ 1-month (%)',
            'Forward Premia of US$ 3-month (%)', 'Forward Premia of US$ 6-month (%)'
        ]

        for col in percentage_columns:
            if col in data.columns:
                data[col] = data[col].replace('-', '0')
                data[col] = data[col].astype(str).str.replace('%', '')
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(0)

        for col in data.columns:
            if col not in percentage_columns and col != 'Date':
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(0)

        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
        data = data.sort_values('Date')

        return data

    @staticmethod
    def prepare_features(data, bond_type='10yr'):
        if bond_type == '10yr':
            target_column = 'Price'
        else:
            target_column = 'Close'

        feature_columns = [
            'Change %', 'Daily Return (%)', 'Volatility (7D)', 'SMA (7D)',
            'Forward Premia of US$ 1-month (%)', 'Forward Premia of US$ 3-month (%)',
            'Forward Premia of US$ 6-month (%)', 'Reverse Repo Rate (%)',
            'Marginal Standing Facility (MSF) Rate (%)', 'Bank Rate (%)',
            '91-Day Treasury Bill (Primary) Yield (%)',
            '182-Day Treasury Bill (Primary) Yield (%)',
            '364-Day Treasury Bill (Primary) Yield (%)',
            'Cash Reserve Ratio (%)', 'Statutory Liquidity Ratio (%)',
            'Policy Repo Rate (%)', 'Foreign Exchange Reserves (US $ Million)'
        ]

        available_features = [col for col in feature_columns if col in data.columns]

        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        X = data[available_features].values
        y = data[target_column].values

        return X, y, data['Date'].values, available_features

    @staticmethod
    def scale_features(X, y=None):
        feature_scaler = MinMaxScaler()
        X_scaled = feature_scaler.fit_transform(X)

        if y is not None:
            target_scaler = MinMaxScaler()
            y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            return X_scaled, y_scaled, feature_scaler, target_scaler

        return X_scaled, feature_scaler
