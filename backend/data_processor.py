import pandas as pd


class DataProcessor:
    @staticmethod
    def load_raw(file_path):
        """
        Minimal raw loader — just reads CSV and cleans column names.
        Each model's train function in ml_engine.py does its own
        preprocessing exactly as written in code.md.
        """
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.replace('\n', '').str.strip()
        return data
