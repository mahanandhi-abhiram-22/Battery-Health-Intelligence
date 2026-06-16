# train_rf_model.py

import json, joblib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_processor import load_all_csvs, split_Xy


class RandomForestTrainingPipeline:
    def __init__(self, data_folder="data", models_dir="models", results_dir="results_rf"):
        self.data_folder = Path(data_folder)

        self.models_dir = Path(models_dir); self.models_dir.mkdir(exist_ok=True)
        self.results_dir = Path(results_dir); self.results_dir.mkdir(exist_ok=True)

        self.scaler_X = MinMaxScaler()

    def load_data(self):
        df = load_all_csvs(self.data_folder)
        X, y = split_Xy(df)

        # Only SOH and RUL
        y = y[["SOH", "RUL"]].values

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y, test_size=0.2, random_state=42
        )

        X_train = self.scaler_X.fit_transform(X_train)
        X_test  = self.scaler_X.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def train(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)

        metrics = {}
        labels = ["SOH", "RUL"]

        for i, label in enumerate(labels):
            mae  = mean_absolute_error(self.y_test[:, i], y_pred[:, i])
            rmse = mean_squared_error(self.y_test[:, i], y_pred[:, i]) ** 0.5
            r2   = r2_score(self.y_test[:, i], y_pred[:, i])

            metrics[label] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            }

        with open(self.results_dir / "rf_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("Random Forest Evaluation:")
        print(json.dumps(metrics, indent=2))

    def save(self):
        joblib.dump(self.model, self.models_dir / "rf_model.joblib")
        joblib.dump(self.scaler_X, self.models_dir / "rf_scaler.joblib")
        print("Random Forest model saved to models/rf_model.joblib")


    def run(self):
        self.load_data()
        self.train()
        self.evaluate()
        self.save()


if __name__ == "__main__":
    rf_pipeline = RandomForestTrainingPipeline()
    rf_pipeline.run()
