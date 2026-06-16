
import os, json, joblib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# show interactive figures on local machines
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_processor import load_all_csvs, split_Xy

import tensorflow as tf

def soh_monotonic_loss(y_pred, cycle):
    """
    Penalize SOH increasing with cycle.
    """
    # sort by cycle
    idx = tf.argsort(cycle[:, 0])
    soh_sorted = tf.gather(y_pred, idx)

    # SOH(t+1) - SOH(t)
    diff = soh_sorted[1:] - soh_sorted[:-1]

    # penalize positive differences
    return tf.reduce_mean(tf.nn.relu(diff))


def rul_monotonic_loss(y_pred, cycle):
    """
    Penalize RUL increasing with cycle.
    """
    idx = tf.argsort(cycle[:, 0])
    rul_sorted = tf.gather(y_pred, idx)

    diff = rul_sorted[1:] - rul_sorted[:-1]
    return tf.reduce_mean(tf.nn.relu(diff))


class BatteryTrainingPipeline:
    def __init__(self, data_folder="data", models_dir="models", results_dir="results"):
        self.data_folder = Path(data_folder)

        self.models_dir = Path(models_dir);  self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(results_dir); self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir   = self.results_dir / "plots"; self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.model = None
        self.feature_names = None

        self.X_train = self.X_test = self.y_train = self.y_test = None

class PINNModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        # split targets
        y_soh = y[:, 0:1]
        y_rul = y[:, 1:2]

        # extract cycle feature (important)
        cycle_idx = self.cycle_index
        cycle = tf.expand_dims(x[:, cycle_idx], axis=1)

        with tf.GradientTape() as tape:
            pred_soh, pred_rul, pred_rulc = self(x, training=True)

            # data loss
            data_loss = (
                tf.reduce_mean(tf.square(y_soh - pred_soh)) +
                tf.reduce_mean(tf.square(y_rul - pred_rul))
            )

            # physics losses
            loss_soh_phy = soh_monotonic_loss(pred_soh, cycle)
            loss_rul_phy = rul_monotonic_loss(pred_rul, cycle)

            # total loss
            total_loss = data_loss + 0.1 * loss_soh_phy + 0.1 * loss_rul_phy

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "physics_soh_loss": loss_soh_phy,
            "physics_rul_loss": loss_rul_phy,
        }

    # --------------------- Build model ---------------------
    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Dense multi-output regressor for SOH, RUL, RUL_Cycles."""
        inp = Input(shape=(input_dim,), name="input")
        x   = Dense(128, activation="relu")(inp)
        x   = Dropout(0.10)(x)
        x   = Dense(128, activation="relu")(x)

        out_soh  = Dense(1, name="SOH")(x)
        out_rul  = Dense(1, name="RUL")(x)
        out_rulc = Dense(1, name="RUL_Cycles")(x)

        model = PINNModel(inp, [out_soh, out_rul, out_rulc])
        model.compile(
            optimizer="adam"
        )
        return model

    # -------------------- Load data ------------------------
    def load_data(self):
        df = load_all_csvs(self.data_folder)
        X, y = split_Xy(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )

        X_train = self.scaler_X.fit_transform(X_train)
        X_test  = self.scaler_X.transform(X_test)
        y_train = self.scaler_y.fit_transform(y_train)
        y_test  = self.scaler_y.transform(y_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        self.feature_names = list(split_Xy(df)[0].columns)
        return self

    # --------------------- Train ---------------------------
    def build(self):
        if self.X_train is None:
            raise RuntimeError("Call load_data() before build().")
        self.model = self._build_model(self.X_train.shape[1])
        self.model.cycle_index = self.feature_names.index("cycle")
        return self

    def train(self, epochs=30, batch_size=32, patience=5):
        if self.model is None:
            self.build()

        es = EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)


        history = self.model.fit(
            self.X_train,
            [self.y_train[:, 0], self.y_train[:, 1], self.y_train[:, 2]],
            validation_split=0.2,
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=[es],
            verbose=1,
        )

        # Save training loss (no popup)
        plt.figure(figsize=(7, 5))
        plt.plot(history.history["loss"], label="Train Loss", color="blue")
        plt.plot(history.history["val_loss"], label="Val Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        # NO LEGEND
        plt.tight_layout()
        plt.savefig(self.plots_dir / "training_loss.png", dpi=140)
        plt.close()

        return self

    # --------------------- Evaluate ------------------------
    def evaluate_and_plot(self):
        preds = self.model.predict(self.X_test, verbose=0)
        y_pred = np.concatenate(preds, axis=1)
        y_true = self.y_test

        # invert scaling
        y_pred_inv = self.scaler_y.inverse_transform(y_pred)
        y_true_inv = self.scaler_y.inverse_transform(y_true)

        labels = ["SOH", "RUL", "RUL_Cycles"]
        metrics = {}

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for i, lab in enumerate(labels):
            ax = axes[i]

            mae  = float(mean_absolute_error(y_true_inv[:, i], y_pred_inv[:, i]))
            rmse = float(mean_squared_error(y_true_inv[:, i], y_pred_inv[:, i]) ** 0.5)
            r2   = float(r2_score(y_true_inv[:, i], y_pred_inv[:, i]))

            metrics[lab] = {"MAE": mae, "RMSE": rmse, "R2": r2}

            ax.scatter(
                y_true_inv[:, i], y_pred_inv[:, i],
                s=10, alpha=0.5, color="tab:blue"
            )

            lo = float(min(y_true_inv[:, i].min(), y_pred_inv[:, i].min()))
            hi = float(max(y_true_inv[:, i].max(), y_pred_inv[:, i].max()))
            ax.plot([lo, hi], [lo, hi], "r--")  # Ideal line

            ax.set_title(f"{lab} Parity Plot")
            ax.set_xlabel(f"Actual {lab}")
            ax.set_ylabel(f"Predicted {lab}")

            # Metrics box (this stays)
            ax.text(
                0.03, 0.97,
                f"R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}\nAccuracy ≈ {r2*100:.1f}%",
                transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.9),
                fontsize=9,
            )

        # NO LEGEND ANYWHERE

        plt.tight_layout()
        plt.savefig(self.plots_dir / "parity_grid.png", dpi=140)
        plt.show()
        plt.close()

        # save metrics
        with open(self.results_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mean_r2 = float(np.mean([m["R2"] for m in metrics.values()]))
        print(f"[Eval] Mean R²: {mean_r2:.4f}  ≈  Accuracy {mean_r2*100:.1f}%")
        print(json.dumps(metrics, indent=2))

        return metrics

    # --------------------- Save ----------------------------
    def save_model_and_scalers(self):
        self.model.save(self.models_dir / "battery_model.keras")
        joblib.dump(
            {"scaler_X": self.scaler_X, "scaler_y": self.scaler_y, "features": self.feature_names},
            self.models_dir / "scalers.joblib",
        )

    # --------------------- Pipeline ------------------------
    def run_complete_training(self, epochs=30, batch_size=32, patience=5):
        self.load_data()
        self.build()
        self.train(epochs=epochs, batch_size=batch_size, patience=patience)
        metrics = self.evaluate_and_plot()
        self.save_model_and_scalers()
        return metrics

    

if __name__ == "__main__":
    pipeline = BatteryTrainingPipeline(data_folder="data", models_dir="models", results_dir="results")
    results = pipeline.run_complete_training(epochs=30, batch_size=32, patience=5)
    print("\nArtifacts saved to:")
    print("  Models :", Path("models").resolve())
    print("  Results:", Path("results").resolve())
