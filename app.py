# app.py
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from pathlib import Path
import json, joblib, numpy as np
import tensorflow as tf
from train_model import BatteryTrainingPipeline

BASE = Path(__file__).parent.resolve()

DATA_DIR    = BASE / "data"
MODELS_DIR  = BASE / "models"
RESULTS_DIR = BASE / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

app = Flask(__name__, template_folder=str(BASE / "templates"), static_folder=str(BASE / "static"))
CORS(app)

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

pipeline = BatteryTrainingPipeline(
    data_folder=str(DATA_DIR),
    models_dir=str(MODELS_DIR),
    results_dir=str(RESULTS_DIR),
)

# ---------- Pages ----------
@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/training")
def training_page():
    return render_template("training.html")
@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/graphs")
def graphs_page():
    return render_template("graphs.html")


# ---------- API ----------
@app.route("/api/train", methods=["POST"])
def api_train():
    payload   = request.get_json(silent=True) or {}
    epochs    = int(payload.get("epochs", 30))
    batch_sz  = int(payload.get("batch_size", 32))
    patience  = int(payload.get("patience", 5))
    try:
        metrics = pipeline.run_complete_training(epochs=epochs, batch_size=batch_sz, patience=patience)
        return jsonify({"status": "ok", "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/results/metrics", methods=["GET"])
def api_metrics():
    p = RESULTS_DIR / "evaluation_metrics.json"
    if not p.exists():
        return jsonify({"error": "Metrics not found. Please train first."}), 404
    return send_file(p, mimetype="application/json")

@app.route("/api/results/plot/<name>", methods=["GET"])
def api_plot(name: str):
    files = {
        "training":            PLOTS_DIR / "training_loss.png",
        "parity_grid":         PLOTS_DIR / "parity_grid.png",
        "parity_SOH":          PLOTS_DIR / "parity_SOH.png",        # (optional if you add single plots)
        "parity_RUL":          PLOTS_DIR / "parity_RUL.png",
        "parity_RUL_Cycles":   PLOTS_DIR / "parity_RUL_Cycles.png",
    }
    fp = files.get(name)
    if not fp or not fp.exists():
        return jsonify({"error": "Plot not found."}), 404
    return send_file(fp, mimetype="image/png")
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict SOH, RUL (relative), and RUL_Cycles using the PROPOSED PINN model.
    No baseline models here (RF is used only in /api/compare).
    """

    data = request.get_json(silent=True) or {}

    try:
        # ---------- Safety checks ----------
        model_path   = MODELS_DIR / "battery_model.keras"
        scaler_path  = MODELS_DIR / "scalers.joblib"

        if not model_path.exists():
            return jsonify({"error": "PINN model not found. Train first."}), 400
        if not scaler_path.exists():
            return jsonify({"error": "Scalers not found. Train first."}), 400

        # ---------- Load model & scalers ----------
        model = tf.keras.models.load_model(str(model_path))
        scalers = joblib.load(str(scaler_path))

        scaler_X = scalers["scaler_X"]
        scaler_y = scalers["scaler_y"]
        features = scalers["features"]

        # ---------- Build input feature vector ----------
        feat = {k: 0.0 for k in features}

        feat.update({
            "voltage":     float(data.get("voltage", 0.0)),
            "current":     float(data.get("current", 0.0)),
            "temperature": float(data.get("temperature", 25.0)),
            "cycle":       float(data.get("cycle", 0)),
            "capacity":    float(data.get("capacity", 1.0)),
        })

        # One-hot encoding for Li-ion (locked chemistry)
        for k in features:
            if k.startswith("battery_type_"):
                feat[k] = 1.0 if k.endswith("Li-ion") else 0.0

        # ---------- Prepare model input ----------
        x = np.array([[feat[k] for k in features]], dtype="float32")
        x_scaled = scaler_X.transform(x)

        # ---------- Model prediction ----------
        preds = model.predict(x_scaled, verbose=0)
        y_pred = np.concatenate(preds, axis=1)
        y_inv  = scaler_y.inverse_transform(y_pred)[0]

        # ---------- Output ----------
        return jsonify({
            "model_used": "PINN (Physics-Informed Neural Network)",
            "SOH": float(y_inv[0]),                 # 0–1 (convert to % in frontend)
            "RUL_relative": float(y_inv[1]),        # normalized remaining life
            "RUL_Cycles": float(y_inv[2])           # cycles remaining
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json(silent=True) or {}

    try:
        if not (MODELS_DIR / "battery_model.keras").exists():
            return jsonify({"error": "PINN model not found"}), 400
        if not (MODELS_DIR / "rf_model.pkl").exists():
            return jsonify({"error": "RF model not found"}), 400

        # Load scalers and features
        scalers = joblib.load(MODELS_DIR / "scalers.joblib")
        scaler_X, scaler_y, features = (
            scalers["scaler_X"],
            scalers["scaler_y"],
            scalers["features"],
        )

        # Build input
        feat = {k: 0.0 for k in features}
        feat.update({
            "voltage": float(data.get("voltage", 0.0)),
            "current": float(data.get("current", 0.0)),
            "temperature": float(data.get("temperature", 25.0)),
            "cycle": float(data.get("cycle", 0)),
            "capacity": float(data.get("capacity", 1.0)),
        })
        for k in features:
            if k.startswith("battery_type_"):
                feat[k] = 1.0 if k.endswith("Li-ion") else 0.0

        x = np.array([[feat[k] for k in features]], dtype="float32")

        # ---------- PINN ----------
        pinn_model = tf.keras.models.load_model(MODELS_DIR / "battery_model.keras")
        x_pinn = scaler_X.transform(x)
        pinn_preds = pinn_model.predict(x_pinn, verbose=0)
        pinn_y = np.concatenate(pinn_preds, axis=1)
        pinn_out = scaler_y.inverse_transform(pinn_y)[0]

        # ---------- RF ----------
        rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
        rf_scaler = joblib.load(MODELS_DIR / "rf_scaler.pkl")
        x_rf = rf_scaler.transform(x)
        rf_out = rf_model.predict(x_rf)[0]

        return jsonify({
            "PINN (Proposed)": {
                "SOH": float(pinn_out[0]),
                "RUL": float(pinn_out[1]),
                "RUL_Cycles": float(pinn_out[2]),
            },
            "Random Forest (Baseline)": {
                "SOH": float(rf_out[0]),
                "RUL": float(rf_out[1]),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/compare/metrics", methods=["GET"])
def api_compare_metrics():
    try:
        pinn_metrics_path = RESULTS_DIR / "evaluation_metrics.json"
        rf_metrics_path   = BASE / "results_rf" / "rf_metrics.json"

        if not pinn_metrics_path.exists() or not rf_metrics_path.exists():
            return jsonify({"error": "Metrics files not found"}), 404

        with open(pinn_metrics_path) as f:
            pinn_metrics = json.load(f)

        with open(rf_metrics_path) as f:
            rf_metrics = json.load(f)

        return jsonify({
            "PINN": pinn_metrics,
            "RandomForest": rf_metrics
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    # Windows-friendly dev server
    app.run(host="127.0.0.1", port=8000, debug=True)
