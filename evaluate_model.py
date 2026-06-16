import json, joblib
from pathlib import Path
import tensorflow as tf

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

def load_artifacts():
    model = tf.keras.models.load_model(str(MODELS_DIR / "battery_model.keras"))
    scalers = joblib.load(MODELS_DIR/"scalers.joblib")
    return model, scalers

def evaluate_return_metrics():
    metrics_path = RESULTS_DIR/"evaluation_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {"info": "Run training first to generate metrics."}

if __name__ == "__main__":
    print(json.dumps(evaluate_return_metrics(), indent=2))
