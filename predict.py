import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('ev_battery_model.h5')

# Example input: voltage, current, temperature, soc
sample_input = np.array([[3.7, 0.5, 25, 80]])
rul_pred, soh_pred, rul_cycles_pred = model.predict(sample_input)

print(f"Predicted RUL: {rul_pred[0][0]:.2f}")
print(f"Predicted SOH: {soh_pred[0][0]:.2f}")
print(f"Predicted RUL Cycles: {rul_cycles_pred[0][0]:.2f}")
