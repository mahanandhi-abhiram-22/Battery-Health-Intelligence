import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout

def build_battery_model(input_dim: int) -> tf.keras.Model:
    """Simple dense multi-output regressor for SOH, RUL, RUL_Cycles."""
    inp = Input(shape=(input_dim,), name="input")
    x = Dense(128, activation="relu")(inp)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)

    soh = Dense(1, name="SOH")(x)
    rul = Dense(1, name="RUL")(x)
    rulc = Dense(1, name="RUL_Cycles")(x)

    model = Model(inp, [soh, rul, rulc])
    # Important: metrics specified PER output to avoid the error you saw
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics={"SOH": "mae", "RUL": "mae", "RUL_Cycles": "mae"}
    )
    return model
