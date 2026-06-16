# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_model(input_shape=(17,)):
    inputs = layers.Input(shape=input_shape, name='input')
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    soh = layers.Dense(1, name='soh')(x)
    rul = layers.Dense(1, name='rul')(x)
    rul_cycles = layers.Dense(1, name='rul_cycles')(x)
    model = Model(inputs=inputs, outputs=[soh, rul, rul_cycles])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
