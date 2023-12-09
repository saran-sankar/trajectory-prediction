# lstm_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def build_lstm_model(sequence_length, num_features):
    model = Sequential()
    model.add(LSTM(10, input_shape=(sequence_length, num_features)))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=128, validation_split=0.1):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def make_predictions(model, X_test):
    return model.predict(X_test)
