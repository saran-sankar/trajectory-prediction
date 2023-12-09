# lstm_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

def build_lstm_model(sequence_length, num_features):
    model = Sequential()
    model.add(LSTM(10, input_shape=(sequence_length, num_features)))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=128, validation_split=0.1):
    history = History()  # Create a history object to store training metrics
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[history])

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def make_predictions(model, X_test):
    return model.predict(X_test)
