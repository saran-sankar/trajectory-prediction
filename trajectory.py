import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(file_path):
    return pd.read_csv(file_path)

def create_pedestrian_dataframe(data):
    pedestrianId = pd.Series(data.columns).apply(lambda x: int(x.split('.')[0]))
    df = pd.DataFrame([data.iloc[1, :].T, data.iloc[2, :].T]).transpose()
    df['pedestrianId'] = list(pedestrianId)
    return df

def prepare_sequences(df, sequence_length):
    X, y = [], []
    pedestrian_ids = df['pedestrianId'].unique()

    for pedestrian_id in pedestrian_ids:
        x_values = df[df['pedestrianId'] == pedestrian_id].drop(columns=['pedestrianId']).values
        for i in range(len(x_values) - sequence_length):
            X.append(x_values[i:i + sequence_length])
            y.append(x_values[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    return X, y

def normalize_data(input_x, input_y, X_min, X_max):
    input_x = (input_x - X_min) / (X_max - X_min)
    if len(input_y) != 0:
        input_y = (input_y - X_min[0]) / (X_max[0] - X_min[0])
    return input_x, input_y

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

def visualize_predictions(y_test, predicted_values):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 0], label='True Values')
    plt.plot(predicted_values[:, 0], label='Predicted Values')
    plt.title('Pedestrian Trajectory Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('X Coordinate')
    plt.legend()
    plt.show()

def visualize_pedestrian_trajectory(model, pedestrian_id, df, sequence_length):
    plt.figure(figsize=(10, 6))

    actual_trajectory, _ = normalize_data(
        df[df['pedestrianId'] == pedestrian_id].drop(columns=['pedestrianId']).values, [], X_min, X_max)

    plt.plot(actual_trajectory[0][:, 0], actual_trajectory[0][:, 1], marker='o', linestyle='-')

    trajectory_length = len(df[df['pedestrianId'] == pedestrian_id])

    initial_trajectory = actual_trajectory[:, 0:sequence_length]

    generated_trajectory = list([list(x) for x in initial_trajectory[0]])
    previous_trajectory = initial_trajectory

    for i in range(trajectory_length - sequence_length):
        next_step = model.predict(previous_trajectory)
        generated_trajectory.append(list(next_step[0]))
        previous_trajectory[0][0:sequence_length - 1] = previous_trajectory[0][1:sequence_length]
        previous_trajectory[0][-1] = actual_trajectory[0, sequence_length + i]

    generated_trajectory = np.array(generated_trajectory)

    plt.plot(generated_trajectory[:, 0], generated_trajectory[:, 1], marker='x', linestyle='-')
    plt.show()

if __name__ == "__main__":
    # Load data
    data = load_data('data/eth/hotel/pixel_pos.csv')

    # Create pedestrian dataframe
    df = create_pedestrian_dataframe(data)

    # Prepare sequences
    sequence_length = 4
    X, y = prepare_sequences(df, sequence_length)

    # Calculate variables for normalization
    X_min = X.min(axis=(0, 1), keepdims=True)
    X_max = X.max(axis=(0, 1), keepdims=True)

    # Normalize data
    X, y = normalize_data(X, y, X_min, X_max)

    # Reshape data for LSTM input
    num_features = 2
    X = X.reshape(X.shape[0], sequence_length, num_features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # Build LSTM model
    model = build_lstm_model(sequence_length, num_features)

    # Train the model
    train_model(model, X_train, y_train)

    # Make predictions on the test set
    predicted_values = make_predictions(model, X_test)

    # Visualize actual vs predicted x-coordinates for testing set
    visualize_predictions(y_test, predicted_values)

    # Visualize actual vs predicted path for a pedestrian
    pedestrian_id_to_visualize = 1221
    visualize_pedestrian_trajectory(model, pedestrian_id_to_visualize, df, sequence_length)
