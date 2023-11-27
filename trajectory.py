import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv('data/eth/hotel/pixel_pos.csv')

pedestrianId = pd.Series(data.columns).apply(lambda x: int(x.split('.')[0]))
df = pd.DataFrame([data.iloc[0, :].T,
                   data.iloc[1, :].T,
                   data.iloc[2, :].T]).transpose()
df['pedestrianId'] = list(pedestrianId)
pedestrian_ids = df['pedestrianId'].unique()

sequence_length = 3
X, y = [], []

for pedestrian_id in pedestrian_ids:
    x_values = df[df['pedestrianId'] == pedestrian_id].drop(
        columns=['pedestrianId']).values
    for i in range(len(x_values) - sequence_length):
        X.append(x_values[i:i + sequence_length])
        y.append(x_values[i + sequence_length])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Calculate variable for normalization
X_min = X.min(axis=(0, 1), keepdims=True)
X_max = X.max(axis=(0, 1), keepdims=True)


def normalize_data(input_x, input_y):
    input_x = (input_x - X_min) / (X_max - X_min)
    if len(input_y) != 0:
        input_y = (input_y - X_min[0]) / (X_max[0] - X_min[0])
    return input_x, input_y


# Normalize the data
X, y = normalize_data(X, y)

# Reshape the data for LSTM input (samples, time steps, features)
num_features = 3
X = X.reshape(X.shape[0], sequence_length, num_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, num_features)))
model.add(Dense(num_features))
model.compile(optimizer='adam', loss='mse')

# Train the model on the training set
model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

# Make predictions on the test set
predicted_values = model.predict(X_test)

# Visualize actual vs predicted x-coordinates for testing set
plt.figure(figsize=(10, 6))
plt.plot(y_test[:, 1], label='True Values')
plt.plot(predicted_values[:, 1], label='Predicted Values')
plt.title('Pedestrian Trajectory Prediction')
plt.xlabel('Time Step')
plt.ylabel('X Coordinate')
plt.legend()
plt.show()


# Visualize actual vs predicted path for a pedestrian
plt.figure(figsize=(10, 6))

pedestrian_id = 51

actual_trajectory, _ = normalize_data(
    df[df['pedestrianId'] == pedestrian_id].drop(
        columns=['pedestrianId']).values, [])

plt.plot(actual_trajectory[0][:, 1],
         actual_trajectory[0][:, 2],
         marker='o', linestyle='-')

trajectory_length = len(df[df['pedestrianId'] == pedestrian_id])

initial_trajectory = actual_trajectory[:, 0:sequence_length]

generated_trajectory = list([list(x) for x in initial_trajectory[0]])
previous_trajectory = initial_trajectory

for i in range(trajectory_length - sequence_length):
    next_step = model.predict(previous_trajectory)
    generated_trajectory.append(list(next_step[0]))
    previous_trajectory[0][0:sequence_length - 1] = previous_trajectory[0][1:sequence_length]
    previous_trajectory[0][-1] = next_step

generated_trajectory = np.array(generated_trajectory)

plt.plot(generated_trajectory[:, 1],
         generated_trajectory[:, 2],
         marker='x', linestyle='-')
plt.show()
