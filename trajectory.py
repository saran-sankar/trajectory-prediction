import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        X.append(x_values[i:i+sequence_length])
        y.append(x_values[i+sequence_length])


# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the data
X = (X - X.min()) / (X.max() - X.min())
y = (y - y.min()) / (y.max() - y.min())

# Reshape the data for LSTM input (samples, time steps, features)
X = X.reshape(X.shape[0], sequence_length, 3)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 3)))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Make predictions
predicted_values = model.predict(X)

# Visualize the results (assuming one pedestrian for simplicity)
plt.figure(figsize=(10, 6))
plt.plot(y, label='True Values')
plt.plot(predicted_values, label='Predicted Values')
plt.title('Pedestrian Trajectory Prediction')
plt.xlabel('Time Step')
plt.ylabel('X Coordinate')
plt.legend()
plt.show()
