import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Normalize the data (optional but recommended)
X = (X - X.min()) / (X.max() - X.min())
y = (y - y.min()) / (y.max() - y.min())

# Reshape the data for LSTM input (samples, time steps, features)
X = X.reshape(X.shape[0], sequence_length, 3)

# plt.figure(figsize=(10, 6))
#
# plt.plot(df[1][df['pedestrianId'] == 1],
#          df[2][df['pedestrianId'] == 1],
#          marker='o', linestyle='-')
#
# # Add labels and legend
# plt.title('Plot of DataFrame Rows')
# plt.xlabel('Column')
# plt.ylabel('Values')
# plt.legend()
#
# # Show the plot
# plt.show()
