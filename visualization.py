# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from data_processing import normalize_data

def visualize_predictions(y_test, predicted_values):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, 0], label='True Values')
    plt.plot(predicted_values[:, 0], label='Predicted Values')
    plt.title('Pedestrian Trajectory Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('X Coordinate')
    plt.legend()
    plt.show()

def visualize_pedestrian_trajectory(model, pedestrian_id, df, sequence_length, X_min, X_max):
    plt.figure(figsize=(10, 6))

    actual_trajectory, _ = normalize_data(
        df[df['pedestrianId'] == pedestrian_id].drop(
            columns=['pedestrianId']).values, [], X_min, X_max)

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
