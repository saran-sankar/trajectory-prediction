# data_processing.py
import pandas as pd
import numpy as np

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
