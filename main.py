# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from lstm_model import build_lstm_model, train_model, make_predictions
from data_processing import load_data, create_pedestrian_dataframe, prepare_sequences, normalize_data
from visualization import visualize_predictions, visualize_pedestrian_trajectory

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
    visualize_pedestrian_trajectory(model, pedestrian_id_to_visualize, 
        df, sequence_length, X_min, X_max)
