import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

def generate_future_dates(start_date, periods):
    future_dates = pd.date_range(start=start_date, periods=periods, freq='12h')  # Two drawings per day
    return future_dates

# Get the filename of the current script
filename = os.path.basename(__file__)

# Step 1: Load and preprocess the data
# Load the dataset containing historical sequences of 6-digit numbers from the text file
# Replace 'dataset.txt' with the path to your text file
with open('full_historical_pancakeswap_plottery_winning_numbers.txt', 'r') as file:
    lines = file.readlines()

# Convert the data into a format suitable for LSTM training
# Assume each line in the text file contains a single 6-digit number as a string
data = [int(line.strip()) for line in lines]

# Convert the data to a NumPy array
data = np.array(data)

# Step 2: Data Preparation
# Scale the input data to improve training stability and convergence
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# Ensure that the validation set size is large enough to accommodate future sequences
time_steps = 10  # Example value, adjust as needed
val_size = len(data) - int(len(data) * 0.8)
if val_size < time_steps:
    raise ValueError("Validation set size is too small for generating future sequences")

# Split the dataset into training and validation sets
# Ensure that the validation set contains sequences of numbers occurring after the training data
train_data, val_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# Function to create input-output pairs from the data
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Generate input-output pairs for training and validation sets
X_train, y_train = create_sequences(train_data, time_steps)
X_val, y_val = create_sequences(val_data, time_steps)

# Step 3: Define the LSTM Model Architecture
model = Sequential([
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=32, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(units=32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the LSTM Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Visualize training and validation loss curves with neon green background
plt.figure(facecolor='#FF6EC7')  #outside background color
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Training and Validation Loss Curves - {filename}')  # Set custom title with filename

# Change the background color of the area where the lines are plotted
ax = plt.gca()
ax.set_facecolor('#0000FF')  # inside background color for the plot area

plt.show()

# Step 5: Prediction and Forecasting
# Use the trained LSTM model to predict future sequences of 6-digit numbers
future_periods =2  # Forecasting for a week (2 drawings per day for 7 days)
start_date = pd.to_datetime('2024-04-18')  # Example start date, replace with your desired start date
future_dates = generate_future_dates(start_date, future_periods)

# Take the last sequence from the validation set as the starting point for prediction
last_sequence = X_val[-1]

# Generate future sequences by iteratively predicting the next sequence
predicted_sequences = []
confidence_ratings = []
for i in range(future_periods):
    # Reshape the last sequence to match the input shape of the model
    current_sequence = last_sequence.reshape((1, time_steps, 1))

    # Predict the next sequence
    next_sequence = model.predict(current_sequence)

    # Append the predicted sequence to the list
    predicted_sequences.append(next_sequence[0, 0])  # Extract the predicted value from the array

    # Calculate confidence rating (optional)
    confidence = 1 - np.abs(next_sequence[0, 0] - np.mean(y_val)) / np.std(y_val)
    confidence_ratings.append(confidence)

    # Update the last sequence for the next iteration
    last_sequence = np.append(last_sequence[1:], next_sequence[0, 0])

    # Print debug information
    print(f"Date: {future_dates[i]}, Predicted sequence = {int(predicted_sequences[-1])}, Confidence: {confidence}")

# Inverse transform the predicted sequences to obtain the original scale
predicted_sequences = scaler.inverse_transform(np.array(predicted_sequences).reshape(-1, 1))

# Print the predicted sequences
print("Predicted Sequences:")
for i, sequence in enumerate(predicted_sequences):
    print(f"Date: {future_dates[i]}, Predicted sequence: {int(sequence[0])}, Confidence: {confidence_ratings[i]}")

# Visualize the predicted sequences
plt.figure(figsize=(10, 6))
plt.plot(future_dates, predicted_sequences, marker='o', color='b', label='Predicted Sequences')
plt.xlabel('Date')
plt.ylabel('Sequence')
plt.title('Predicted Sequences Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
