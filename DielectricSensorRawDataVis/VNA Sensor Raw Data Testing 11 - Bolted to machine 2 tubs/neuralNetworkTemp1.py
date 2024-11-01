import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def read_s_param_file(file_path):
    """Read S-parameter file and return frequencies, S11 components, magnitudes, and phases."""
    frequencies, s11_real, s11_imag = [], [], []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(('!', '#')):
                continue
            parts = line.split()
            if len(parts) >= 3:
                frequencies.append(float(parts[0]))
                s11_real.append(float(parts[1]))
                s11_imag.append(float(parts[2]))

    frequencies = np.array(frequencies)
    s11_real = np.array(s11_real)
    s11_imag = np.array(s11_imag)
    
    magnitudes = np.sqrt(s11_real**2 + s11_imag**2)
    phases = np.arctan2(s11_imag, s11_real)

    return frequencies, s11_real, s11_imag, magnitudes, phases

def load_data(folder_paths):
    """Load data from specified folders and return combined data and labels."""
    data, labels = [], []
    print("Folder paths being used:")
    for folder in folder_paths:
        print(folder)

    # Load dry data
    dry_folder = folder_paths[0]
    if os.path.exists(dry_folder):
        for dry_file in os.listdir(dry_folder):
            if dry_file.endswith('.s1p'):
                dry_file_path = os.path.join(dry_folder, dry_file)
                frequencies, dry_real, dry_imag, dry_mag, dry_phase = read_s_param_file(dry_file_path)
                sample_data = np.concatenate([dry_real, dry_imag, dry_mag, dry_phase])
                data.append(sample_data)
                labels.append(0)
        print(f"Loaded {len(os.listdir(dry_folder))} samples from '{dry_folder}' with label 0.")
    else:
        print(f"Warning: Folder '{dry_folder}' does not exist.")

    # Load water data
    for i in range(1, len(folder_paths)):
        water_folder = folder_paths[i]
        if os.path.exists(water_folder):
            for water_file in os.listdir(water_folder):
                if water_file.endswith('.s1p'):
                    water_file_path = os.path.join(water_folder, water_file)
                    frequencies, water_real, water_imag, water_mag, water_phase = read_s_param_file(water_file_path)
                    sample_data = np.concatenate([water_real, water_imag, water_mag, water_phase])
                    data.append(sample_data)
                    labels.append((i) * 14.37)
                    print(f"Loaded sample from '{water_folder}' with label {(i) * 14.37:.2f}.")
        else:
            print(f"Warning: Folder '{water_folder}' does not exist.")

    return np.array(data), np.array(labels)

# Example usage
folder_paths = [
    '1. Dry',
    '2. Water +1',
    '3. Water +2',
    '4. Water +3',
    '5. Water +4',
    '6. Water +5',
    '7. Water +6',
    '8. Water +7'
]

data, labels = load_data(folder_paths)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize the data
mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
std_X_train[std_X_train == 0] = 1  # Avoid division by zero
X_train = (X_train - mean_X_train) / std_X_train
X_test = (X_test - mean_X_train) / std_X_train

# Reshape the input data for LSTM
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the neural network for regression
model = Sequential([
    LSTM(724, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.05),
    LSTM(512),
    Dropout(0.05),
    Dense(262, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Add model checkpointing
checkpoint = ModelCheckpoint('DNN_V2_Best_Validations__T11_rmsprop_T112.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model and capture history
history = model.fit(X_train_reshaped, y_train, epochs=120, batch_size=16, validation_split=0.2, callbacks=[checkpoint])

# Evaluate the model
loss, mae = model.evaluate(X_test_reshaped, y_test)

# Combine training and test data for overall evaluation
X_all = np.concatenate((X_train_reshaped, X_test_reshaped), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)
loss_all, mae_all = model.evaluate(X_all, y_all)
print(f'Test Mean Absolute Error: {mae:.2f}')
print(f'Overall Mean Absolute Error on All Data: {mae_all:.2f}')

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Save the model
model.save('trained_model_T11_rmsprop_T112.keras')
print("Model saved as 'trained_model_T11_rmsprop_T112.keras'")

# Predicting on new data from NN_Testing folder
def predict_on_testing_folder(model, folder_path, mean, std):
    """Make predictions on .s1p files in the specified testing folder."""
    test_data = []

    s1p_files = [file for file in os.listdir(folder_path) if file.endswith('.s1p')]
    for s1p_file in s1p_files:
        file_path = os.path.join(folder_path, s1p_file)
        _, s11_real, s11_imag, magnitudes, phases = read_s_param_file(file_path)
        sample_data = np.concatenate([s11_real, s11_imag, magnitudes, phases])
        test_data.append(sample_data)

    test_data = np.array(test_data)
    test_data = (test_data - mean) / std
    test_data_reshaped = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

    predictions = model.predict(test_data_reshaped)
    average_prediction = np.mean(predictions)
    print(predictions)
    print(f'Average Prediction from NN_Testing: {average_prediction:.2f}')

# Example usage for prediction
testing_folder_path = 'NN_Testing'
predict_on_testing_folder(model, testing_folder_path, mean_X_train, std_X_train)