import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def read_s_param_file(file_path):
    frequencies = []
    s11_real = []
    s11_imag = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('!') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                frequencies.append(float(parts[0]))
                s11_real.append(float(parts[1]))
                s11_imag.append(float(parts[2]))

    # Convert to numpy arrays
    frequencies = np.array(frequencies)
    s11_real = np.array(s11_real)
    s11_imag = np.array(s11_imag)
    
    # Calculate magnitude and phase
    magnitudes = np.sqrt(s11_real**2 + s11_imag**2)
    phases = np.arctan2(s11_imag, s11_real)

    return frequencies, s11_real, s11_imag, magnitudes, phases

def load_data(folder_paths):
    data = []
    labels = []
    
    # Load the reference data from the "1. After Calibration" folder
    calibration_folder = folder_paths[0]  # '1. After Calibration'
    calibration_files = [file for file in os.listdir(calibration_folder) if file.endswith('.s1p')]
    
    # Store calibration data for random selection
    calibration_data = {}
    for calibration_file in calibration_files:
        calibration_file_path = os.path.join(calibration_folder, calibration_file)
        frequencies, cal_real, cal_imag, cal_mag, cal_phase = read_s_param_file(calibration_file_path)
        calibration_data[calibration_file] = (cal_real, cal_imag, cal_mag, cal_phase)

    # Load dry data (label 0)
    dry_folder = folder_paths[1]  # '2. Dry'
    dry_files = [file for file in os.listdir(dry_folder) if file.endswith('.s1p')]
    
    for dry_file in dry_files:
        dry_file_path = os.path.join(dry_folder, dry_file)
        frequencies, dry_real, dry_imag, dry_mag, dry_phase = read_s_param_file(dry_file_path)
        
        # Randomly select a calibration file and extract its data
        cal_file = random.choice(list(calibration_data.keys()))
        cal_real, cal_imag, cal_mag, cal_phase = calibration_data[cal_file]
        
        # Subtract calibration features from dry features
        sample_data = np.concatenate([
            dry_real - cal_real, 
            dry_imag - cal_imag, 
            dry_mag - cal_mag, 
            dry_phase - cal_phase
        ])
        
        data.append(sample_data)
        labels.append(0)  # Label for dry

    # Load water data (labels from 10 to 100)
    for i in range(1, 8):
        water_folder = folder_paths[i + 2]  # '3. Water +1' to '9. Water +7'
        water_files = [file for file in os.listdir(water_folder) if file.endswith('.s1p')]
        
        for water_file in water_files:
            water_file_path = os.path.join(water_folder, water_file)
            frequencies, water_real, water_imag, water_mag, water_phase = read_s_param_file(water_file_path)
            
            # Randomly select a calibration file and extract its data
            cal_file = random.choice(list(calibration_data.keys()))
            cal_real, cal_imag, cal_mag, cal_phase = calibration_data[cal_file]
            
            # Subtract calibration features from water features
            sample_data = np.concatenate([
                water_real - cal_real, 
                water_imag - cal_imag, 
                water_mag - cal_mag, 
                water_phase - cal_phase
            ])
            
            data.append(sample_data)
            labels.append(i * 14.37)  # Example labels (10, 20, ..., 70)

    print(f"Total samples loaded: {len(data)}")
    return np.array(data), np.array(labels)

# Example usage
folder_paths = [
    '1. After Calibration',
    '2. Dry',
    '3. Water +1',
    '4. Water +2',
    '5. Water +3',
    '6. Water +4',
    '7. Water +5',
    '8. Water +6',
    '9. Water +7',
    '10. Validation'
]

data, labels = load_data(folder_paths)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.05, random_state=42)

# Normalize the data
mean_X_train = np.mean(X_train, axis=0)
std_X_train = np.std(X_train, axis=0)
std_X_train[std_X_train == 0] = 1  # Avoid division by zero
X_train = (X_train - mean_X_train) / std_X_train
X_test = (X_test - mean_X_train) / std_X_train  # Use training mean/std to normalize test data

# Reshape the input data for LSTM
N = X_train.shape[1] // 4  # number of frequency points considering 4 features now
X_train_reshaped = X_train.reshape((X_train.shape[0], N, 4))  # 4 features
X_test_reshaped = X_test.reshape((X_test.shape[0], N, 4))  # 4 features

# Build the neural network for regression
model = Sequential()
model.add(LSTM(128, input_shape=(N, 4), return_sequences=True))  # Adjusted for 4 features
model.add(LSTM(64))  # Add a second LSTM layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))  # Single output neuron for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model and capture history
history = model.fit(X_train_reshaped, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test_reshaped, y_test)
print(f'Test Mean Absolute Error: {mae:.2f}')

# Combine training and test data for overall evaluation
X_all = np.concatenate((X_train_reshaped, X_test_reshaped), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# Evaluate the model on all data
loss_all, mae_all = model.evaluate(X_all, y_all)
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