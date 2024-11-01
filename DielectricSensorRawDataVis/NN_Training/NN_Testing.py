import numpy as np
import os
from tensorflow.keras.models import load_model

def read_s_param_file(file_path):
    """
    Reads S-parameter data from a .s1p file and extracts S11 parameters.
    
    Parameters:
    file_path (str): Path to the .s1p file.
    
    Returns:
    tuple: S11 real part, S11 imaginary part, magnitudes, and phases.
    """
    s11_real = []
    s11_imag = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('!') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                s11_real.append(float(parts[1]))
                s11_imag.append(float(parts[2]))

    # Convert to numpy arrays
    s11_real = np.array(s11_real)
    s11_imag = np.array(s11_imag)
    
    # Calculate magnitude and phase
    magnitudes = np.sqrt(s11_real**2 + s11_imag**2)
    phases = np.arctan2(s11_imag, s11_real)

    return s11_real, s11_imag, magnitudes, phases

def load_and_predict(model_path, folder_path):
    """
    Loads the trained model and makes predictions based on .s1p files in a folder.

    Parameters:
    model_path (str): Path to the trained Keras model.
    folder_path (str): Path to the folder containing .s1p files.
    """
    # Load the trained model
    model = load_model(model_path)

    # Prepare data for predictions
    data = []

    # Load .s1p files
    s1p_files = [file for file in os.listdir(folder_path) if file.endswith('.s1p')]
    
    for s1p_file in s1p_files:
        file_path = os.path.join(folder_path, s1p_file)
        s11_real, s11_imag, magnitudes, phases = read_s_param_file(file_path)

        # Concatenate all features into one vector for the file
        sample_data = np.concatenate([s11_real, s11_imag, magnitudes, phases])
        data.append(sample_data)

    # Convert data to numpy array and reshape for LSTM
    data = np.array(data)
    data_reshaped = data.reshape((data.shape[0], 1, data.shape[1]))

    # Make predictions
    predictions = model.predict(data_reshaped)

    # Print predictions and their average
    print(f'Predictions: {predictions.flatten()}')  # Print individual predictions
    average_prediction = np.mean(predictions)
    print(f'Average Prediction: {average_prediction:.2f}')

# Example usage
model_path = 'DNN_V2_Best_Validations__T8_T7_T9_rmsprop.keras'  # Path to the trained model
folder_path = 'NN_Testing'  # Folder path containing .s1p files

load_and_predict(model_path, folder_path)