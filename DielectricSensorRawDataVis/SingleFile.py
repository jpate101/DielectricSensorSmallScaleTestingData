import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

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
    
    return np.array(frequencies), np.array(s11_real), np.array(s11_imag)

def print_data(file_path):
    frequencies, s11_real, s11_imag = read_s_param_file(file_path)
    
    print(f"{'Frequency (Hz)':<20} {'Real Part (S11)':<20} {'Imaginary Part (S11)':<20}")
    print("=" * 60)
    
    for freq, real, imag in zip(frequencies, s11_real, s11_imag):
        print(f"{freq:<20} {real:<20} {imag:<20}")

    # Calculate area under the curve
    area_real = simps(s11_real, frequencies)
    area_imag = simps(s11_imag, frequencies)
    
    print(f"\nArea under the Real Part (S11) curve: {area_real:.4f}")
    print(f"Area under the Imaginary Part (S11) curve: {area_imag:.4f}")

def plot_data(file_path):
    frequencies, s11_real, s11_imag = read_s_param_file(file_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, s11_real, label='Real Part (S11)', color='blue', marker='o')
    plt.plot(frequencies, s11_imag, label='Imaginary Part (S11)', color='red', marker='x')
    
    plt.title('S-Parameter Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('S11 Value')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
file_path = '2024-09-27T02-10-15.050670Z_210_vnaData.s1p'  # Replace with your actual file path
print_data(file_path)
plot_data(file_path)

