import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, StringVar, Label, Scale, Button, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import random

def read_s_param_files_in_folder(file_path):
    frequency_dict = {}
    f, r, i = read_s_param_file(file_path)
    for freq, real, imag in zip(f, r, i):
        if freq not in frequency_dict:
            frequency_dict[freq] = ([], [])
        frequency_dict[freq][0].append(real)  # Real part
        frequency_dict[freq][1].append(imag)  # Imaginary part
    return frequency_dict

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

def calculate_magnitude_phase(frequency_dict):
    magnitude = {}
    phase = {}

    for freq in frequency_dict:
        real_values = frequency_dict[freq][0]
        imag_values = frequency_dict[freq][1]

        if real_values and imag_values:
            avg_real = np.mean(real_values)
            avg_imag = np.mean(imag_values)
            mag = 20 * np.log10(np.sqrt(avg_real**2 + avg_imag**2))
            ph = np.degrees(np.arctan2(avg_imag, avg_real))

            magnitude[freq] = mag
            phase[freq] = ph

    return magnitude, phase

# Load data from specified folders
folder_paths = ['Group1', 'Group2', 'Group3', 'Group4']
group_magnitudes = {}
group_phases = {}
colors = ['green', 'blue', 'red', 'purple']
group_labels = ['Group 1 Cal', 'Group 2 Wet', 'Group 3 Dry', 'Group 4 Val']

for group in folder_paths:
    files = [file for file in os.listdir(group) if file.endswith('.s1p')]
    selected_file = random.choice(files) if files else None
    if selected_file:
        file_path = os.path.join(group, selected_file)
        frequency_dict = read_s_param_files_in_folder(file_path)
        group_magnitudes[group], group_phases[group] = calculate_magnitude_phase(frequency_dict)

# Set up the main Tkinter window
root = Tk()
root.title("S-Parameter Plotter")

# Frame for sliders
slider_frame = Frame(root)
slider_frame.pack(pady=10)

# Sliders for frequency filtering
min_freq_label = Label(slider_frame, text="Min Frequency (GHz):")
min_freq_label.grid(row=0, column=0)
min_freq_slider = Scale(slider_frame, from_=0.0, to=1.8, resolution=0.01, orient='horizontal')
min_freq_slider.set(0.0)
min_freq_slider.grid(row=0, column=1)

max_freq_label = Label(slider_frame, text="Max Frequency (GHz):")
max_freq_label.grid(row=1, column=0)
max_freq_slider = Scale(slider_frame, from_=0.0, to=1.8, resolution=0.01, orient='horizontal')
max_freq_slider.set(1.8)
max_freq_slider.grid(row=1, column=1)

# Function to plot data
def plot_data():
    min_freq = min_freq_slider.get() * 1e9
    max_freq = max_freq_slider.get() * 1e9

    # Clear existing plots
    ax_magnitude.clear()
    ax_phase.clear()

    # Plot Magnitude
    ax_magnitude.set_title('Magnitude vs Frequency')
    ax_magnitude.set_xlabel('Frequency (Hz)')
    ax_magnitude.set_ylabel('Magnitude (dB)')
    
    for group, color, label in zip(folder_paths, colors, group_labels):
        freqs = list(group_magnitudes[group].keys())
        mag_values = [group_magnitudes[group][freq] for freq in freqs if min_freq <= freq <= max_freq]
        filtered_freqs = [freq for freq in freqs if min_freq <= freq <= max_freq]

        ax_magnitude.plot(filtered_freqs, mag_values, label=label, marker='o', color=color)

    ax_magnitude.axhline(0, color='grey', lw=0.5, ls='--')
    ax_magnitude.legend()
    ax_magnitude.grid()

    # Plot Phase
    ax_phase.set_title('Phase vs Frequency')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (Degrees)')

    for group, color, label in zip(folder_paths, colors, group_labels):
        freqs = list(group_phases[group].keys())
        ph_values = [group_phases[group][freq] for freq in freqs if min_freq <= freq <= max_freq]
        filtered_freqs = [freq for freq in freqs if min_freq <= freq <= max_freq]

        ax_phase.plot(filtered_freqs, ph_values, label=label, marker='o', color=color)

    ax_phase.axhline(0, color='grey', lw=0.5, ls='--')
    ax_phase.legend()
    ax_phase.grid()

    # Draw the updated figures
    canvas_magnitude.draw()
    canvas_phase.draw()

# Button to refresh the plots
refresh_button = Button(root, text="Refresh Plots", command=plot_data)
refresh_button.pack(pady=10)

# Set up Matplotlib figures and canvas for embedding
fig_magnitude, ax_magnitude = plt.subplots(figsize=(8, 6))
canvas_magnitude = FigureCanvasTkAgg(fig_magnitude, master=root)
canvas_magnitude.get_tk_widget().pack(side='top', fill='both', expand=True)

fig_phase, ax_phase = plt.subplots(figsize=(8, 6))
canvas_phase = FigureCanvasTkAgg(fig_phase, master=root)
canvas_phase.get_tk_widget().pack(side='top', fill='both', expand=True)

# Initial plot
plot_data()

# Start the Tkinter main loop
root.mainloop()