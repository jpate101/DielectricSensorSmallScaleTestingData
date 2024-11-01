import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, StringVar, OptionMenu, Button, Checkbutton, BooleanVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

def read_s_param_files_in_folder(folder_path):
    frequency_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.s1p'):
            file_path = os.path.join(folder_path, file_name)
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

def plot_frequency(selected_frequency, plot_all):
    ax.clear()  # Clear the current axes

    # Plot data from all groups
    for group, color in zip(frequency_groups, colors):
        if plot_all:  # Plot all frequencies
            for freq in frequency_dicts[group]:
                real_values = frequency_dicts[group][freq][0]
                imag_values = frequency_dicts[group][freq][1]
                ax.plot(real_values, imag_values, 'o', color=color, label=f'Group {group} - {freq/1e9:.2f} GHz')
        else:  # Plot selected frequency
            frequency = float(selected_frequency)
            if frequency in frequency_dicts[group]:
                real_values = frequency_dicts[group][frequency][0]
                imag_values = frequency_dicts[group][frequency][1]
                ax.plot(real_values, imag_values, 'o', color=color, label=f'Group {group}: {frequency/1e9:.2f} GHz')

    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_title('S-Parameter: Real vs Imaginary Parts')
    ax.set_xlabel('Real Part (S11)')
    ax.set_ylabel('Imaginary Part (S11)')
    ax.legend()
    ax.grid()

    canvas.draw()  # Draw the updated figure on the canvas

# Load data from specified folders
folder_paths = ['Group1', 'Group2', 'Group3', 'Group4']  # Replace with your actual folder paths
frequency_dicts = {}
frequency_groups = ['Group1', 'Group2', 'Group3', 'Group4']  # Group names
colors = ['green', 'blue', 'red', 'purple']  # Colors for each group

for folder in folder_paths:
    frequency_dicts[folder] = read_s_param_files_in_folder(folder)

# Set up the main Tkinter window
root = Tk()
root.title("S-Parameter Plotter")

# Create a StringVar for the dropdown
selected_frequency = StringVar(root)
selected_frequency.set(list(frequency_dicts[frequency_groups[0]].keys())[0])  # Set the default value

# Create the dropdown menu
frequency_menu = OptionMenu(root, selected_frequency, *sorted(set(freq for group in frequency_groups for freq in frequency_dicts[group].keys())))
frequency_menu.pack(pady=10)

# Create a BooleanVar for the "Plot All" option
plot_all_var = BooleanVar()
plot_all_var.set(False)  # Default to not plot all

# Create a checkbox to toggle "Plot All" option
checkbutton = Checkbutton(root, text="Plot All Frequencies", variable=plot_all_var)
checkbutton.pack(pady=5)

# Create a button to plot the selected frequency
plot_button = Button(root, text="Plot", command=lambda: plot_frequency(selected_frequency.get(), plot_all_var.get()))
plot_button.pack(pady=5)

# Set up the Matplotlib figure and canvas
fig, ax = plt.subplots(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

# Start the Tkinter main loop
root.mainloop()