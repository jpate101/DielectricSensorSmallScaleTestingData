import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, StringVar, OptionMenu, Button, Label
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

def average_across_frequency_range(start_freq, end_freq, frequency_dict):
    avg_real = []
    avg_imag = []
    
    for freq in frequency_dict:
        if start_freq <= freq <= end_freq:
            real_values = frequency_dict[freq][0]
            imag_values = frequency_dict[freq][1]
            avg_real.append(np.mean(real_values))
            avg_imag.append(np.mean(imag_values))
    
    if avg_real and avg_imag:
        return np.mean(avg_real), np.mean(avg_imag)
    else:
        return None, None  # No data in the specified range

def calculate_group_averages():
    group_averages = {}
    for group in frequency_groups:
        all_real = []
        all_imag = []
        
        for freq in frequency_dicts[group]:
            real_values = frequency_dicts[group][freq][0]
            imag_values = frequency_dicts[group][freq][1]
            all_real.extend(real_values)
            all_imag.extend(imag_values)

        if all_real and all_imag:
            avg_real = np.mean(all_real)
            avg_imag = np.mean(all_imag)
            group_averages[group] = (avg_real, avg_imag)
    
    return group_averages

def plot_frequency_range(start_freq, end_freq):
    ax.clear()  # Clear the current axes

    for group, color in zip(frequency_groups, colors):
        avg_real, avg_imag = average_across_frequency_range(start_freq, end_freq, frequency_dicts[group])
        
        if avg_real is not None and avg_imag is not None:
            ax.plot(avg_real, avg_imag, 'o', color=color, label=f'Group {group} Avg ({start_freq/1e9:.2f} - {end_freq/1e9:.2f} GHz)')

    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_title('S-Parameter: Average Real vs Imaginary Parts')
    ax.set_xlabel('Real Part (S11)')
    ax.set_ylabel('Imaginary Part (S11)')
    ax.legend()
    ax.grid()

    canvas.draw()  # Draw the updated figure on the canvas

def plot_group_averages():
    group_averages = calculate_group_averages()
    groups = list(group_averages.keys())
    avg_reals = [group_averages[group][0] for group in groups]
    avg_imags = [group_averages[group][1] for group in groups]

    ax_bar.clear()
    bar_width = 0.35
    indices = np.arange(len(groups))

    bar1 = ax_bar.bar(indices, avg_reals, bar_width, label='Average Real', color='b')
    bar2 = ax_bar.bar(indices + bar_width, avg_imags, bar_width, label='Average Imaginary', color='r')

    ax_bar.set_xlabel('Groups')
    ax_bar.set_ylabel('Average Values')
    ax_bar.set_title('Average Real and Imaginary Parts per Group')
    ax_bar.set_xticks(indices + bar_width / 2)
    ax_bar.set_xticklabels(groups)
    ax_bar.legend()
    ax_bar.grid()

    canvas_bar.draw()  # Draw the updated bar graph on the canvas

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

# Frequency range entry
start_freq_label = Label(root, text="Start Frequency (GHz):")
start_freq_label.pack(pady=5)
start_freq_entry = StringVar(root)
start_freq_entry.set("0.9")  # Default value
start_freq_input = OptionMenu(root, start_freq_entry, *[f"{i/100:.2f}" for i in range(1000, 2000)])  # 1.00 to 2.00 GHz
start_freq_input.pack(pady=5)

end_freq_label = Label(root, text="End Frequency (GHz):")
end_freq_label.pack(pady=5)
end_freq_entry = StringVar(root)
end_freq_entry.set("1.79")  # Default value
end_freq_input = OptionMenu(root, end_freq_entry, *[f"{i/100:.2f}" for i in range(1000, 2000)])  # 1.00 to 2.00 GHz
end_freq_input.pack(pady=5)

# Create buttons to plot the average for the selected frequency range and show group averages
plot_button = Button(root, text="Plot Average", command=lambda: plot_frequency_range(float(start_freq_entry.get()) * 1e9, float(end_freq_entry.get()) * 1e9))
plot_button.pack(pady=5)

average_button = Button(root, text="Show Group Averages", command=plot_group_averages)
average_button.pack(pady=5)

# Set up the Matplotlib figure and canvas for frequency plot
fig, ax = plt.subplots(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

# Set up the Matplotlib figure and canvas for bar graph
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
canvas_bar = FigureCanvasTkAgg(fig_bar, master=root)
canvas_bar.get_tk_widget().pack(side='right', fill='both', expand=True)

# Start the Tkinter main loop
root.mainloop()