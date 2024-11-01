import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Scale, Button, Frame, Checkbutton, IntVar, Scrollbar, Canvas
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

# Load data from specified folders
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

# Set up the main Tkinter window
root = Tk()
root.title("Scatter Plot for Real and Imaginary Values")

# Create a dictionary to hold checkbutton states
check_vars = {group: IntVar(value=1) for group in folder_paths}
colors = ['green', 'red'] + [plt.cm.Blues(i / 7) for i in range(1, 8)] + ['purple']
group_labels = ['Cal', 'Dry'] + [f'Water {i}' for i in range(1, 8)] + ['Validation']

# Create a canvas for scrolling
canvas = Canvas(root)
scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Pack the canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

# Frame for sliders
slider_frame = Frame(scrollable_frame)
slider_frame.pack(pady=10)

# Sliders for frequency filtering with increased length
min_freq_label = Label(slider_frame, text="Min Frequency (GHz):")
min_freq_label.grid(row=0, column=0)
min_freq_slider = Scale(slider_frame, from_=0.0, to=4.0, resolution=0.01, orient='horizontal', length=400)
min_freq_slider.set(0.0)
min_freq_slider.grid(row=0, column=1)

max_freq_label = Label(slider_frame, text="Max Frequency (GHz):")
max_freq_label.grid(row=1, column=0)
max_freq_slider = Scale(slider_frame, from_=0.0, to=4.0, resolution=0.01, orient='horizontal', length=400)
max_freq_slider.set(4.0)
max_freq_slider.grid(row=1, column=1)

# Function to plot data
def plot_scatter():
    min_freq = min_freq_slider.get() * 1e9
    max_freq = max_freq_slider.get() * 1e9

    ax.clear()  # Clear existing plot

    for group, color in zip(folder_paths, colors):
        if check_vars[group].get() == 1:  # Check if the line is toggled on
            files = [file for file in os.listdir(group) if file.endswith('.s1p')]
            selected_file = random.choice(files) if files else None
            if selected_file:
                file_path = os.path.join(group, selected_file)
                frequency_dict = read_s_param_files_in_folder(file_path)

                # Plot real vs imaginary
                for freq in frequency_dict:
                    if min_freq <= freq <= max_freq:
                        real_values = frequency_dict[freq][0]
                        imag_values = frequency_dict[freq][1]
                        ax.scatter(real_values, imag_values, color=color)

    ax.set_title('Scatter Plot of Real vs Imaginary Values')
    ax.set_xlabel('Real Values')
    ax.set_ylabel('Imaginary Values')
    ax.grid()
    
    # Draw the updated figure
    canvas_scatter.draw()

# Create checkbuttons for each group
check_frame = Frame(scrollable_frame)
check_frame.pack(pady=10)

for group, label in zip(folder_paths, group_labels):
    checkbutton = Checkbutton(check_frame, text=label, variable=check_vars[group], command=plot_scatter)
    checkbutton.pack(side='left')

# Button to refresh the scatter plot
refresh_button = Button(scrollable_frame, text="Refresh Scatter Plot", command=plot_scatter)
refresh_button.pack(pady=10)

# Set up Matplotlib figure and canvas for embedding
fig, ax = plt.subplots(figsize=(8, 6))
canvas_scatter = FigureCanvasTkAgg(fig, master=scrollable_frame)
canvas_scatter.get_tk_widget().pack(side='top', fill='both', expand=True)

# Initial plot
plot_scatter()

# Start the Tkinter main loop
root.mainloop()