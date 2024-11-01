import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Scale, Button, Frame, Checkbutton, IntVar, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import random
from scipy.integrate import trapezoid

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
group_magnitudes = {}
group_phases = {}
colors = ['green', 'red'] + [plt.cm.Blues(i / 7) for i in range(1, 8)] + ['purple']
group_labels = ['Cal', 'Dry'] + [f'Water {i}' for i in range(1, 8)] + ['Validation']

def load_data(use_random=False):
    for group in folder_paths:
        files = [file for file in os.listdir(group) if file.endswith('.s1p')]
        if use_random:
            selected_file = random.choice(files) if files else None
            if selected_file:
                file_path = os.path.join(group, selected_file)
                frequency_dict = read_s_param_files_in_folder(file_path)
                group_magnitudes[group], group_phases[group] = calculate_magnitude_phase(frequency_dict)
        else:
            all_mags = []
            for file in files:
                file_path = os.path.join(group, file)
                frequency_dict = read_s_param_files_in_folder(file_path)
                mag, _ = calculate_magnitude_phase(frequency_dict)
                all_mags.append(mag)
                
            # Average magnitudes
            if all_mags:
                avg_magnitude = {}
                for freq in all_mags[0].keys():
                    avg_magnitude[freq] = np.mean([mag[freq] for mag in all_mags])
                group_magnitudes[group] = avg_magnitude
                group_phases[group] = calculate_magnitude_phase(frequency_dict)[1]


# Set up the main Tkinter window
root = Tk()
root.title("S-Parameter Plotter")

# Frame for sliders
slider_frame = Frame(root)
slider_frame.pack(pady=10)

# Sliders for frequency filtering
min_freq_label = Label(slider_frame, text="Min Frequency (GHz):")
min_freq_label.grid(row=0, column=0)
min_freq_slider = Scale(slider_frame, from_=0.4, to=4.0, resolution=0.01, orient='horizontal', length=600)
min_freq_slider.set(0.0)
min_freq_slider.grid(row=0, column=1)

max_freq_label = Label(slider_frame, text="Max Frequency (GHz):")
max_freq_label.grid(row=1, column=0)
max_freq_slider = Scale(slider_frame, from_=0.4, to=4.0, resolution=0.01, orient='horizontal', length=600)
max_freq_slider.set(4.0)
max_freq_slider.grid(row=1, column=1)

# Toggle button for random/average data
toggle_random = IntVar(value=0)  # 0 for average, 1 for random
toggle_button = Checkbutton(root, text="Use Random Data", variable=toggle_random, command=lambda: load_data(use_random=toggle_random.get()))
toggle_button.pack(pady=10)

# Create a Treeview for displaying data in a table format
columns = ('Group', 'Area Under Curve', 'Max Mag', 'Min Mag')
area_table = ttk.Treeview(root, columns=columns, show='headings')
area_table.heading('Group', text='Group', command=lambda: sort_column(area_table, 'Group'))
area_table.heading('Area Under Curve', text='Area Under Curve', command=lambda: sort_column(area_table, 'Area Under Curve'))
area_table.heading('Max Mag', text='Max Mag', command=lambda: sort_column(area_table, 'Max Mag'))
area_table.heading('Min Mag', text='Min Mag', command=lambda: sort_column(area_table, 'Min Mag'))
area_table.pack(pady=10)

def sort_column(treeview, col):
    items = [(treeview.item(item)['values'], item) for item in treeview.get_children()]
    items.sort(key=lambda x: x[0][columns.index(col)], reverse=False)
    for index, (values, item) in enumerate(items):
        treeview.move(item, '', index)

# Function to plot data
def plot_data():
    min_freq = min_freq_slider.get() * 1e9
    max_freq = max_freq_slider.get() * 1e9

    # Clear existing plots and table
    ax_magnitude.clear()
    ax_phase.clear()
    area_table.delete(*area_table.get_children())

    # Load data based on the toggle state
    use_random = toggle_random.get() == 1
    load_data(use_random=use_random)

    # Initialize a dictionary to hold area calculations
    area_results = {}

    # Plot Magnitude
    ax_magnitude.set_title('Magnitude vs Frequency')
    ax_magnitude.set_xlabel('Frequency (Hz)')
    ax_magnitude.set_ylabel('Magnitude (dB)')

    for group, color, label in zip(folder_paths, colors, group_labels):
        if check_vars[group].get() == 1:  # Check if the line is toggled on
            freqs = list(group_magnitudes[group].keys())
            mag_values = [group_magnitudes[group][freq] for freq in freqs if min_freq <= freq <= max_freq]
            filtered_freqs = [freq for freq in freqs if min_freq <= freq <= max_freq]

            ax_magnitude.plot(filtered_freqs, mag_values, marker='o', color=color)

            # Calculate area under the curve
            if len(filtered_freqs) > 1:
                area = trapezoid(mag_values, filtered_freqs)
                area_results[group] = {
                    'area': area,
                    'max_mag': max(mag_values),
                    'min_mag': min(mag_values)
                }

    ax_magnitude.axhline(0, color='grey', lw=0.5, ls='--')
    ax_magnitude.grid()

    # Add results to the table
    for group, metrics in area_results.items():
        area_table.insert("", "end", values=(group, f"{metrics['area']:.2f}", f"{metrics['max_mag']:.2f}", f"{metrics['min_mag']:.2f}"))

    # Plot Phase
    ax_phase.set_title('Phase vs Frequency')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (Degrees)')

    for group, color, label in zip(folder_paths, colors, group_labels):
        if check_vars[group].get() == 1:  # Check if the line is toggled on
            freqs = list(group_phases[group].keys())
            ph_values = [group_phases[group][freq] for freq in freqs if min_freq <= freq <= max_freq]
            filtered_freqs = [freq for freq in freqs if min_freq <= freq <= max_freq]

            ax_phase.plot(filtered_freqs, ph_values, marker='o', color=color)

    ax_phase.axhline(0, color='grey', lw=0.5, ls='--')
    ax_phase.grid()

    # Draw the updated figures
    canvas_magnitude.draw()
    canvas_phase.draw()
# Create checkbuttons for each group
check_frame = Frame(root)
check_frame.pack(pady=10)

check_vars = {group: IntVar(value=1) for group in folder_paths}
for group, label in zip(folder_paths, group_labels):
    checkbutton = Checkbutton(check_frame, text=label, variable=check_vars[group], command=plot_data)
    checkbutton.pack(side='left')

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

# Initial data load and plot
load_data()  # Load average data initially
plot_data()

# Start the Tkinter main loop
root.mainloop()