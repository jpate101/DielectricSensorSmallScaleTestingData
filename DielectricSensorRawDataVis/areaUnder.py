import numpy as np
import pandas as pd
from scipy.integrate import simps
import os
from tkinter import Tk, Label, Button, filedialog, Text, Scrollbar

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

def calculate_area(file_path):
    frequencies, s11_real, s11_imag = read_s_param_file(file_path)
    
    area_real = simps(s11_real, frequencies)
    area_imag = simps(s11_imag, frequencies)
    
    return area_real, area_imag

def process_group_files(folder_path):
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.s1p'):
            file_path = os.path.join(folder_path, file_name)
            area_real, area_imag = calculate_area(file_path)
            results.append({
                'File Name': file_name,
                'Area Under Real Part (S11)': area_real,
                'Area Under Imaginary Part (S11)': area_imag
            })
    
    return pd.DataFrame(results)

def load_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        results_df = process_group_files(folder_path)
        display_results(results_df)

def display_results(results_df):
    output_text.delete(1.0, 'end')  # Clear previous output
    output_text.insert('end', results_df.to_string(index=False))

# Set up the main Tkinter window
root = Tk()
root.title("S-Parameter Area Calculator")

# Create GUI elements
label = Label(root, text="Select Folder Containing S1P Files:")
label.pack(pady=10)

select_button = Button(root, text="Load Folder", command=load_folder)
select_button.pack(pady=5)

# Text area for displaying results
output_text = Text(root, height=20, width=120)  # Adjusted width to 120
output_text.pack(pady=10)

# Scrollbar for the text area
scrollbar = Scrollbar(root, command=output_text.yview)
scrollbar.pack(side='right', fill='y')
output_text.config(yscrollcommand=scrollbar.set)

# Start the Tkinter main loop
root.mainloop()