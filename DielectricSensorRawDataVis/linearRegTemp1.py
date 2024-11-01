import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Function to read S-parameter files
def read_s_param_file(file_path):
    frequencies = []
    s11_real = []
    s11_imag = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('!') or not line.strip():
                continue
            try:
                values = list(map(float, line.split()))
                frequencies.append(values[0] / 1e9)  # Convert Hz to GHz
                s11_real.append(values[1])
                s11_imag.append(values[2])
            except ValueError:
                continue

    return np.array(frequencies), np.array(s11_real), np.array(s11_imag)

# Function to load data from directories
def load_data(wet_folder, dry_folder):
    data = []
    labels = []

    for folder, label in [(wet_folder, 'wet'), (dry_folder, 'dry')]:
        for filename in os.listdir(folder):
            if filename.endswith('.s1p'):
                file_path = os.path.join(folder, filename)
                frequencies, s11_real, s11_imag = read_s_param_file(file_path)
                for freq, s11_r, s11_i in zip(frequencies, s11_real, s11_imag):
                    data.append([freq, s11_r, s11_i])
                    labels.append(label)

    return np.array(data), np.array(labels)

# Load data
wet_folder = 'Group2'  # Update with your actual path
dry_folder = 'Group3'  # Update with your actual path
data, labels = load_data(wet_folder, dry_folder)

# Create a DataFrame
df = pd.DataFrame(data, columns=['Frequency', 'S11_Real', 'S11_Imag'])
df['Label'] = labels

# Fit Random Forest model
X = df[['Frequency', 'S11_Real', 'S11_Imag']]
y = df['Label']
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importance
importance = model.feature_importances_
print(f"Feature Importance: {importance}")

# Calculate impact
df['Impact'] = np.abs(df['S11_Real']) + np.abs(df['S11_Imag'])

# Summarize duplicate frequencies
summary_df = df.groupby('Frequency').agg({
    'S11_Real': 'sum',
    'S11_Imag': 'sum',
    'Impact': 'sum',
    'Label': 'first'  # Keep the first label encountered
}).reset_index()

# Display top impactful frequencies
top_impactful = summary_df.sort_values(by='Impact', ascending=False)

# Print summarized impactful frequencies
print("\nTop Impactful Frequencies (GHz) with their S11 values (summed):")
print(top_impactful)

# Optional: Plotting the results
plt.figure(figsize=(12, 6))
for label in top_impactful['Label'].unique():
    subset = top_impactful[top_impactful['Label'] == label]
    plt.scatter(subset['Frequency'], subset['S11_Real'], label=f'S11 Real ({label})', alpha=0.5)
    plt.scatter(subset['Frequency'], subset['S11_Imag'], label=f'S11 Imag ({label})', alpha=0.5, marker='x')

plt.title('Summed S11 Real and Imaginary Parts vs Frequency')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 Value (Summed)')
plt.legend()
plt.grid()
plt.show()

# Display the top frequencies based on importance from the model
top_features = pd.DataFrame({
    'Feature': ['Frequency', 'S11_Real', 'S11_Imag'],
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(top_features)