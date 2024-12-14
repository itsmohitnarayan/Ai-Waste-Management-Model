import pandas as pd
import numpy as np
import os

# Parameters for data generation
num_samples = 100000
condition_distribution = {
    "Normal": 60000,
    "Moderate": 30000,
    "Dangerous": 10000
}

# Helper function to generate data points
def generate_data(condition):
    if condition == "Normal":
        return [
            np.random.uniform(10, 25),  # Temperature
            np.random.uniform(20, 50),  # Humidity
            np.random.uniform(50, 200),  # Gas Level
            np.random.uniform(0.1, 0.8),  # Vibration
            condition
        ]
    elif condition == "Moderate":
        return [
            np.random.uniform(20, 35),
            np.random.uniform(40, 70),
            np.random.uniform(150, 300),
            np.random.uniform(0.6, 1.2),
            condition
        ]
    elif condition == "Dangerous":
        return [
            np.random.uniform(35, 50),
            np.random.uniform(70, 100),
            np.random.uniform(300, 500),
            np.random.uniform(1.0, 2.5),
            condition
        ]

# Generate dataset
data = []
for condition, count in condition_distribution.items():
    for _ in range(count):
        data.append(generate_data(condition))

# Create DataFrame
columns = ["Temperature", "Humidity", "Gas_Level", "Vibration", "Condition"]
df = pd.DataFrame(data, columns=columns)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Save to CSV in the same directory
output_path = os.path.join(current_dir, 'dataset.csv')
df.to_csv(output_path, index=False)

df.head(), df.shape
