import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
# Step 1: Generate Simulated Data
data = {
    'equipment_id': range(1, 101),
    'temperature': [round(x, 2) for x in np.random.uniform(50, 100, 100)],
    'vibration': [round(x, 2) for x in np.random.uniform(0, 10, 100)],
    'usage_hours': [round(x) for x in np.random.uniform(0, 5000, 100)],
    'maintenance_interval': [round(x) for x in np.random.uniform(100, 1000, 100)],
    'maintenance_duration': [round(x, 2) for x in np.random.uniform(1, 24, 100)],
    'failure': [np.random.choice([0, 1], p=[0.9, 0.1]) for _ in range(100)]
}

# Step 2: Load the Data
df = pd.DataFrame(data)

# Step 3: Data Cleaning
# Check for missing values
print(df.isnull().sum())

# Step 4: Data Preprocessing
# Scale numerical features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['temperature', 'vibration', 'usage_hours', 'maintenance_interval', 'maintenance_duration']])
df[['temperature', 'vibration', 'usage_hours', 'maintenance_interval', 'maintenance_duration']] = scaled_features

# Convert timestamps (not applicable in this simulated scenario)

# Step 5: Exploratory Data Analysis (EDA)
# Visualize sensor readings
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.hist(df['temperature'], bins=20, color='skyblue', edgecolor='black')
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')

plt.subplot(2, 1, 2)
plt.hist(df['vibration'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Vibration Distribution')
plt.xlabel('Vibration')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 6: Handle Imbalanced Classes (Optional)
# Not applicable in this simulated scenario

# Step 7: Save Preprocessed Data
df.to_csv('preprocessed_data.csv', index=False)
