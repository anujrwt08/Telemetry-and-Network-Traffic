import pandas as pd

# Load the processed data
print("Loading processed data...")
df = pd.read_csv('processed_network_data.csv')

# Check the counts of the 'type' column
print("\n--- ATTACK TYPE DISTRIBUTION ---")
print(df['type'].value_counts())

print("\n--- PERCENTAGE ---")
print(df['type'].value_counts(normalize=True) * 100)