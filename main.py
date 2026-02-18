import pandas as pd
import numpy as np

# 1. Load the Dataset
print("Loading dataset... this may take a moment.")
# Using the filename you provided: train_test_network.csv
df = pd.read_csv('train_test_network.csv', low_memory=False)

# 2. Inspect the Data
print(f"Dataset Loaded. Shape: {df.shape}")

# 3. List all columns to decide what to drop
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 4. Check for 'Object' type columns (Strings that need One-Hot Encoding)
print("\n--- CATEGORICAL COLUMNS (Need Encoding) ---")
print(df.select_dtypes(include=['object']).columns.tolist())