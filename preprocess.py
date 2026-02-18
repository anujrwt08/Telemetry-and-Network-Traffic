import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# 1. Load Data
print("Loading data...")
df = pd.read_csv('train_test_network.csv', low_memory=False)

# --- A. DATA CLEANSING ---
# Drop identifiers and high-cardinality text columns
cols_to_drop = [
    'src_ip', 'dst_ip', 'src_port', 'dst_port',
    'dns_query', 'ssl_subject', 'ssl_issuer',
    'http_uri', 'http_user_agent',
    'weird_name', 'weird_addl', 'weird_notice',
    'http_orig_mime_types', 'http_resp_mime_types',
    'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', # DNS flags often useless if high nulls
    'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established'
]

# Only drop columns that actually exist
existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
df.drop(columns=existing_drop_cols, inplace=True)
print(f"Dropped {len(existing_drop_cols)} columns.")

# --- B. HANDLE MISSING VALUES (The Fix) ---
# Replace empty strings with NaN first
df.replace(['-', ''], np.nan, inplace=True)

# Separate columns by type
numeric_cols = df.select_dtypes(include=['number']).columns
string_cols = df.select_dtypes(exclude=['number']).columns

# Fill NaNs: 0 for numbers, "Unknown" for text
df[numeric_cols] = df[numeric_cols].fillna(0)
df[string_cols] = df[string_cols].fillna("Unknown")

print("Missing values handled.")

# --- C. SEPARATE TARGETS FROM FEATURES ---
# Save targets
y_binary = df['label']
y_type = df['type']

# Drop targets from features
X = df.drop(columns=['label', 'type'])

# --- D. ONE-HOT ENCODING ---
print("Performing One-Hot Encoding...")
# Identify categorical columns again (after dropping)
categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical Columns to Encode: {categorical_cols}")

# Apply One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(X[categorical_cols])

# Create DataFrame for encoded data
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# --- E. SCALING (Min-Max) ---
print("Performing Min-Max Scaling...")
scaler = MinMaxScaler()
scaled_numerical = scaler.fit_transform(X[numerical_cols])
scaled_df = pd.DataFrame(scaled_numerical, columns=numerical_cols)

# --- F. RECOMBINE ---
# Reset index to avoid misalignment
scaled_df.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)
y_type.reset_index(drop=True, inplace=True)
y_binary.reset_index(drop=True, inplace=True)

final_df = pd.concat([scaled_df, encoded_df], axis=1)

# Add targets back
final_df['type'] = y_type
final_df['label'] = y_binary

print("\n--- PRE-PROCESSING COMPLETE ---")
print(f"Original Shape: (211043, 44)")
print(f"New Shape: {final_df.shape}")

# Save to CSV
print("Saving to 'processed_network_data.csv'...")
final_df.to_csv('processed_network_data.csv', index=False)
print("File saved successfully.")