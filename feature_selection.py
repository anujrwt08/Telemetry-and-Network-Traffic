import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn_genetic import GAFeatureSelectionCV
import time
import os

# --- CONFIGURATION ---
# We use a smaller subset for feature selection to make it run fast
# (Feature selection on 200k rows takes forever; 10k is enough to find the best patterns)
SUBSET_SIZE = 10000 

print("Loading balanced data...")
if not os.path.exists('balanced_network_data.csv'):
    print("Error: 'balanced_network_data.csv' not found.")
    exit()

df = pd.read_csv('balanced_network_data.csv')

# 1. Prepare Data for Selection
X = df.drop(columns=['label', 'type'])
y = df['label'] # We select features based on Binary Classification (Attack vs Normal)

# Take a random subset for speed
if len(X) > SUBSET_SIZE:
    print(f"Subsetting data to {SUBSET_SIZE} rows for faster Feature Selection...")
    # Stratified sample to keep class balance
    _, X_small, _, y_small = train_test_split(X, y, test_size=SUBSET_SIZE, stratify=y, random_state=42)
else:
    X_small, y_small = X, y

print(f"Feature Selection Input Shape: {X_small.shape}")

# --- 2. BIO-INSPIRED SELECTION (Genetic Algorithm) ---
# This mimics the "Crocodile/Bee" evolutionary search
print("\n--- STAGE 1: Bio-Inspired Evolutionary Search (GA) ---")
print("This may take a few minutes...")

clf = DecisionTreeClassifier()
evolved_selector = GAFeatureSelectionCV(
    estimator=clf,
    scoring="accuracy",
    population_size=10,  # Small population for speed (Paper uses more)
    generations=5,       # Fewer generations for local PC
    crossover_probability=0.8,
    mutation_probability=0.1,
    verbose=True,
    keep_top_k=4,
    n_jobs=-1            # Use all CPU cores
)

start_time = time.time()
evolved_selector.fit(X_small, y_small)
print(f"GA Completed in {time.time() - start_time:.2f} seconds.")

# Get the features selected by the Bio-Inspired stage
selected_indices = np.where(evolved_selector.support_)[0]
selected_features_ga = X.columns[selected_indices]
print(f"GA selected {len(selected_indices)} features.")

# --- 3. RECURSIVE FEATURE ELIMINATION (RFE) ---
# The paper uses RFE to refine the output of the bio-inspired stage
print("\n--- STAGE 2: Recursive Feature Elimination (RFE) ---")

# We only feed the GA-selected features into RFE
X_refined = X_small[selected_features_ga]

# RFE to select top 15 features (or half of what GA found)
n_features_to_select = min(15, len(selected_features_ga))
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features_to_select)
rfe.fit(X_refined, y_small)

# Final Features
final_mask = rfe.support_
final_features = selected_features_ga[final_mask]

print("\n=== FINAL SELECTED FEATURES ===")
print(list(final_features))

# --- 4. SAVE DATASET WITH ONLY SELECTED FEATURES ---
print("\nSaving final dataset with selected features...")

# We go back to the FULL dataset (200k rows) but keep only these columns
# We must keep 'label' and 'type' for the final classification training
cols_to_keep = list(final_features) + ['label', 'type']
final_df = df[cols_to_keep]

final_df.to_csv('final_selected_data.csv', index=False)
print(f"Saved 'final_selected_data.csv' with shape {final_df.shape}")
print("Ready for Final Classification (Phase 4).")