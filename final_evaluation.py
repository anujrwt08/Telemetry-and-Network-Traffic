import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
print("Loading selected feature data...")
df = pd.read_csv('final_selected_data.csv')

# 1. Prepare Data
# We use 'label' (0/1) for binary classification (Attack vs Normal)
X = df.drop(columns=['label', 'type'])
y = df['label']

# Split 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape:  {X_test.shape}")

# --- 2. DEFINE MODELS ---
# We replicate the models used in the paper (excluding SVM as it is too slow for 200k rows on CPU)
models_dict = {
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(verbose=-1)
}

results = []

# --- 3. TRAIN & EVALUATE CLASSICAL MODELS ---
print("\n--- Training Machine Learning Models ---")

for name, model in models_dict.items():
    print(f"Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Time (s)": train_time
    })
    print(f"   -> Accuracy: {acc:.4f}")

# --- 4. TRAIN DNN (Deep Neural Network) ---
print("\n--- Training DNN (Deep Neural Network) ---")
# Paper architecture: Input -> Dense layers -> Output
model_dnn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start = time.time()
# Train for 10 epochs (fast but effective)
history = model_dnn.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
train_time = time.time() - start

# Predict (DNN outputs probabilities, so we round to 0 or 1)
y_pred_prob = model_dnn.predict(X_test)
y_pred_dnn = (y_pred_prob > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred_dnn)
prec = precision_score(y_test, y_pred_dnn)
rec = recall_score(y_test, y_pred_dnn)
f1 = f1_score(y_test, y_pred_dnn)

results.append({
    "Model": "DNN",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1,
    "Time (s)": train_time
})

# --- 5. FINAL COMPARISON TABLE ---
print("\n" + "="*60)
print("FINAL RESULTS (Replicating Table 1 of the Paper)")
print("="*60)
results_df = pd.DataFrame(results)
# Sort by Accuracy
results_df = results_df.sort_values(by='Accuracy', ascending=False)
print(results_df)
print("="*60)

# Save results
results_df.to_csv('final_model_comparison.csv', index=False)