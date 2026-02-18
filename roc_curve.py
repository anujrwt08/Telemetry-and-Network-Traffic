import matplotlib
matplotlib.use('Agg') # Fixes the TclError for saving images
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load Data
print("Loading selected feature data...")
df = pd.read_csv('final_selected_data.csv')
X = df.drop(columns=['label', 'type'])
y = df['label']

# Split 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Define Models
models_dict = {
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": lgb.LGBMClassifier(verbose=-1)
}

# 3. Plot Setup
plt.figure(figsize=(10, 8))

# 4. Train & Plot Machine Learning Models
for name, model in models_dict.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    # Get probabilities for the positive class (Attack)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

# 5. Train & Plot DNN
print("Training DNN...")
model_dnn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
model_dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

y_prob_dnn = model_dnn.predict(X_test).ravel()
fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_prob_dnn)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
plt.plot(fpr_dnn, tpr_dnn, label=f'DNN (AUC = {roc_auc_dnn:.4f})')

# 6. Final Plot Formatting
plt.plot([0, 1], [0, 1], 'k--', lw=2) # Diagonal random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Save
filename = 'roc_curve.png'
plt.savefig(filename)
print(f"Success! ROC Curve saved as '{filename}'.")