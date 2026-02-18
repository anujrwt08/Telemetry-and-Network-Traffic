import matplotlib
matplotlib.use('Agg') # <--- THIS FIXES THE ERROR (No popup window needed)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 1. Load Data
print("Loading data...")
df = pd.read_csv('final_selected_data.csv')
X = df.drop(columns=['label', 'type'])
y = df['label']

# 2. Retrain the Best Model (XGBoost)
print("Training XGBoost for visualization...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 4. Plot & Save
print("Generating plot...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'])
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.title('Confusion Matrix: XGBoost on ToN_IoT (Replicated Paper)')

# SAVE instead of SHOW
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Success! The chart has been saved as '{output_filename}'. Open it to see your results.")