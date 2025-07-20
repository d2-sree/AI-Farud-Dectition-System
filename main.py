import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
import random

# Dataset file name
filename = 'Fraud Detection in Financial Transactions - DataSet.csv'
if not os.path.exists(filename):
    raise FileNotFoundError(f"File '{filename}' not found in the directory!")

# Load dataset
df = pd.read_csv(filename)

# Show data preview
print("🟢 First 5 Rows:")
print(df.head())

# Check for label column
label_col = 'is_Fraud'
print("\n🔍 Column Names:", df.columns.tolist())
if label_col not in df.columns:
    print(f"\n⚠️ '{label_col}' column not found. Adding synthetic fraud labels...")
    df[label_col] = [random.choices([0, 1], weights=[0.95, 0.05])[0] for _ in range(len(df))]
    df.to_csv(filename, index=False)
    print(f"✅ '{label_col}' column added and saved to same file.")

# Features and target
X = df.drop(label_col, axis=1)
y = df[label_col]

# Convert categorical columns to numerical
X = pd.get_dummies(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'fraud_model.pkl')
print("\n✅ Model saved as 'fraud_model.pkl'")
