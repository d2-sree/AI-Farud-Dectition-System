import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 🔹 Step 1: Load dataset
df = pd.read_csv("Fraud Detection in Financial Transactions - DataSet.csv")

# 🔹 Step 2: Create smart fraud labels using logic
def label_fraud(row):
    if row['LoginAttempts'] > 3:
        return 1
    elif row['TransactionAmount'] > 10000 and row['AccountBalance'] < 1000:
        return 1
    elif row['TransactionDuration'] < 2 and row['TransactionAmount'] > 5000:
        return 1
    elif row['CustomerAge'] < 20 and row['TransactionAmount'] > 3000:
        return 1
    else:
        return 0

df['isFraud'] = df.apply(label_fraud, axis=1)

# 🔹 Step 3: Save updated dataset (optional)
df.to_csv("Fraud Detection in Financial Transactions - DataSet.csv", index=False)

# 🔹 Step 4: Select features (drop non-useful or ID/time columns)
X = df.drop(columns=['isFraud', 'TransactionID', 'TransactionDate', 'PreviousTransactionDate'])
y = df['isFraud']

# 🔹 Step 5: Handle categorical variables
X = pd.get_dummies(X)

# 🔹 Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 7: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Step 8: Save model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved as fraud_model.pkl")
