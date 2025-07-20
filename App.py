from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")
model_columns = model.feature_names_in_

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # Convert values to appropriate types
    for col in ['TransactionAmount', 'TransactionDuration', 'LoginAttempts', 'CustomerAge', 'AccountBalance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add missing columns from training
    df = pd.get_dummies(df)
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]

    prediction = model.predict(df)[0]
    result = "Fraud" if prediction == 1 else "Not Fraud"
    return render_template("index.html", prediction_text=f"🧾 Prediction: {result}")


# ✅ Move dashboard route above the main block
@app.route("/dashboard")
def dashboard():
    df = pd.read_csv("Fraud Detection in Financial Transactions - DataSet.csv")

    fraud_count = df[df['isFraud'] == 1].shape[0]
    non_fraud_count = df[df['isFraud'] == 0].shape[0]

    # ✅ Convert to string for JSON compatibility
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    frauds_by_day = df[df['isFraud'] == 1]['TransactionDate'].dt.strftime('%Y-%m-%d').value_counts().sort_index()

    return render_template(
        "dashboard.html",
        fraud_count=fraud_count,
        non_fraud_count=non_fraud_count,
        frauds_by_day=frauds_by_day.to_dict()
    )
if __name__ == "__main__":
    app.run(debug=True)
