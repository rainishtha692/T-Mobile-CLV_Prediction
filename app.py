from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("xgboost_clv_model_final.pkl")

# Home route
@app.route("/")
def home():
    return "✅ CLV Prediction! Click <a href='/form'>/form</a> to try it."

# HTML Form with simplified raw inputs
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Basic conversions
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
        df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        df['tenure'] = df['tenure'].astype(float)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure'].replace(0, np.nan)
        df['IsSenior'] = df['SeniorCitizen']
        df['Churn'] = 0  # Assume active customer

        # One-hot encoding
        dummies = pd.get_dummies(df[['Contract', 'PaymentMethod', 'InternetService']], prefix_sep='_', drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df.drop(['Contract', 'PaymentMethod', 'InternetService'], axis=1, inplace=True)

        model_features = model.feature_names_in_
        for col in model_features:
            if col not in df:
                df[col] = 0
        df = df[model_features]

        prediction = model.predict(df)[0]
        result = float(round(prediction, 2))

        # Interpret the result
        if result < 1000:
            message = "🔴 Customer is at risk. Consider retention strategies."
        elif result < 2500:
            message = "🟡 Moderate CLV. Maintain engagement."
        else:
            message = "🟢 High-value customer. Consider loyalty programs."

        return f"""
        <h2>Predicted CLV: ${result}</h2>
        <h3>{message}</h3>
        <br><a href='/form'>Predict Another</a>
        """

    except Exception as e:
        return f"<h2>Error: {e}</h2><br><a href='/form'>Try Again</a>

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Basic feature preprocessing
        df = pd.DataFrame([data])

        # Convert binary
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
        df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

        # Convert numeric
        df['tenure'] = df['tenure'].astype(float)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['TotalCharges'] = df['TotalCharges'].astype(float)

        # Derived features
        df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure'].replace(0, np.nan)
        df['IsSenior'] = df['SeniorCitizen']
        df['Churn'] = 0  # Assume active customer for CLV estimation

        # One-hot encoding of selected categoricals
        dummies = pd.get_dummies(df[['Contract', 'PaymentMethod', 'InternetService']], prefix_sep='_', drop_first=False)
        df = pd.concat([df, dummies], axis=1)

        # Drop original string cols
        df.drop(['Contract', 'PaymentMethod', 'InternetService'], axis=1, inplace=True)

        # Align with model features
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in df:
                df[col] = 0  # fill missing engineered cols
        df = df[model_features]  # reorder

        prediction = model.predict(df)[0]
        result = float(round(prediction, 2))
        return f"<h2>Predicted CLV: ${result}</h2><br><a href='/form'>Try Again</a>"

    except Exception as e:
        return f"<h2>Error: {e}</h2><br><a href='/form'>Try Again</a>"

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
