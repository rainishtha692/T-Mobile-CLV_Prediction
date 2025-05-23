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
    return "✅ CLV Prediction API is Running! Go to <a href='/form'>/form</a> to try it."

# HTML Form with simplified raw inputs
@app.route("/form")
def form():
    return render_template_string('''
    <h2>CLV Prediction Form (Simplified)</h2>
    <form action="/predict" method="post">
        <label>Gender: <select name="gender"><option value="Male">Male</option><option value="Female">Female</option></select></label><br>
        <label>Senior Citizen: <select name="SeniorCitizen"><option value="0">No</option><option value="1">Yes</option></select></label><br>
        <label>Partner: <select name="Partner"><option value="Yes">Yes</option><option value="No">No</option></select></label><br>
        <label>Dependents: <select name="Dependents"><option value="Yes">Yes</option><option value="No">No</option></select></label><br>
        <label>Tenure: <input name="tenure" type="number" required /></label><br>
        <label>Monthly Charges: <input name="MonthlyCharges" type="number" step="0.01" required /></label><br>
        <label>Total Charges: <input name="TotalCharges" type="number" step="0.01" required /></label><br>
        <label>Contract: <select name="Contract">
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option></select></label><br>
        <label>Internet Service: <select name="InternetService">
            <option value="DSL">DSL</option>
            <option value="Fiber optic">Fiber optic</option>
            <option value="No">No</option></select></label><br>
        <label>Payment Method: <select name="PaymentMethod">
            <option value="Electronic check">Electronic check</option>
            <option value="Mailed check">Mailed check</option>
            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
            <option value="Credit card (automatic)">Credit card (automatic)</option></select></label><br>
        <input type="submit" value="Predict CLV" />
    </form>
    ''')

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