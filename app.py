from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("xgboost_clv_model_final.pkl")

# Homepage route
@app.route("/")
def home():
    return "✅ CLV Prediction API is Running! Go to <a href='/form'>/form</a> to try it."

# Form route
@app.route("/form")
def form():
    return render_template_string('''
    <html>
    <head>
        <title>CLV Prediction</title>
        <style>
            body { font-family: Arial; background-color: #f2f2f2; padding: 40px; }
            .form-box { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px #ccc; width: 400px; }
            label { display: block; margin-top: 15px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; margin-top: 5px; }
            input[type=submit] { background-color: #4CAF50; color: white; cursor: pointer; margin-top: 20px; border: none; }
            input[type=submit]:hover { background-color: #45a049; }
        </style>
    </head>
    <body>
        <div class="form-box">
            <h2>CLV Prediction Form</h2>
            <form action="/predict" method="post">
                <label>Gender:</label>
                <select name="gender"><option>Male</option><option>Female</option></select>
                <label>Senior Citizen:</label>
                <select name="SeniorCitizen"><option value="0">No</option><option value="1">Yes</option></select>
                <label>Partner:</label>
                <select name="Partner"><option value="Yes">Yes</option><option value="No">No</option></select>
                <label>Dependents:</label>
                <select name="Dependents"><option value="Yes">Yes</option><option value="No">No</option></select>
                <label>Tenure:</label>
                <input name="tenure" type="number" required />
                <label>Monthly Charges:</label>
                <input name="MonthlyCharges" type="number" step="0.01" required />
                <label>Total Charges:</label>
                <input name="TotalCharges" type="number" step="0.01" required />
                <label>Contract:</label>
                <select name="Contract">
                    <option>Month-to-month</option>
                    <option>One year</option>
                    <option>Two year</option>
                </select>
                <label>Internet Service:</label>
                <select name="InternetService">
                    <option>DSL</option>
                    <option>Fiber optic</option>
                    <option>No</option>
                </select>
                <label>Payment Method:</label>
                <select name="PaymentMethod">
                    <option>Electronic check</option>
                    <option>Mailed check</option>
                    <option>Bank transfer (automatic)</option>
                    <option>Credit card (automatic)</option>
                </select>
                <input type="submit" value="Predict CLV" />
            </form>
        </div>
    </body>
    </html>
    ''')

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Preprocessing
        df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
        df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        df['tenure'] = df['tenure'].astype(float)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['TotalCharges'] = df['TotalCharges'].astype(float)

        # Derived features
        df['AvgMonthlySpend'] = df['TotalCharges'] / df['tenure'].replace(0, np.nan)
        df['IsSenior'] = df['SeniorCitizen']
        df['Churn'] = 0

        # One-hot encoding
        dummies = pd.get_dummies(df[['Contract', 'PaymentMethod', 'InternetService']], prefix_sep='_', drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df.drop(['Contract', 'PaymentMethod', 'InternetService'], axis=1, inplace=True)

        # Align with model
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in df:
                df[col] = 0
        df = df[model_features]

        # Predict
        prediction = model.predict(df)[0]
        result = float(round(prediction, 2))

        # Interpretation
        if result < 1000:
            message = "🔴 Customer is at risk. Consider retention strategies."
        elif result < 2500:
            message = "🟡 Moderate CLV. Maintain engagement."
        else:
            message = "🟢 High-value customer. Consider loyalty offers."

        return f"<h2>Predicted CLV: ${result}</h2><h3>{message}</h3><br><a href='/form'>Try Again</a>"

    except Exception as e:
        return f"<h2>Error: {e}</h2><br><a href='/form'>Try Again</a>"

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
     

  
