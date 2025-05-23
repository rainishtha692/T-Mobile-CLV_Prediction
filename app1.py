#!/usr/bin/env python
# coding: utf-8

# expected_cols = [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
#  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
#  'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges',
#  'TotalCharges', 'Contract_One year', 'Contract_Two year',
#  'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
#  'PaymentMethod_Mailed check', 'InternetService_Fiber optic',
#  'InternetService_No', 'AvgMonthlySpend', 'TotalAddons',
#  'TenureGroup_13-24m', 'TenureGroup_25-48m', 'TenureGroup_49-60m',
#  'TenureGroup_61-72m', 'HasFiberOptic', 'IsAutoPay', 'IsSenior',
#  'MultipleLines_Yes', 'Contract_Two year.1', 'PaymentMethod_E-Check',
#  'PaymentMethod_Mail Check', 'InternetService_No Internet' ]
# 
# df = df[expected_cols]  # ensure correct column order

# ##from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# 
# app = Flask(__name__)
# model = joblib.load("xgboost_clv_model_final.pkl")
# 
# @app.route("/")
# def home():
#     return "✅ CLV Prediction API is Running!"
# 
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         df = pd.DataFrame([data])
#         prediction = model.predict(df)[0]
#         return jsonify({"Predicted_CLV": round(prediction, 2)})
#     except Exception as e:
#         return jsonify({"error": str(e)})
# 
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

# In[1]:


from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('xgboost_clv_model_final.pkl', 'rb'))

@app.route("/")
def home():
    return "✅ CLV Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()           # This expects a JSON dict
        df = pd.DataFrame([data])           # Converts it into a DataFrame

        prediction = model.predict(df)[0]
        return jsonify({"Predicted_CLV": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# In[ ]:




