This project predicts customer churn using machine learning models on a Telco dataset. It helps the business identify high-risk customers and take proactive retention steps.

📁 Project Structure
. ├── app/ # Flask App or Streamlit App (Optional) │ └── model.pkl # Trained ML model ├── notebooks/ # EDA and development notebooks ├── data/ # Original and cleaned datasets ├── src/ # Source code (functions, pipeline) │ └── churn_model.py ├── Dockerfile # For containerizing the app ├── requirements.txt # Python dependencies └── README.md

🧠 Models Used
Logistic Regression
Random Forest
XGBoost
Best Model: ✅ XGBoost with ~85% accuracy and high recall for churn prediction.

📊 Dashboard Insights
57% of customers were identified as high risk.
Contracts, MonthlyCharges, and Tenure were key churn indicators.
📈 Business Recommendations
Focus on long-term contracts with promotions
Target high-churn segments with loyalty offers
Improve customer support in early tenure period
🚀 Deployment Options
Local: python app.py (Flask)
Docker: docker build -t churn-app . && docker run -p 5000:5000 churn-app
Cloud: Upload to Render / Heroku / AWS (Optional)
💻 Setup Instructions
# Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app (if Flask)
python app.py
