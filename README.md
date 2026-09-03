# 📊 Customer Churn Intelligence

An end-to-end **machine learning customer churn prediction project** built with Python, Scikit-learn, and Streamlit.

The project analyzes customer characteristics and service information to predict the probability that a customer will churn and provides actionable retention recommendations.

## 🚀 Live Demo

👉 **Streamlit App:**

PASTE_YOUR_STREAMLIT_APP_URL_HERE

## 📌 Project Overview

Customer churn is an important business problem because retaining an existing customer can be more cost-effective than acquiring a new one.

This project builds a machine learning solution that:

- Analyzes customer and service characteristics
- Preprocesses categorical and numerical data
- Compares multiple classification algorithms
- Tunes a Random Forest model
- Optimizes the prediction threshold for F1-score
- Predicts individual customer churn probability
- Classifies customers into Low, Medium, and High risk
- Provides recommended retention actions
- Deploys the final model through Streamlit Community Cloud

## 🧠 Machine Learning Workflow

Raw Customer Data → Data Cleaning → Feature Engineering → One-Hot Encoding → Train/Test Split → Model Comparison → Random Forest Tuning → Threshold Optimization → Final Model → Streamlit Deployment

## 📂 Project Structure

customer-intelligence-churn/
├── app.py
├── data/raw/telco_customer_churn.csv
├── models/customer_churn_random_forest.pkl
├── models/churn_threshold.txt
├── models/feature_importance.csv
├── notebooks/01_data_understanding.ipynb
├── README.md
└── requirements.txt

## 📊 Models Compared

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 79.84% | 65.00% | 52.14% | 57.86% | 0.8423 |
| Decision Tree | 79.42% | 63.12% | 54.01% | 58.21% | 0.8272 |
| Random Forest | **80.48%** | **66.67%** | 52.94% | **59.02%** | **0.8450** |

The Random Forest model provided the strongest overall performance based on ROC-AUC and F1-score among the evaluated baseline models.

## 🌲 Final Model

The final solution uses a **Random Forest Classifier** with hyperparameter tuning.

Key steps included:

- GridSearchCV
- 5-fold cross-validation
- ROC-AUC based model selection
- Prediction threshold optimization
- Feature importance analysis

The optimized threshold is stored separately in `models/churn_threshold.txt` so the deployed application can use the selected decision threshold.

## 🔎 Important Churn Drivers

Feature importance analysis identified important customer and service-related variables, including factors related to:

- Customer tenure
- Monthly charges
- Total charges
- Internet service type
- Payment method
- Contract type
- Service subscriptions

Feature importance should be interpreted as model behavior rather than proof of causation.

## 🌐 Streamlit Application

The deployed application allows users to enter a customer's:

- Personal information
- Tenure
- Service subscriptions
- Contract information
- Payment method
- Monthly charges
- Total charges

The application produces:

### 🔮 Churn Prediction

- Churn probability
- Predicted churn / no-churn outcome
- Low / Medium / High risk classification
- Customer summary
- Retention recommendations

### 📈 Model Insights

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve
- Feature importance

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Git & GitHub

## ⚙️ Run Locally

Clone the repository:

`git clone https://github.com/SarveshKasar-IT/customer-intelligence-churn.git`

Navigate into the project:

`cd customer-intelligence-churn`

Create a virtual environment:

`python -m venv .venv`

Activate it on Windows:

`.venv\\Scripts\\activate`

Install dependencies:

`pip install -r requirements.txt`

Run the application:

`streamlit run app.py`

## 📈 Business Value

A churn prediction system can help a business:

- Identify customers who may be at risk of leaving
- Prioritize retention campaigns
- Allocate customer-success resources more efficiently
- Understand patterns associated with customer churn
- Develop targeted retention strategies

The model is intended as a **decision-support tool**, not as a replacement for business judgment.

## ⚠️ Limitations

- Model performance depends on the underlying dataset.
- Feature importance does not imply causation.
- Predictions represent estimated probabilities.
- Retention decisions should consider additional business context.
- The dataset is historical and may not represent current behavior.

## 👨‍💻 Author

**Darshak Shah**

Machine Learning Project — Customer Churn Prediction

GitHub: https://github.com/SarveshKasar-IT/customer-intelligence-churn

---

⭐ If you found this project useful, consider starring the repository.
