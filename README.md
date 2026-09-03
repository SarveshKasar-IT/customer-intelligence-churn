
# Customer Churn Prediction

## Project Overview

This project predicts whether a telecom customer is likely to churn using machine learning.

The project covers the complete machine learning workflow:

- Data understanding and preprocessing
- Exploratory data analysis
- Feature engineering
- Categorical variable encoding
- Train/test splitting
- Logistic Regression
- Decision Tree
- Random Forest
- Model evaluation
- Hyperparameter tuning
- Probability threshold optimization
- Feature importance analysis
- Model saving
- Streamlit deployment

## Dataset

The project uses the Telco Customer Churn dataset.

Target variable:

- `Churn`
  - `0` = Customer does not churn
  - `1` = Customer churns

Important features include:

- Tenure
- Monthly Charges
- Total Charges
- Contract type
- Internet Service
- Payment Method
- Online Security
- Tech Support
- Streaming services
- Customer demographics

## Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Random Forest was selected as the final model based on its overall predictive performance.

The final model was further improved using hyperparameter tuning and probability threshold optimization.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

ROC-AUC was emphasized because churn prediction is a classification problem where correctly identifying potential churners is important.

## Feature Importance

Feature importance analysis was performed to understand which customer characteristics contribute most to churn predictions.

Important predictors included factors such as:

- Tenure
- Monthly Charges
- Total Charges
- Internet Service
- Contract
- Payment Method
- Online Security
- Tech Support

## Streamlit Application

A Streamlit web application was created to allow users to enter customer information and receive:

- Churn probability
- Churn prediction
- Risk level
- Suggested customer retention actions

### Run the Application

From the project root directory:

    streamlit run app.py

The application will open in a browser at:

    http://localhost:8501

## Project Structure

    customer-intelligence-churn/
    |
    +-- app.py
    +-- README.md
    +-- requirements.txt
    |
    +-- data/
    |   +-- raw/
    |   |   +-- telco_customer_churn.csv
    |   +-- processed/
    |
    +-- models/
    |   +-- customer_churn_random_forest.pkl
    |   +-- churn_threshold.txt
    |   +-- feature_importance.csv
    |
    +-- notebooks/
    |   +-- 01_data_understanding.ipynb
    |
    +-- sql/
    +-- src/
    +-- tests/

## Business Value

The model can help a telecom company identify customers who are at higher risk of leaving.

Potential business actions include:

- Offering targeted retention discounts
- Improving customer support
- Providing service upgrades
- Promoting longer-term contracts
- Addressing customers with high monthly charges
- Proactively contacting high-risk customers

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## Conclusion

This project demonstrates an end-to-end customer churn prediction system, from raw customer data through machine learning model development and evaluation to an interactive deployed application.
