# End-to-End Customer Churn Prediction System

This project is an end-to-end Machine Learning pipeline designed to predict customer churn using the Telco Customer Churn dataset.

## 🚀 Project Overview

Customer churn is a critical metric for subscription-based businesses. This system predicts whether a customer is likely to leave ("churn") based on their demographics, usage patterns, and account information. 

**Key Features of this Project:**
- **Full ML Pipeline:** Covers data loading, preprocessing, EDA, feature engineering, model training, hyperparameter tuning, and deployment.
- **Model Comparison:** Evaluates both Logistic Regression and Random Forest.
- **Handling Categorical Data:** Uses One-Hot Encoding to robustly transform all categorical variables.
- **Hyperparameter Tuning:** Uses `GridSearchCV` to optimize the Random Forest model.
- **Interactive UI:** Deploys the predictive model using an interactive Streamlit dashboard.

## 🧰 Tech Stack
- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Pickle
- **Web UI:** Streamlit

## 📁 Project Structure
```text
churn-project/
│
├── data/
│   └── churn.csv                # The Telco Customer Churn dataset
├── notebooks/                   
│   ├── eda_churn_distribution.png # EDA visualization
│   └── feature_importance.png     # Feature importance plot
├── model/
│   └── churn_model.pkl          # Saved Random Forest model & features
├── train.py                     # Training script (preprocessing, tuning, evaluation)
├── app.py                       # Streamlit Web Application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## ⚙️ How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed. Run the following command:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script. This will process the data, train the models, save visualizations to `notebooks/`, and create the `churn_model.pkl` file.
```bash
python train.py
```

### 3. Run the Streamlit UI
Launch the interactive web application to make predictions:
```bash
streamlit run app.py
```

## 📊 Key Findings & Model Evaluation
During training, we evaluated models using Accuracy, Precision, Recall, and F1-Score. For churn prediction, **Recall** is often highly important, as we want to capture as many potential churners as possible, even at the cost of some false positives.

*The exact metrics are printed to the console when running `train.py`.*

**Feature Importance:**
The Random Forest model identified features such as `TotalCharges`, `MonthlyCharges`, `Tenure`, and `Contract type` as the most influential indicators of churn. (See `notebooks/feature_importance.png` after running the training script).

## 💡 What Makes This Project Stand Out
- **"I built an end-to-end ML pipeline from preprocessing to deployment"**
- **"Used Random Forest and improved performance via GridSearch tuning"**
- **"Handled categorical data using encoding techniques and ensured robust prediction input handling"**
- **"Evaluated model using comprehensive metrics and analyzed feature importance"**
