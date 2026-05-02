import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def main():
    print("[START] Starting End-to-End Customer Churn Prediction Pipeline...")
    
    # 1. Load Data
    print("\n[DATA] Loading dataset...")
    data_path = "data/churn.csv"
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}. Please download it first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"[SUCCESS] Data loaded successfully. Shape: {df.shape}")
    
    # 2. Data Preprocessing
    print("\n[PROCESS] Preprocessing data...")
    # Drop customer ID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Handle TotalCharges (convert to numeric, coerce errors to NaN)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing with median or drop
        df.dropna(subset=['TotalCharges'], inplace=True)
        
    # Convert Churn Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # EDA: Save a simple plot
    print("\n[EDA] Generating EDA plots...")
    plt.figure(figsize=(6,4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Churn Distribution')
    plt.savefig('notebooks/eda_churn_distribution.png')
    plt.close()
    
    # Feature Engineering / Encoding
    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    # One-Hot Encode categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    features_list = X_encoded.columns.tolist()
    
    print(f"[SUCCESS] Preprocessing complete. Number of features: {len(features_list)}")
    
    # 3. Train-Test Split
    print("\n[SPLIT] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Model Training (Logistic Regression & Random Forest)
    print("\n[TRAIN] Training models...")
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(random_state=42)
    
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # 5. Model Evaluation
    print("\n[EVAL] Evaluating Models...")
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    
    print("--- Logistic Regression ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(classification_report(y_test, y_pred_lr))
    
    print("--- Random Forest (Default) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf))
    
    # 6. Hyperparameter Tuning (Random Forest)
    print("\n[TUNING] Tuning Random Forest Hyperparameters...")
    params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, n_jobs=-1, scoring='f1')
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"[SUCCESS] Best parameters: {grid.best_params_}")
    
    y_pred_best = best_model.predict(X_test)
    print("--- Tuned Random Forest ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    print(classification_report(y_test, y_pred_best))
    
    # Feature Importance Plot
    print("\n[EDA] Generating Feature Importance Plot...")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10
    
    plt.figure(figsize=(8,6))
    plt.title('Top 10 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features_list[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('notebooks/feature_importance.png')
    plt.close()
    
    # 7. Save Model and Feature Names
    print("\n[SAVE] Saving best model...")
    model_data = {
        "model": best_model,
        "features": features_list
    }
    # To support simple pickle.load directly returning the model as user requested,
    # we can save the model separately, and features in another file. 
    # Or, the user's snippet model.predict([[input]]) might fail if we one-hot encoded heavily.
    # We will save model_data as a dictionary to ensure app.py can reconstruct the exact dataframe structure.
    pickle.dump(model_data, open("model/churn_model.pkl", "wb"))
    print("DONE: Model saved to model/churn_model.pkl")
    
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
