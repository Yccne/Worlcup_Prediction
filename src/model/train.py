import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

def load_data():
    # Load the pre-split data from Phase 1
    train_df = pd.read_csv("data/dataset1/train_features.csv")
    test_df = pd.read_csv("data/dataset1/test_features.csv")
    
    # CRITICAL: For TimeSeriesSplit, our training data MUST be sorted chronologically
    train_df = train_df.sort_values("date").reset_index(drop=True)
    
    # Define our input features (X) and what we want to predict (y)
    features = [
        "elo_diff", "form_diff", "h2h_home_win_rate", 
        "goal_diff_diff", "neutral_venue", "tournament_stage"
    ]
    
    X_train = train_df[features]
    y_train = train_df["target"]
    
    X_test = test_df[features]
    y_test = test_df["target"]
    
    return X_train, y_train, X_test, y_test, test_df

def train_model(X_train, y_train):
    # 1. Define the base model
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=4,         
        learning_rate=0.05,  
        n_estimators=200,    
        random_state=42      
    )
    
    # 2. TimeSeriesSplit
    # Splits the data into 5 chunks chronologically. 
    # The model trains on past chunks and evaluates on future chunks to prevent data leakage.
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 3. Probability Calibration
    # Wraps the XGBoost model. It uses the TimeSeriesSplit to figure out exactly how
    # overconfident the XGBoost model is, and applies 'isotonic' regression to fix the percentages.
    calibrated_model = CalibratedClassifierCV(estimator=base_model, cv=tscv, method='isotonic')
    
    print("Training and Calibrating XGBoost Model with TimeSeriesSplit...")
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    
    print("\n=== Model Evaluation (2022 World Cup Test Set) ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Log Loss: {ll:.4f} (Lower means probabilities are more accurate)")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"], zero_division=0))
    
    return y_proba

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, test_df = load_data()
    
    model = train_model(X_train, y_train)
    probabilities = evaluate_model(model, X_test, y_test)
    
    print("\n=== Example Prediction (Calibrated Output) ===")
    sample_match = test_df.iloc[0]
    sample_probs = probabilities[0]
    
    print(f"Match: {sample_match['home_team']} vs {sample_match['away_team']}")
    print(f"  - Home Win Probability: {sample_probs[0]*100:.1f}%")
    print(f"  - Draw Probability:     {sample_probs[1]*100:.1f}%")
    print(f"  - Away Win Probability: {sample_probs[2]*100:.1f}%")
    print(f"Actual Outcome: {['Home Win', 'Draw', 'Away Win'][int(sample_match['target'])]}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/worldcup_xgb.joblib")
    print("\n[OK] Calibrated model saved to models/worldcup_xgb.joblib")
