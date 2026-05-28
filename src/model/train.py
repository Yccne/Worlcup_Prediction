import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 1. Base model configuration
    xgb_base = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )
    
    # 2. Hyperparameter Grid
    # The computer will test random combinations of these settings to find the absolute best fit.
    param_distributions = {
        'max_depth': [2, 3, 4, 5, 6],           # How deep the trees can grow
        'learning_rate': [0.01, 0.05, 0.1, 0.2],# How fast the model learns
        'n_estimators': [100, 200, 300, 500],   # Number of trees to build
        'subsample': [0.6, 0.8, 1.0],           # Fraction of data used to train each tree
        'colsample_bytree': [0.6, 0.8, 1.0],    # Fraction of features used for each tree
        'min_child_weight': [1, 3, 5]           # Minimum instances needed to split a node
    }
    
    print("Starting Hyperparameter Tuning (Testing 15 configurations)...")
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=15,             # Test 15 random combinations (keep it relatively fast)
        scoring='neg_log_loss',# Optimize for the best Log Loss
        cv=tscv,               # Use chronological TimeSeriesSplit
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    search.fit(X_train, y_train)
    print(f"\nBest parameters found:\n{search.best_params_}")
    
    # 3. Create a fresh model using the mathematically best parameters
    best_xgb = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
        **search.best_params_
    )
    
    # 4. Probability Calibration
    print("\nCalibrating the optimized model with TimeSeriesSplit...")
    calibrated_model = CalibratedClassifierCV(estimator=best_xgb, cv=tscv, method='isotonic')
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
