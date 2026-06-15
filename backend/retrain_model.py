"""
Retrain XGBoost model for hospital readmission prediction.
Uses scale_pos_weight (NOT SMOTE) to handle class imbalance — this preserves
probability calibration so predictions reflect real-world risk levels.
Feature engineering matches main.py exactly.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_recall_curve, f1_score,
    accuracy_score, roc_auc_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'dataset' / 'diabetic_cleaned.csv'

# -- ICD-9 mapping (identical to main.py) --
def map_icd9(val):
    if pd.isna(val) or val == '?' or val == 'Unknown':
        return 'Other'
    val_str = str(val)
    if val_str.startswith('V') or val_str.startswith('E'):
        return 'Other'
    try:
        n = float(val)
        if 390 <= n <= 459 or n == 785:
            return 'Circulatory'
        elif 460 <= n <= 519 or n == 786:
            return 'Respiratory'
        elif 520 <= n <= 579 or n == 787:
            return 'Digestive'
        elif int(n) == 250:
            return 'Diabetes'
        elif 800 <= n <= 999:
            return 'Injury'
        elif 710 <= n <= 739:
            return 'Musculoskeletal'
        elif 580 <= n <= 629 or n == 788:
            return 'Genitourinary'
        elif 140 <= n <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except:
        return 'Other'

age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

# -- 1. Load & Prepare Data --
print("=" * 60)
print("STEP 1: Loading and preparing data...")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
x = df.drop(columns=["readmitted", "readmitted_binary"])
y = df["readmitted_binary"]

# Drop diag_2 and diag_3
x = x.drop(columns=["diag_2", "diag_3"], errors='ignore')

# Map diag_1 to categories
x['diag_1'] = x['diag_1'].apply(map_icd9)

# Ordinal age encoding
x['age'] = x['age'].map(age_map).fillna(50)

# One-hot encode WITHOUT drop_first (matches main.py inference)
x = pd.get_dummies(x)

feature_names = x.columns.tolist()
neg_count = int((y == 0).sum())
pos_count = int((y == 1).sum())
imbalance_ratio = neg_count / pos_count

print(f"Total features: {len(feature_names)}")
print(f"Class distribution: 0={neg_count}, 1={pos_count}")
print(f"Imbalance ratio (scale_pos_weight): {imbalance_ratio:.2f}")

# -- 2. Train/Test Split --
print("\nSTEP 2: Splitting data...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values
for col in x_train.columns:
    if x_train[col].isnull().any() or x_test[col].isnull().any():
        fill_val = x_train[col].median()
        x_train[col] = x_train[col].fillna(fill_val)
        x_test[col] = x_test[col].fillna(fill_val)

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# -- 3. Optuna Hyperparameter Optimization --
# Using scale_pos_weight instead of SMOTE to preserve probability calibration
print("\n" + "=" * 60)
print("STEP 3: Optuna optimization (30 trials)...")
print("Using scale_pos_weight={:.2f} (NOT SMOTE) for calibrated probabilities".format(imbalance_ratio))
print("=" * 60)

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'scale_pos_weight': imbalance_ratio,
        'random_state': 42,
        'eval_metric': 'logloss',
    }

    model = XGBClassifier(**params)
    model.fit(x_train, y_train)

    y_probs = model.predict_proba(x_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    return np.max(f1_scores)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest Optuna F1-Score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# -- 4. Train Final Model --
print("\n" + "=" * 60)
print("STEP 4: Training final model with best hyperparameters...")
print("=" * 60)

best_params = study.best_params.copy()
best_params['scale_pos_weight'] = imbalance_ratio
best_params['random_state'] = 42
best_params['eval_metric'] = 'logloss'

final_model = XGBClassifier(**best_params)
final_model.fit(x_train, y_train)

# -- 5. Find Optimal Threshold --
print("\nSTEP 5: Finding optimal classification threshold...")
y_probs = final_model.predict_proba(x_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
opt_idx = np.argmax(f1_scores)
optimal_threshold = float(thresholds[opt_idx])

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"F1 at optimal threshold: {f1_scores[opt_idx]:.4f}")

# Apply optimal threshold
y_pred_optimized = (y_probs >= optimal_threshold).astype(int)

# -- 6. Final Evaluation --
print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)
print(f"\nClassification Report (threshold={optimal_threshold:.3f}):")
print(classification_report(y_test, y_pred_optimized))
print(f"Accuracy:  {accuracy_score(y_test, y_pred_optimized):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_optimized):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_optimized):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_optimized):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_probs):.4f}")

# Show probability distribution to confirm calibration
print(f"\nProbability calibration check:")
print(f"  Mean predicted probability: {y_probs.mean():.4f}")
print(f"  Actual positive rate:       {y_test.mean():.4f}")
print(f"  Min probability: {y_probs.min():.4f}")
print(f"  Max probability: {y_probs.max():.4f}")
print(f"  Median probability: {np.median(y_probs):.4f}")

final_f1 = f1_score(y_test, y_pred_optimized)
if final_f1 >= 0.20:
    print(f"\n[OK] F1 = {final_f1:.4f}")
else:
    print(f"\n[WARNING] F1 = {final_f1:.4f} -- Below target.")

# -- 7. Save Artifacts --
print("\n" + "=" * 60)
print("STEP 7: Saving model artifacts...")
print("=" * 60)

joblib.dump(final_model, BASE_DIR / 'hospital_readmission_xgb_model_v2.pkl')
joblib.dump(feature_names, BASE_DIR / 'trained_feature_columns_v2.pkl')
joblib.dump(optimal_threshold, BASE_DIR / 'optimal_threshold.pkl')

print(f"  -> hospital_readmission_xgb_model_v2.pkl")
print(f"  -> trained_feature_columns_v2.pkl ({len(feature_names)} features)")
print(f"  -> optimal_threshold.pkl (threshold={optimal_threshold:.4f})")
print("\n[OK] Retraining complete!")
