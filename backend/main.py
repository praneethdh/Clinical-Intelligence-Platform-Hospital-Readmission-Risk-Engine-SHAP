from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Hospital Readmission AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Artifacts on Startup
try:
    model = joblib.load(BASE_DIR / 'hospital_readmission_xgb_model_v2.pkl')
    expected_features = joblib.load(BASE_DIR / 'trained_feature_columns_v2.pkl')
    optimal_threshold = joblib.load(BASE_DIR / 'optimal_threshold.pkl')
    print(f"Model loaded. Features: {len(expected_features)}, Threshold: {optimal_threshold:.4f}")
except FileNotFoundError as e:
    print(f"WARNING: Model artifacts missing: {e}")
    model = None
    expected_features = None
    optimal_threshold = 0.5

# Try loading SHAP explainer (tree-based, instant for XGBoost)
shap_explainer = None
try:
    import shap
    if model is not None:
        shap_explainer = shap.TreeExplainer(model)
        print("SHAP Explainer loaded successfully.")
except Exception as e:
    print(f"SHAP not available: {e}")

# 2. Input Schema
class PatientData(BaseModel):
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_inpatient: int
    number_outpatient: int = 0
    number_emergency: int = 0
    number_diagnoses: int = 5
    discharge_disposition_id: int
    admission_type_id: int
    admission_source_id: int = 7
    gender: str
    race: str = "Caucasian"
    age: str
    diabetesMed_Yes: int
    insulin: str = "No"
    change: str = "No"
    diag_1: str
    payer_code: str = "MC"

# Feature engineering helpers
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

FRIENDLY_NAMES = {
    'number_inpatient': 'Prior Inpatient Visits',
    'number_emergency': 'Emergency Room Visits',
    'number_outpatient': 'Outpatient Visits',
    'num_medications': 'Number of Medications',
    'time_in_hospital': 'Days in Hospital',
    'num_lab_procedures': 'Lab Procedures Count',
    'num_procedures': 'Medical Procedures',
    'number_diagnoses': 'Number of Diagnoses',
    'discharge_disposition_id': 'Discharge Destination',
    'admission_type_id': 'Admission Type',
    'admission_source_id': 'Admission Source',
    'age': 'Patient Age',
    'diabetesMed_Yes': 'On Diabetes Medication',
}

age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

# 3. Prediction Endpoint with SHAP Reasoning
@app.post("/predict")
def predict_risk(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")
    try:
        raw_data = patient.model_dump()

        # Rename diabetesMed_Yes to diabetesMed for proper one-hot encoding
        raw_data['diabetesMed'] = 'Yes' if raw_data.pop('diabetesMed_Yes') else 'No'

        single_df = pd.DataFrame([raw_data])

        # Feature engineering (matches retrain_model.py exactly)
        single_df['diag_1'] = single_df['diag_1'].apply(map_icd9)
        single_df['age'] = single_df['age'].map(age_map).fillna(50)

        # One-hot encode WITHOUT drop_first (matches training)
        single_encoded = pd.get_dummies(single_df)
        final_features = single_encoded.reindex(columns=expected_features, fill_value=0)

        # Predict using saved optimal threshold
        risk_probability = float(model.predict_proba(final_features)[0][1])
        is_high_risk = bool(risk_probability > optimal_threshold)

        # SHAP Reasoning
        reasoning = []
        if shap_explainer is not None:
            try:
                shap_values = shap_explainer.shap_values(final_features)
                if isinstance(shap_values, list):
                    sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    sv = shap_values[0]
                feature_names = final_features.columns.tolist()
                feature_vals = final_features.iloc[0].values

                # Pair each feature with its SHAP value and actual value
                pairs = list(zip(feature_names, sv, feature_vals))
                # Sort by absolute SHAP impact (descending)
                pairs.sort(key=lambda x: abs(float(x[1])), reverse=True)

                for feat_name, shap_val, feat_val in pairs[:6]:
                    shap_val = float(shap_val)
                    feat_val = float(feat_val) if not pd.isna(feat_val) else 0.0
                    direction = "increases" if shap_val > 0 else "decreases"
                    impact = abs(shap_val)

                    # Use friendly names
                    display_name = FRIENDLY_NAMES.get(feat_name, feat_name.replace('_', ' ').title())

                    if impact < 0.005:
                        continue

                    reasoning.append({
                        "feature": display_name,
                        "direction": direction,
                        "impact": round(impact, 3),
                        "value": round(feat_val, 2),
                        "explanation": f"{display_name} = {round(feat_val, 1)} {direction} readmission risk"
                    })
            except Exception as e:
                reasoning = [{"feature": "SHAP Error", "direction": "unknown", "impact": 0, "value": 0, "explanation": str(e)}]

        # Generate Text Summary
        text_summary = ""
        if reasoning:
            valid_reasons = [r for r in reasoning if r.get("feature") != "SHAP Error"]
            if valid_reasons:
                pos_factors = [f"**{r['feature']}** (value: {r['value']})" for r in valid_reasons if r['direction'] == 'increases']
                neg_factors = [f"**{r['feature']}** (value: {r['value']})" for r in valid_reasons if r['direction'] == 'decreases']
                
                summary_parts = []
                if pos_factors:
                    summary_parts.append("primarily driven by " + ", ".join(pos_factors[:3]))
                if neg_factors:
                    summary_parts.append("mitigated by " + ", ".join(neg_factors[:3]))
                
                if summary_parts:
                    text_summary = "The patient's readmission risk is " + " and ".join(summary_parts) + "."
                else:
                    text_summary = "No single clinical factor shows a dominant impact on this patient's prediction."
            else:
                text_summary = f"Unable to generate explanation: {reasoning[0]['explanation']}"
        else:
            text_summary = "Detailed SHAP explanation features are not available."

        return {
            "high_risk_alert": is_high_risk,
            "probability_score": float(round(risk_probability * 100, 1)),
            "recommendation": "Recommend delayed discharge & home intervention." if is_high_risk else "Standard discharge approved.",
            "reasoning": reasoning,
            "reasoning_summary": text_summary,
            "risk_level": "Critical" if risk_probability > 0.5 else "High" if risk_probability > 0.25 else "Elevated" if risk_probability > optimal_threshold else "Low",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "shap_available": shap_explainer is not None,
        "optimal_threshold": optimal_threshold,
    }
