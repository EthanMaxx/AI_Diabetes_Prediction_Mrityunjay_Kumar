import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

LOG_FILE = "app_logs.log"
PREDICTION_LOG_DIR = "prediction_logs"
MODEL_PATH = "diabetes_model.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

# Risk thresholds for probability
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

# Physiological bounds for validation
PHYSIOLOGICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Pregnancies": (0, 25),                # max pregnancies unlikely to exceed 25
    "Glucose": (40, 500),                  # wide range to capture extreme cases
    "BloodPressure": (40, 250),            # diastolic; wide range
    "SkinThickness": (0, 100),             # mm
    "Insulin": (0, 1000),                  # mu U/ml
    "BMI": (10.0, 80.0),                   # kg/m²
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (0, 120),
}

# -------------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------


@st.cache_resource
def load_model() -> Tuple[Optional[Any], Optional[List[str]]]:
    """
    Load the trained model and the ordered list of feature names.

    Returns
    -------
    model : sklearn-compatible estimator or None
    feature_names : list of str or None
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(FEATURE_NAMES_PATH, "rb") as f:
            feature_names = pickle.load(f)

        if not isinstance(feature_names, (list, tuple)):
            raise ValueError("feature_names.pkl does not contain a list of feature names.")

        logger.info("Model and feature names loaded successfully.")
        return model, list(feature_names)

    except Exception as exc:
        logger.error(f"Error loading model or feature names: {exc}")
        st.error("Failed to load the prediction model. Please contact support.")
        return None, None


# -------------------------------------------------------------------
# INPUT VALIDATION & FEATURE ENGINEERING
# -------------------------------------------------------------------


def validate_input(feature: str, value: float) -> Tuple[bool, str]:
    """
    Validate a single input value against physiological bounds.

    Parameters
    ----------
    feature : str
        Feature name as used in PHYSIOLOGICAL_BOUNDS.
    value : float
        Numeric value provided by the user.

    Returns
    -------
    (is_valid, error_message) : (bool, str)
    """
    bounds = PHYSIOLOGICAL_BOUNDS.get(feature)
    if bounds is None:
        # If not configured, treat as valid (no constraints)
        return True, ""

    min_val, max_val = bounds
    if value < min_val or value > max_val:
        return False, f"{feature} should be between {min_val} and {max_val}."

    return True, ""


def engineer_features(input_data: Dict[str, float]) -> Dict[str, float]:
    """
    Engineer additional features from raw inputs.

    Adds:
    - BMI category one-hot features
    - Glucose category one-hot features
    - Interaction terms: Glucose*BMI, Age*BMI
    """
    try:
        engineered = dict(input_data)  # work on a copy

        # ---------------------------
        # BMI categories
        # ---------------------------
        bmi = engineered["BMI"]

        engineered["BMI_Category_Normal"] = 0
        engineered["BMI_Category_Overweight"] = 0
        engineered["BMI_Category_Obese"] = 0

        if bmi < 18.5:
            # Underweight – not explicitly modelled, all zeros
            pass
        elif bmi < 25:
            engineered["BMI_Category_Normal"] = 1
        elif bmi < 30:
            engineered["BMI_Category_Overweight"] = 1
        else:
            engineered["BMI_Category_Obese"] = 1

        # ---------------------------
        # Glucose categories
        # ---------------------------
        glucose = engineered["Glucose"]

        engineered["Glucose_Category_Normal"] = 0
        engineered["Glucose_Category_Prediabetes"] = 0
        engineered["Glucose_Category_Diabetes"] = 0

        if glucose < 70:
            # Hypoglycaemia – not explicitly modelled, all zeros
            pass
        elif glucose < 100:
            engineered["Glucose_Category_Normal"] = 1
        elif glucose < 126:
            engineered["Glucose_Category_Prediabetes"] = 1
        else:
            engineered["Glucose_Category_Diabetes"] = 1

        # ---------------------------
        # Interaction features
        # ---------------------------
        engineered["Glucose_BMI"] = engineered["Glucose"] * engineered["BMI"]
        engineered["Age_BMI"] = engineered["Age"] * engineered["BMI"]

        return engineered

    except KeyError as exc:
        logger.error(f"Missing key in engineer_features: {exc}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in engineer_features: {exc}")
        raise


# -------------------------------------------------------------------
# PREDICTION + LOGGING
# -------------------------------------------------------------------


def predict_diabetes(
    input_data: Dict[str, float],
    model: Any,
    feature_names: List[str],
) -> Tuple[Optional[int], Optional[float]]:
    """
    Run the model on a single patient's data.

    Returns
    -------
    prediction : int or None
        1 for high risk, 0 for low risk.
    probability : float or None
        Predicted probability of diabetes (class 1).
    """
    try:
        engineered = engineer_features(input_data)

        # Build DataFrame with engineered features
        input_df = pd.DataFrame([engineered])

        # Ensure all required features exist; fill missing with 0
        for feat in feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0

        # Use exact model feature ordering
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        return int(prediction[0]), float(proba)

    except Exception as exc:
        logger.error(f"Error in prediction: {exc}")
        st.error(f"An error occurred during prediction: {exc}")
        return None, None


def log_prediction(input_data: Dict[str, float], prediction: int, probability: float) -> None:
    """
    Persist the input features and prediction for monitoring.

    Logs to: prediction_logs/predictions_YYYYMMDD.csv
    """
    try:
        os.makedirs(PREDICTION_LOG_DIR, exist_ok=True)
        fname = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        log_file = os.path.join(PREDICTION_LOG_DIR, fname)

        row = dict(input_data)
        row["prediction"] = prediction
        row["probability"] = probability
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df = pd.DataFrame([row])

        if os.path.exists(log_file):
            df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            df.to_csv(log_file, index=False)

        logger.info(f"Logged prediction={prediction}, probability={probability:.4f}")

    except Exception as exc:
        logger.error(f"Error logging prediction: {exc}")


# -------------------------------------------------------------------
# UI HELPERS
# -------------------------------------------------------------------


def display_risk_block(prediction: int, probability: float) -> None:
    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

    with col2:
        st.metric("Predicted Probability", f"{probability:.1%}")


def display_risk_interpretation(probability: float) -> None:
    st.subheader("Risk Interpretation")

    if probability < LOW_RISK_THRESHOLD:
        st.write("**Low Risk:** The model predicts a low probability of diabetes.")
        st.write("Recommendation: Maintain a healthy lifestyle with regular exercise and a balanced diet.")
    elif probability < HIGH_RISK_THRESHOLD:
        st.write("**Moderate Risk:** There are some indicators suggesting elevated risk of diabetes.")
        st.write("Recommendation: Consider consulting with a healthcare provider for further evaluation.")
    else:
        st.write("**High Risk:** The model predicts a high probability of diabetes.")
        st.write("Recommendation: Please consult with a healthcare provider as soon as possible for diagnosis and care.")


def display_risk_factors(glucose: float, bmi: float, dpf: float, age: int) -> None:
    st.subheader("Key Risk Factors")

    factors = []

    if glucose > 125:
        factors.append(
            (
                "High Glucose Level",
                f"{glucose} mg/dL",
                "Elevated blood glucose is a primary indicator of diabetes.",
            )
        )

    if bmi > 30:
        factors.append(
            ("Obesity", f"BMI: {bmi:.1f}", "Obesity is strongly associated with type 2 diabetes.")
        )

    if dpf > 0.8:
        factors.append(
            (
                "Family History",
                f"Diabetes Pedigree Function: {dpf:.3f}",
                "A high diabetes pedigree function indicates genetic predisposition.",
            )
        )

    if age > 45:
        factors.append(
            (
                "Age",
                f"{age} years",
                "Risk of type 2 diabetes increases with age.",
            )
        )

    if factors:
        for title, value, desc in factors:
            st.write(f"**{title}:** {value}")
            st.write(desc)
    else:
        st.write("No specific high-risk factors identified based on the entered values.")


def display_disclaimer() -> None:
    st.info(
        """
**Disclaimer:** This prediction is based on a machine learning model and is intended
for educational purposes only. It is **not** a substitute for professional medical
advice, diagnosis, or treatment. Please consult a qualified healthcare professional
for any health concerns.
"""
    )


# -------------------------------------------------------------------
# MAIN STREAMLIT APP
# -------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Diabetes Risk Prediction System",
        page_icon="🩺",
        layout="wide",
    )

    st.title("Diabetes Risk Prediction System")
    st.write(
        """
### Supply Chain Approach to Healthcare

This system uses a machine learning model to estimate diabetes risk based on clinical
and demographic features. Please enter the information in the sidebar for an assessment.
"""
    )

    # Load model
    model, feature_names = load_model()
    if model is None or feature_names is None:
        st.stop()

    # Sidebar input form
    st.sidebar.header("Patient Information")
    error_container = st.empty()

    try:
        with st.sidebar.form("patient_data_form"):
            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=25, value=0)
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=500, value=120)
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=250, value=80)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

            with col2:
                insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=79)
                bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, format="%.1f")
                dpf = st.number_input(
                    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f"
                )
                age = st.number_input("Age", min_value=0, max_value=120, value=33)

            submit_button = st.form_submit_button("Predict Diabetes Risk")

        if submit_button:
            # Collect raw inputs
            raw_input = {
                "Pregnancies": pregnancies,
                "Glucose": float(glucose),
                "BloodPressure": float(blood_pressure),
                "SkinThickness": float(skin_thickness),
                "Insulin": float(insulin),
                "BMI": float(bmi),
                "DiabetesPedigreeFunction": float(dpf),
                "Age": int(age),
            }

            # Validate
            all_valid = True
            messages = []

            for feat, val in raw_input.items():
                valid, msg = validate_input(feat, float(val))
                if not valid:
                    all_valid = False
                    messages.append(msg)

            if not all_valid:
                error_container.error("\n".join(messages))
            else:
                error_container.empty()

                prediction, probability = predict_diabetes(raw_input, model, feature_names)

                if prediction is not None and probability is not None:
                    # Log
                    log_prediction(raw_input, prediction, probability)

                    # Show results
                    st.subheader("Prediction Results")
                    display_risk_block(prediction, probability)
                    display_risk_interpretation(probability)
                    display_risk_factors(glucose, bmi, dpf, age)
                    display_disclaimer()

        # Extra information sections
        with st.expander("About this System"):
            st.write(
                """
This diabetes prediction system uses a machine learning model trained on the
Pima Indians Diabetes Dataset. It considers several factors associated with
diabetes risk:

- Pregnancies: Number of times pregnant  
- Glucose: Plasma glucose concentration (mg/dL)  
- Blood Pressure: Diastolic blood pressure (mm Hg)  
- Skin Thickness: Triceps skin fold thickness (mm)  
- Insulin: 2-hour serum insulin (mu U/ml)  
- BMI: Body mass index (weight in kg/(height in m)²)  
- Diabetes Pedigree Function: Represents genetic influence  
- Age: Age in years  

The system applies a supply chain perspective to data handling: quality checks,
structured processing, and monitoring over time via prediction logs.
"""
            )

        with st.expander("Understanding Your Results"):
            st.write(
                """
### How to Interpret Your Results

The probability score is derived from statistical patterns in historical data.

- **Below 30%** – Generally considered low risk  
- **30% to 70%** – Moderate risk that warrants attention  
- **Above 70%** – High risk that should be discussed with a healthcare provider  

### General Recommendations (for prevention and risk reduction)

1. Maintain a healthy diet rich in fruits, vegetables, and whole grains  
2. Engage in regular physical activity (≥150 minutes of moderate exercise/week)  
3. Maintain a healthy weight  
4. Schedule regular health check-ups  
5. Monitor blood glucose if you have risk factors  

This tool is an educational aid, not a diagnostic device.
"""
            )

    except Exception as exc:
        logger.error(f"Unexpected error in Streamlit app: {exc}")
        st.error(f"An unexpected error occurred: {exc}")
        st.write("Please try again later or contact support if the problem persists.")


if __name__ == "__main__":
    main()
