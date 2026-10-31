import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# PAGE CONFIG (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# MODEL FEATURE SCHEMA
# Keep these names/order synchronized with Notebook 05.
# -----------------------------
NUMERIC_FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "total_diabetes_drugs",
    "insulin_flag",
    "oral_med_flag",
    "change",
    "diabetesMed",
    "high_medication_count_10plus",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "payer_code",
    "A1Cresult",
    "admission_type",
    "discharge_disposition",
    "admission_source",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "model" /
    "research_final_model_eligible_discharge.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR / "model" /
    "research_threshold_eligible_discharge.pkl"
)


# -----------------------------
# LOAD MODEL AND THRESHOLD
# -----------------------------
@st.cache_resource
def load_model_artifacts():

    """Load the fitted eligible-cohort pipeline and its validation threshold."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    fitted_model = joblib.load(MODEL_PATH)

    decision_threshold = float(
        joblib.load(THRESHOLD_PATH)
    )

    # Verify model feature names.
    trained_features = set(
        getattr(fitted_model, "feature_names_in_", [])
    )

    expected_features = set(MODEL_FEATURES)

    if trained_features and trained_features != expected_features:

        missing = sorted(trained_features - expected_features)
        extra = sorted(expected_features - trained_features)

        raise ValueError(
            f"Model/app feature mismatch. Missing from app: {missing}; "
            f"extra in app: {extra}"
        )

    # Recover exact categorical values learned by the fitted encoder.
    try:

        categorical_encoder = (
            fitted_model
            .named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["encoder"]
        )

    except (AttributeError, KeyError) as exc:

        raise ValueError(
            "The saved model does not have the expected preprocessing structure. "
            "Use the eligible-discharge Pipeline saved by Notebook 05."
        ) from exc

    if len(categorical_encoder.categories_) != len(CATEGORICAL_FEATURES):

        raise ValueError(
            "The model's categorical encoder does not match the app's feature schema."
        )

    # Extract categories for Streamlit dropdowns.
    category_options = {
        feature: [str(value) for value in values]
        for feature, values in zip(
            CATEGORICAL_FEATURES,
            categorical_encoder.categories_
        )
    }

    return fitted_model, decision_threshold, category_options


try:

    model, decision_threshold, category_options = load_model_artifacts()

except Exception as exc:

    st.error(f"Unable to load the research model: {exc}")
    st.stop()


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def categorical_selectbox(label, feature, preferred=None, **kwargs):

    """Create a selectbox using categories learned by the fitted encoder."""

    options = category_options[feature]

    if not options:
        raise ValueError(
            f"No trained categories are available for {feature}."
        )

    index = options.index(preferred) if preferred in options else 0

    return st.selectbox(
        label,
        options,
        index=index,
        **kwargs
    )


def yes_no_selectbox(label, key=None, default=0):

    return st.selectbox(
        label,
        options=[0, 1],
        index=default,
        format_func=lambda value: "Yes" if value == 1 else "No",
        key=key
    )


# -----------------------------
# LIGHT THEME CSS
# -----------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #f0f4f8;
        }

        .header-card {
            background: #ffffff;
            border: 1px solid #bfdbfe;
            padding: 1.4rem 2rem;
            border-radius: 14px;
            text-align: center;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(14, 165, 233, 0.07);
        }

        .header-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #0ea5e9, #6366f1, #8b5cf6);
        }

        .header-card h1 {
            color: #0f172a;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .header-card p {
            color: #64748b;
            font-size: 0.95rem;
            margin: 0.4rem 0 0 0;
            letter-spacing: 0.01em;
        }

        .section-header {
            color: #0284c7;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin: 1.4rem 0 0.75rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid #bae6fd;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .stButton > button {
            background: #0ea5e9;
            color: #ffffff;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            border-radius: 10px;
            transition: background 0.2s ease, transform 0.15s ease;
            width: 100%;
            margin-top: 0.5rem;
            box-shadow: 0 2px 8px rgba(14, 165, 233, 0.25);
        }

        .stButton > button:hover {
            background: #0284c7;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.35);
        }

        .result-high, .result-low {
            border-radius: 14px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .result-high {
            background: #fff5f5;
            border: 1px solid #fecaca;
            box-shadow: 0 1px 6px rgba(239, 68, 68, 0.08);
        }

        .result-high::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #ef4444, #dc2626);
        }

        .result-low {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            box-shadow: 0 1px 6px rgba(16, 185, 129, 0.08);
        }

        .result-low::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #10b981, #059669);
        }

        .status-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .result-high .status-text {
            color: #b91c1c;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .result-low .status-text {
            color: #047857;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .result-high .probability,
        .result-low .probability {
            font-size: 3rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .result-high .probability {
            color: #ef4444;
        }

        .result-low .probability {
            color: #10b981;
        }

        .recommendation {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background-color: transparent;
            border-bottom: 1px solid #bae6fd;
            padding-bottom: 0;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 1.1rem;
            font-size: 0.88rem;
            font-weight: 500;
            color: #94a3b8;
            border: 1px solid transparent;
            border-bottom: none;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #0284c7 !important;
            border-color: #bae6fd !important;
            border-bottom: 1px solid #ffffff !important;
        }

        hr {
            border: none;
            height: 1px;
            background: #bae6fd;
            margin: 1.5rem 0;
        }

        .footer {
            text-align: center;
            color: #94a3b8;
            padding: 2rem;
            margin-top: 2rem;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            border-radius: 8px !important;
            background-color: #ffffff !important;
        }

        .stSpinner > div {
            color: #0284c7 !important;
        }

    </style>
""", unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
    <div class="header-card">
        <h1>🏥 Diabetes Patient Readmission Predictor</h1>
        <p>Eligible-discharge research model · 30-Day Hospital Readmission</p>
    </div>
""", unsafe_allow_html=True)


st.info(
    "Research prototype only. This model was trained on historical U.S. hospital "
    "encounters from 1999–2008. It is not a medical device and must not be used "
    "to make clinical decisions."
)


# -----------------------------
# MAIN CONTENT WITH TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📋 Patient Demographics",
    "🏥 Hospital Visit",
    "💊 Medications & Labs"
])


# ============================================================
# TAB 1: PATIENT DEMOGRAPHICS
# ============================================================
with tab1:

    st.markdown(
        '<div class="section-header">👤 Personal Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        race = categorical_selectbox(
            "Race", "race", preferred="Caucasian"
        )

    with col2:
        gender = categorical_selectbox(
            "Gender", "gender", preferred="Female"
        )

    with col3:
        age = st.slider(
            "Age (years)", 0, 100, 50
        )

    with col4:
        payer_code = categorical_selectbox(
            "Payer Code",
            "payer_code",
            preferred="Unknown",
            help="Insurance/payment category recorded in the source dataset."
        )

    # The training data contains age-band midpoints (5, 15, ..., 95).
    # Convert entered age into the corresponding dataset age-band midpoint.
    age_model_value = min((int(age) // 10) * 10 + 5, 95)

    st.caption(
        f"The model uses the dataset's 10-year age-band midpoint: {age_model_value}."
    )

    st.markdown(
        '<div class="section-header">🏥 Admission Details</div>',
        unsafe_allow_html=True
    )

    col5, col6, col7 = st.columns(3)

    with col5:
        admission_type = categorical_selectbox(
            "Admission Type",
            "admission_type",
            preferred="Emergency"
        )

    with col6:
        discharge_disposition = categorical_selectbox(
            "Discharge Disposition",
            "discharge_disposition",
            preferred="Discharged to home",
            help="The eligible-discharge model's categories exclude death and hospice discharges."
        )

    with col7:
        admission_source = categorical_selectbox(
            "Admission Source",
            "admission_source",
            preferred="Emergency Room"
        )

    st.markdown(
        '<div class="section-header">👨‍⚕️ Medical Information</div>',
        unsafe_allow_html=True
    )

    col8, col9 = st.columns(2)

    with col8:
        medical_specialty = categorical_selectbox(
            "Medical Specialty",
            "medical_specialty",
            preferred="InternalMedicine"
        )

    with col9:
        number_diagnoses = st.number_input(
            "Total Number of Diagnoses",
            min_value=1,
            max_value=16,
            value=5,
            step=1
        )


# ============================================================
# TAB 2: HOSPITAL VISIT
# ============================================================
with tab2:

    st.markdown(
        '<div class="section-header">📊 Visit Statistics</div>',
        unsafe_allow_html=True
    )

    col10, col11, col12, col13 = st.columns(4)

    with col10:
        time_in_hospital = st.number_input(
            "Time in Hospital (days)",
            min_value=1,
            max_value=14,
            value=3,
            step=1
        )

    with col11:
        number_outpatient = st.number_input(
            "Outpatient Visits",
            min_value=0,
            value=0,
            step=1
        )

    with col12:
        number_emergency = st.number_input(
            "Emergency Visits",
            min_value=0,
            value=0,
            step=1
        )

    with col13:
        number_inpatient = st.number_input(
            "Inpatient Visits",
            min_value=0,
            value=0,
            step=1
        )

    st.markdown(
        '<div class="section-header">🔬 Procedures & Labs</div>',
        unsafe_allow_html=True
    )

    col14, col15, col16 = st.columns(3)

    with col14:
        num_lab_procedures = st.number_input(
            "Lab Procedures",
            min_value=0,
            max_value=132,
            value=40,
            step=1
        )

    with col15:
        num_procedures = st.number_input(
            "Procedures",
            min_value=0,
            max_value=6,
            value=1,
            step=1
        )

    with col16:
        num_medications = st.number_input(
            "Number of Medications",
            min_value=1,
            max_value=85,
            value=10,
            step=1
        )

    st.markdown(
        '<div class="section-header">🩺 Diagnoses</div>',
        unsafe_allow_html=True
    )

    col17, col18, col19 = st.columns(3)

    with col17:
        diag_1 = categorical_selectbox(
            "Primary Diagnosis",
            "diag_1",
            preferred="Diabetes"
        )

    with col18:
        diag_2 = categorical_selectbox(
            "Secondary Diagnosis",
            "diag_2",
            preferred="Circulatory"
        )

    with col19:
        diag_3 = categorical_selectbox(
            "Third Diagnosis",
            "diag_3",
            preferred="Circulatory"
        )


# ============================================================
# TAB 3: MEDICATIONS AND LABS
# ============================================================
with tab3:

    st.markdown(
        '<div class="section-header">💊 Diabetes Medications</div>',
        unsafe_allow_html=True
    )

    col20, col21, col22 = st.columns(3)

    with col20:
        total_diabetes_drugs = st.number_input(
            "Total Diabetes Drugs",
            min_value=0,
            max_value=23,
            value=1,
            step=1
        )

    with col21:
        insulin_flag = yes_no_selectbox(
            "Insulin Used",
            key="insulin_flag"
        )

    with col22:
        oral_med_flag = yes_no_selectbox(
            "Oral Medication Used",
            key="oral_med_flag"
        )

    st.markdown(
        '<div class="section-header">📋 Medication Changes & Lab Results</div>',
        unsafe_allow_html=True
    )

    col23, col24, col25 = st.columns(3)

    with col23:
        change = yes_no_selectbox(
            "Medication Changed",
            key="change"
        )

    with col24:
        diabetesMed = yes_no_selectbox(
            "Diabetes Medication Given",
            key="diabetesMed"
        )

    with col25:

        a1c_display = {
            "NotTested": "Not tested",
            "Norm": "Normal",
            ">7": ">7%",
            ">8": ">8%",
        }

        A1Cresult = categorical_selectbox(
            "A1C Result",
            "A1Cresult",
            preferred="NotTested",
            format_func=lambda value: a1c_display.get(value, value)
        )


# ============================================================
# FEATURE ENGINEERING FOR USER INPUT
# ============================================================

# Defined in Notebook 04:
# high_medication_count_10plus = 1 when num_medications >= 10.

high_medication_count_10plus = int(
    num_medications >= 10
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================
input_data = pd.DataFrame([{

    "race": race,
    "gender": gender,
    "age": age_model_value,

    "admission_type": admission_type,
    "discharge_disposition": discharge_disposition,
    "admission_source": admission_source,

    "time_in_hospital": time_in_hospital,
    "payer_code": payer_code,
    "medical_specialty": medical_specialty,

    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,

    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,

    "diag_1": diag_1,
    "diag_2": diag_2,
    "diag_3": diag_3,

    "number_diagnoses": number_diagnoses,

    "A1Cresult": A1Cresult,

    "change": change,
    "diabetesMed": diabetesMed,

    "total_diabetes_drugs": total_diabetes_drugs,
    "insulin_flag": insulin_flag,
    "oral_med_flag": oral_med_flag,

    "high_medication_count_10plus": high_medication_count_10plus,

}])


# ============================================================
# VALIDATE INPUT FEATURE SCHEMA
# ============================================================
missing_inputs = sorted(
    set(MODEL_FEATURES) - set(input_data.columns)
)

extra_inputs = sorted(
    set(input_data.columns) - set(MODEL_FEATURES)
)

if missing_inputs or extra_inputs:

    st.error(
        f"Input schema mismatch. Missing: {missing_inputs}; "
        f"extra: {extra_inputs}"
    )

    st.stop()


# Match exact feature order.
input_data = input_data[MODEL_FEATURES]


# ============================================================
# PREDICTION SECTION
# ============================================================
st.markdown(
    "<hr>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-header">🔮 Prediction</div>',
    unsafe_allow_html=True
)


if st.button(
    "🔮 Predict Readmission Risk",
    use_container_width=True
):

    try:

        with st.spinner("Analyzing patient data..."):

            # Obtain the probability of class 1.
            probability = float(
                model.predict_proba(input_data)[0, 1]
            )

            # Apply the validation-selected threshold.
            prediction = int(
                probability >= decision_threshold
            )

        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])

        with col_result2:

            if prediction == 1:

              result_html = f"""
              <div class="result-high">
              <div class="status-icon">⚠️</div>
              <div class="status-text">Model predicts elevated readmission likelihood</div>
              <div class="probability">{probability:.1%}</div>
              <div class="recommendation">
              Classification threshold: {decision_threshold:.2f}<br>
              This is a model estimate for research demonstration only,
              not a clinical assessment.
              </div>
              </div>
              """

            else:

              result_html = f"""
              <div class="result-low">
              <div class="status-icon">✅</div>
              <div class="status-text">Model predicts lower readmission likelihood</div>
              <div class="probability">{probability:.1%}</div>
              <div class="recommendation">
              Classification threshold: {decision_threshold:.2f}<br>
              A lower model estimate does not rule out readmission.
              </div>
              </div>
              """

            st.markdown(result_html, unsafe_allow_html=True)


    except Exception as exc:

        st.error(f"Prediction failed: {exc}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        Research Demonstration · Logistic Regression · Eligible-Discharge Cohort
    </div>
""", unsafe_allow_html=True)