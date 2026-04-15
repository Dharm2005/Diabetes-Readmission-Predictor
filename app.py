import streamlit as st
import pandas as pd
import joblib

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
# LIGHT THEME CSS
# -----------------------------
st.markdown("""
    <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Page background ── */
        .stApp {
            background-color: #f0f4f8;
        }

        /* ── Header Card ── */
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
            top: 0; left: 0; right: 0;
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

        /* ── Section Headers ── */
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

        /* ── Predict Button ── */
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

        .stButton > button:active {
            transform: translateY(0px);
        }

        /* ── Result Card: High Risk ── */
        .result-high {
            background: #fff5f5;
            border: 1px solid #fecaca;
            border-radius: 14px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 6px rgba(239, 68, 68, 0.08);
        }

        .result-high::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #ef4444, #dc2626);
        }

        .result-high .status-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .result-high .status-text { color: #b91c1c; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.4rem; letter-spacing: 0.01em; }
        .result-high .probability { color: #ef4444; font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; }
        .result-high .recommendation {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #fee2e2;
        }

        /* ── Result Card: Low Risk ── */
        .result-low {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 14px;
            padding: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 6px rgba(16, 185, 129, 0.08);
        }

        .result-low::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #10b981, #059669);
        }

        .result-low .status-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .result-low .status-text { color: #047857; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.4rem; letter-spacing: 0.01em; }
        .result-low .probability { color: #10b981; font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; }
        .result-low .recommendation {
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #d1fae5;
        }

        /* ── Tabs ── */
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

        /* ── Divider ── */
        hr {
            border: none;
            height: 1px;
            background: #bae6fd;
            margin: 1.5rem 0;
        }

        /* ── Footer ── */
        .footer {
            text-align: center;
            color: #94a3b8;
            padding: 2rem;
            margin-top: 2rem;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        /* ── Widget surfaces ── */
        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            border-radius: 8px !important;
            background-color: #ffffff !important;
        }

        /* ── Spinner text ── */
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
        <p>Advanced ML Model · 30-Day Hospital Readmission Risk Assessment</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/final_model.pkl")

# -----------------------------
# MAIN CONTENT WITH TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📋 Patient Demographics", "🏥 Hospital Visit", "💊 Medications & Labs"])

with tab1:
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        age = st.slider("Age", 0, 100, 50)

    st.markdown('<div class="section-header">🏥 Admission Details</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        admission_type = st.selectbox(
            "Admission Type",
            ["Emergency", "Urgent", "Elective", "Other"]
        )
    with col5:
        discharge_disposition = st.selectbox(
            "Discharge Disposition",
            [
                "Discharged to Home", "Transferred to SNF",
                "Home with home health service", "Transferred to short term hospital",
                "Transferred to rehab facility", "Expired",
                "Transferred to inpatient care institution", "Transferred to ICF",
                "Left AMA", "Transferred to long term care hospital",
                "Hospice home", "Hospice medical facility",
                "Transferred to psychiatric hospital",
                "Home under care of Home IV provider", "Other"
            ]
        )
    with col6:
        admission_source = st.selectbox(
            "Admission Source",
            [
                "Emergency Room", "Physician Referral", "Transfer from hospital",
                "Transfer from health care facility", "Clinic Referral",
                "Transfer from SNF", "HMO Referral", "Other"
            ]
        )

    st.markdown('<div class="section-header">👨‍⚕️ Medical Information</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        medical_specialty = st.selectbox(
            "Medical Specialty",
            [
                'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
                'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
                'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology',
                'Psychiatry', 'Urology', 'ObstetricsandGynecology',
                'Surgery-Cardiovascular/Thoracic', 'Other'
            ]
        )
    with col8:
        number_diagnoses = st.number_input("Total Number of Diagnoses", 1, 16, 5)

with tab2:
    st.markdown('<div class="section-header">📊 Visit Statistics</div>', unsafe_allow_html=True)
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        time_in_hospital = st.number_input("Time in Hospital (days)", 1, 15, 3)
    with col10:
        number_outpatient = st.number_input("Outpatient Visits", 0, 10, 0)
    with col11:
        number_emergency = st.number_input("Emergency Visits", 0, 10, 0)
    with col12:
        number_inpatient = st.number_input("Inpatient Visits", 0, 20, 0)

    st.markdown('<div class="section-header">🔬 Procedures & Labs</div>', unsafe_allow_html=True)
    col13, col14, col15 = st.columns(3)
    with col13:
        num_lab_procedures = st.number_input("Lab Procedures", 1, 130, 40)
    with col14:
        num_procedures = st.number_input("Procedures", 0, 10, 1)
    with col15:
        num_medications = st.number_input("Number of Medications", 1, 85, 10)

    st.markdown('<div class="section-header">🩺 Diagnoses</div>', unsafe_allow_html=True)
    col16, col17, col18 = st.columns(3)
    with col16:
        diag_1 = st.selectbox(
            "Primary Diagnosis",
            ['Diabetes', 'Circulatory', 'Respiratory', 'Digestive', 'Injury',
             'Genitourinary', 'Musculoskeletal', 'Neoplasms', 'Supplementary',
             'External Injury', 'Other']
        )
    with col17:
        diag_2 = st.selectbox(
            "Secondary Diagnosis",
            ['Diabetes', 'Neoplasms', 'Circulatory', 'Respiratory', 'Injury',
             'Musculoskeletal', 'Genitourinary', 'Digestive', 'Supplementary',
             'External Injury', 'Other']
        )
    with col18:
        diag_3 = st.selectbox(
            "Third Diagnosis",
            ['Supplementary', 'Circulatory', 'Diabetes', 'Respiratory', 'Injury',
             'Neoplasms', 'Genitourinary', 'Musculoskeletal', 'Digestive',
             'External Injury', 'Other']
        )

with tab3:
    st.markdown('<div class="section-header">💊 Diabetes Medications</div>', unsafe_allow_html=True)
    col19, col20, col21, col22 = st.columns(4)
    with col19:
        total_diabetes_drugs = st.number_input("Total Diabetes Drugs", 0, 10, 1)
    with col20:
        insulin_flag = st.selectbox("Insulin Used", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col21:
        oral_med_flag = st.selectbox("Oral Medication", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col22:
        is_polypharmacy = st.selectbox("Polypharmacy", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown('<div class="section-header">📋 Medication Changes</div>', unsafe_allow_html=True)
    col23, col24, col25 = st.columns(3)
    with col23:
        change = st.selectbox("Medication Changed", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col24:
        diabetesMed = st.selectbox("Diabetes Medication Given", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col25:
        A1Cresult = st.selectbox(
            "A1C Result",
            [0, 1, 2, 3],
            format_func=lambda x: ["Normal", "Abnormal", "High", "Very High"][x]
        )

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame([{
    "race": race,
    "gender": gender,
    "age": age,
    "admission_type": admission_type,
    "discharge_disposition": discharge_disposition,
    "admission_source": admission_source,
    "medical_specialty": medical_specialty,
    "diag_1": diag_1,
    "diag_2": diag_2,
    "diag_3": diag_3,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "total_diabetes_drugs": total_diabetes_drugs,
    "insulin_flag": insulin_flag,
    "oral_med_flag": oral_med_flag,
    "is_polypharmacy": is_polypharmacy,
    "change": change,
    "diabetesMed": diabetesMed,
    "A1Cresult": A1Cresult
}])

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🔮 Prediction</div>', unsafe_allow_html=True)

if st.button("🔮 Predict Readmission Risk", use_container_width=True):
    with st.spinner("Analyzing patient data..."):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

    col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
    with col_result2:
        if prediction == 1:
            st.markdown(f"""
                <div class="result-high">
                    <div class="status-icon">⚠️</div>
                    <div class="status-text">High Risk of Readmission</div>
                    <div class="probability">{probability:.1%}</div>
                    <div class="recommendation">
                        <strong>Recommendation:</strong> Close monitoring and immediate follow-up care advised.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-low">
                    <div class="status-icon">✅</div>
                    <div class="status-text">Low Risk of Readmission</div>
                    <div class="probability">{probability:.1%}</div>
                    <div class="recommendation">
                        <strong>Recommendation:</strong> Standard follow-up care is sufficient.
                    </div>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
    <div class="footer">
        Predictive Modeling · Powered by Machine Learning
    </div>
""", unsafe_allow_html=True)