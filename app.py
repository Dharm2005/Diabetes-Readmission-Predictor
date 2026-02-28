import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
  page_title="Diabetes Readmission Predictor",
  page_icon="🏥",
  layout="centered"
)

st.title("🏥 Diabetes Patient Readmission Prediction")

st.write(
  "Predict whether a patient will be readmitted within 30 days."
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/final_model.pkl")

# -----------------------------
# USER INPUT SECTION
# -----------------------------
st.header("Patient Information")

race = st.selectbox("Race", ["Caucasian","AfricanAmerican","Asian","Hispanic","Asian","Other"])
gender = st.selectbox("Gender", ["Male","Female"])
age = st.slider("Age", 0, 100, 50)

admission_type = st.selectbox(
  "Admission Type",
  ["Emergency","Urgent","Elective","Other"]
)

discharge_disposition = st.selectbox(
  "Discharge Disposition",
  [
    "Discharged to Home",
    "Transferred to SNF",
    "Home with home health service",
    "Transferred to short term hospital",
    "Transferred to rehab facility",
    "Expired",
    "Transferred to inpatient care institution",
    "Transferred to ICF",
    "Left AMA",
    "Transferred to long term care hospital",
    "Hospice home",
    "Hospice medical facility",
    "Transferred to psychiatric hospital",
    "Home under care of Home IV provider",
    "Other"
  ]
)

admission_source = st.selectbox(
  "Admission Source",
  [
    "Emergency Room",
    "Physician Referral",
    "Transfer from hospital",
    "Transfer from health care facility",
    "Clinic Referral",
    "Transfer from SNF",
    "HMO Referral",
    "Other"
  ]
)

medical_specialty = st.selectbox(
  "Medical Specialty",
  [
    'InternalMedicine', 
    'Emergency/Trauma',
    'Family/GeneralPractice',
    'Cardiology', 
    'Surgery-General', 
    'Nephrology',
    'Orthopedics',
    'Orthopedics-Reconstructive', 
    'Radiologist',
    'Pulmonology', 
    'Psychiatry', 
    'Urology', 
    'ObstetricsandGynecology', 
    'Surgery-Cardiovascular/Thoracic', 
    'Other'
  ]
)

diag_1 = st.selectbox(
  "Primary Diagnosis", 
  [
    'Diabetes', 
    'Circulatory', 
    'Respiratory',
    'Digestive', 
    'Injury', 
    'Genitourinary', 
    'Musculoskeletal', 
    'Neoplasms', 
    'Supplementary',
    'External Injury',
    'Other'
  ]
)
diag_2 = st.selectbox(
  "Secondary Diagnosis", 
  [
    'Diabetes', 
    'Neoplasms', 
    'Circulatory', 
    'Respiratory',
    'Injury', 
    'Musculoskeletal', 
    'Genitourinary', 
    'Digestive',
    'Supplementary', 
    'External Injury',
    'Other'
  ]
)
diag_3 = st.selectbox(
  "Third Diagnosis", 
  [
    'Supplementary', 
    'Circulatory', 
    'Diabetes', 
    'Respiratory',
    'Injury', 
    'Neoplasms', 
    'Genitourinary', 
    'Musculoskeletal',
    'Digestive', 
    'External Injury',
    'Other' 
  ]
)

# Numeric inputs
time_in_hospital = st.number_input("Time in Hospital", 1, 15, 3)
num_lab_procedures = st.number_input("Lab Procedures", 1, 130, 40)
num_procedures = st.number_input("Procedures", 0, 10, 1)
num_medications = st.number_input("Number of Medications", 1, 85, 10)
number_outpatient = st.number_input("Outpatient Visits", 0, 10, 0)
number_emergency = st.number_input("Emergency Visits", 0, 10, 0)
number_inpatient = st.number_input("Inpatient Visits", 0, 20, 0)
number_diagnoses = st.number_input("Number of Diagnoses", 1, 16, 5)

total_diabetes_drugs = st.number_input("Total Diabetes Drugs", 0, 10, 1)
insulin_flag = st.selectbox("Insulin Used", [0,1])
oral_med_flag = st.selectbox("Oral Medication", [0,1])
is_polypharmacy = st.selectbox("Polypharmacy", [0,1])

change = st.selectbox("Medication Changed", [0,1])
diabetesMed = st.selectbox("Diabetes Medication Given", [0,1])

A1Cresult = st.selectbox("A1C Result", [0,1,2,3])

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
# PREDICTION
# -----------------------------
if st.button("Predict Readmission"):

  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1]

  if prediction == 1:
    st.error(f"⚠ High Risk of Readmission ({probability:.2%})")
  else:
    st.success(f"✅ Low Risk of Readmission ({probability:.2%})")