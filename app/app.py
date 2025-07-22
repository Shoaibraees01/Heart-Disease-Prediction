import streamlit as st
import numpy as np
import joblib
from fpdf import FPDF
import base64
import os

# Load models and scaler
log_reg = joblib.load('models/logistic_regression_model.pkl')
rf_clf = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Streamlit settings
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Custom background and styles
page_bg = """
<style>
body {
    background: pink;
    color: #1f1f1f;
}
.stApp {
    background-color: pink;
    color: black;
}
div.stButton > button {
    background-color: #e63946;
    color: white;
    border: none;
    padding: 0.5em 1.5em;
    border-radius: 8px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #d62828;
}
input, select, textarea {
    background-color: #1e1e1e;
    color: white;
}
.custom-button a {
    display: inline-block;
    background-color: #e63946;
    color: white;
    padding: 0.5em 1.5em;
    text-decoration: none;
    border-radius: 8px;
    font-weight: bold;
    transition: 0.3s;
}
.custom-button a:hover {
    background-color: #d62828;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction App")

# Patient details
patient_name = st.text_input("👤 Patient Name (Optional)")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol Level", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["TRUE", "FALSE"])
        restecg = st.selectbox("Resting ECG", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["TRUE", "FALSE"])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
        ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])

    model_choice = st.radio("Choose Model", ("Random Forest", "Logistic Regression"))
    submit = st.form_submit_button("🔮 Predict")

# Predict on submission
if submit:
    # Convert categorical inputs
    sex_val = 1 if sex == "Male" else 0
    cp_val = ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"].index(cp)
    fbs_val = 1 if fbs == "TRUE" else 0
    restecg_val = ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"].index(restecg)
    exang_val = 1 if exang == "TRUE" else 0
    slope_val = ["upsloping", "flat", "downsloping"].index(slope)
    thal_val = ["normal", "fixed defect", "reversible defect"].index(thal)

    input_data = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                            restecg_val, thalach, exang_val, oldpeak, slope_val,
                            ca, thal_val]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    if model_choice == "Random Forest":
        prediction = rf_clf.predict(input_scaled)[0]
        acc = 82.26
    else:
        prediction = log_reg.predict(input_scaled)[0]
        acc = 79.84

    pred_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
    pred_label = "🔴" if prediction == 1 else "🟢"

    st.subheader("🩺 Prediction Result")
    st.success(f"{pred_label} {pred_text}")
    st.info(f"Model Used: {model_choice} | Accuracy: {acc}%")

    # PDF Report Generation
    class PDF(FPDF):
        def header(self):
            self.set_fill_color(255, 228, 240)  # light pink
            self.rect(5.0, 5.0, 200.0, 287.0)   # border
            self.set_fill_color(255, 228, 240)
            self.rect(6.0, 6.0, 198.0, 285.0, 'F')  # fill inside light pink

    pdf = PDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(0, 10, "Heart Disease Prediction Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Section: Patient Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Name: {patient_name or 'Anonymous'}", ln=True)
    pdf.cell(0, 10, f"Sex: {sex}", ln=True)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.ln(5)

    # Section: Health Metrics in Table Format
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Health Metrics", ln=True)
    pdf.set_font("Arial", "", 12)

    pdf.set_fill_color(173, 216, 230)  # Light blue
    pdf.set_draw_color(139, 0, 0)      # Dark red

    metrics = [
        ("Chest Pain Type", cp),
        ("Resting BP (mmHg)", trestbps),
        ("Cholesterol (mg/dl)", chol),
        ("Fasting Blood Sugar > 120", fbs),
        ("Resting ECG", restecg),
        ("Max Heart Rate (bpm)", thalach),
        ("Exercise Induced Angina", exang),
        ("ST Depression", oldpeak),
        ("Slope", slope),
        ("Number of Major Vessels", ca),
        ("Thalassemia", thal),
    ]

    col_width = 70
    for label, value in metrics:
        pdf.cell(col_width, 10, str(label), border=1, fill=True)
        pdf.cell(0, 10, str(value), border=1, fill=True, ln=True)

    pdf.ln(5)

    # Section: Prediction Result
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Prediction Result", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Prediction: {pred_text}", ln=True)
    pdf.cell(0, 10, f"Model Used: {model_choice}", ln=True)
    pdf.cell(0, 10, f"Model Accuracy: {acc}%", ln=True)

    # Save PDF
    pdf_output_path = "heart_report.pdf"
    pdf.output(pdf_output_path)

    # Show download button
    with open(pdf_output_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'''
        <div class="custom-button">
            <a href="data:application/pdf;base64,{base64_pdf}" download="heart_disease_report.pdf">
                📥 Download PDF Report
            </a>
        </div>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)

    # Cleanup
    os.remove(pdf_output_path)
