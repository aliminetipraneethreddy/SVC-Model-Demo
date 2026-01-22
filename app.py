import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Ministry of Finance | Govt of India",
    page_icon="🇮🇳",
    layout="wide"
)

# --------------------------------------------------
# EXECUTIVE PATRIOTIC CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* Patriotic Gradient Background */
    .stApp {
        background: linear-gradient(180deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        background-attachment: fixed;
    }

    /* Executive Header Bar */
    .exec-header {
        background-color: white;
        padding: 15px 30px;
        border-bottom: 4px solid #000080;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    .govt-logos {
        display: flex;
        align-items: center;
        gap: 20px;
    }

    .ministry-text {
        color: #000080;
        font-family: 'serif';
        line-height: 1.2;
    }

    .hindi-title {
        font-size: 24px;
        font-weight: bold;
    }

    .english-title {
        font-size: 18px;
        font-weight: 500;
        letter-spacing: 1px;
    }

    /* Minister Profile Section */
    .minister-card {
        text-align: center;
        border: 2px solid #FF9933;
        padding: 5px;
        background: white;
        border-radius: 8px;
    }

    .minister-img {
        border-radius: 5px;
        width: 120px;
        height: 140px;
        object-fit: cover;
    }

    .minister-name {
        font-size: 14px;
        font-weight: bold;
        color: #000080;
        margin-top: 5px;
    }

    .minister-desc {
        font-size: 10px;
        color: #444;
    }

    /* Content Card */
    .main-card {
        background-color: rgba(255, 255, 255, 0.97);
        padding: 40px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 900px;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
        border-left: 10px solid #000080;
    }

    /* Custom Button */
    .stButton>button {
        background: #000080 !important;
        color: white !important;
        border-radius: 0px !important;
        width: 100%;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# EXECUTIVE TOP NAVIGATION (BILINGUAL)
# --------------------------------------------------
col_title, col_minister = st.columns([3, 1])

with col_title:
    st.markdown("""
        <div class="govt-logos">
            <img src="https://www.india.gov.in/sites/upload_files/npi/files/logo_0.png" width="80">
            <div class="ministry-text">
                <div class="hindi-title">वित्त मंत्रालय</div>
                <div class="english-title">MINISTRY OF FINANCE</div>
                <div style="color: #138808; font-weight: bold; font-size: 14px;">भारत सरकार | Government of India</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_minister:
    # Displaying the image and credentials as requested
    st.markdown(f"""
        <div class="minister-card">
            <img src="https://ibb.co/jZvF0dnt" class="minister-img">
            <div class="minister-name">Shri A. Praneeth Reddy</div>
            <div class="minister-desc">Hon'ble Prime Minister &<br>Finance Minister of India</div>
        </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# ML ENGINE (LOAD DATA)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approved (1).csv")
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Property_Area'].fillna(df['Property_Area'].mode()[0], inplace=True)

    encoders = {}
    for col in ['Self_Employed', 'Property_Area', 'Loan_Status (Approved)']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed', 'Property_Area']]
    y = df['Loan_Status (Approved)']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled, y, scaler, encoders

df, X_scaled, y, scaler, encoders = load_data()

# --------------------------------------------------
# APPLICATION FORM
# --------------------------------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#000080;'>Smart Loan Verification System</h2>", unsafe_allow_html=True)
st.write("---")

c1, c2 = st.columns(2)
with c1:
    income = st.number_input("Monthly Applicant Income (₹)", value=60000)
    loan = st.number_input("Requested Loan Amount (₹ in Thousands)", value=200)
    property_area = st.selectbox("Property Type", encoders['Property_Area'].classes_)

with c2:
    employment = st.selectbox("Employment Status", ["No", "Yes"])
    credit = st.selectbox("Credit History Clean?", ["Yes", "No"])
    kernel = st.selectbox("AI Model Kernel", ["rbf", "linear", "poly"])

# Model Training
model = SVC(kernel=kernel, probability=True)
model.fit(X_scaled, y)

if st.button("VERIFY LOAN STATUS"):
    # Encode and predict
    credit_val = 1 if credit == "Yes" else 0
    emp_val = encoders['Self_Employed'].transform([employment])[0]
    area_val = encoders['Property_Area'].transform([property_area])[0]
    
    input_data = scaler.transform([[income, loan, credit_val, emp_val, area_val]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("✅ ELIGIBILITY VERIFIED: THE LOAN IS APPROVED UNDER GOVT SCHEMES.")
    else:
        st.error("❌ ELIGIBILITY FAILED: THE APPLICANT DOES NOT MEET FINANCIAL CRITERIA.")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #fff; text-shadow: 1px 1px 2px #000;">
        <p>National Portal of India | आत्मनिर्भर भारत</p>
        <p style="font-size: 12px;">© 2026 Ministry of Finance. All Rights Reserved.</p>
    </div>
""", unsafe_allow_html=True)