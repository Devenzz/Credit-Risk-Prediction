import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="AI Credit Risk Engine", layout="wide")

# =================== CUSTOM CSS ===================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Title */
.main-title {
    text-align:center;
    font-size:45px;
    font-weight:800;
    margin-top:20px;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius:15px;
    padding:25px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
}

/* KPI Cards */
.kpi {
    background: linear-gradient(135deg,#667eea,#764ba2);
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color:white;
    font-size:18px;
    font-weight:600;
    padding:12px 25px;
    border-radius:8px;
    border:none;
    width:100%;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🏦 AI Credit Risk Intelligence System</div>', unsafe_allow_html=True)
st.markdown("<center>Advanced Machine Learning • Real-Time Risk Analytics • Random Forest</center>", unsafe_allow_html=True)

st.markdown("---")

# ================= LOAD MODEL =================
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= KPI SECTION =================
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown('<div class="kpi">⚡ Instant Prediction</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="kpi">📊 Risk Probability</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="kpi">🤖 AI Powered</div>', unsafe_allow_html=True)

st.markdown("")

# ================= INPUT SECTION =================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Financial Details")

    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Annual Income", value=50000)
    emp_length = st.slider("Employment Length", 0, 40, 5)
    loan_amount = st.number_input("Loan Amount", value=10000)
    interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 10.0)
    loan_percent_income = st.slider("Loan Percent Income", 0.0, 1.0, 0.2)
    credit_history = st.slider("Credit History Length", 1, 30, 5)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📝 Profile Information")

    home = st.selectbox("Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"])

    intent = st.selectbox("Loan Purpose",
        ["EDUCATION","MEDICAL","PERSONAL","VENTURE",
         "HOMEIMPROVEMENT","DEBTCONSOLIDATION"])

    grade = st.selectbox("Loan Grade",
        ["A","B","C","D","E","F","G"])

    default_history = st.selectbox("Previous Default",
        ["N","Y"])

    st.markdown('</div>', unsafe_allow_html=True)

# ================= ENCODING =================
home_map = {"RENT":0, "OWN":1, "MORTGAGE":2, "OTHER":3}
intent_map = {
    "EDUCATION":0,"MEDICAL":1,"PERSONAL":2,
    "VENTURE":3,"HOMEIMPROVEMENT":4,"DEBTCONSOLIDATION":5
}
grade_map = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6}
default_map = {"N":0,"Y":1}

# ================= PREDICTION =================
if st.button("🚀 Analyze Credit Risk"):

    input_data = np.array([[ 
        age,
        income,
        home_map[home],
        emp_length,
        intent_map[intent],
        grade_map[grade],
        loan_amount,
        interest_rate,
        loan_percent_income,
        default_map[default_history],
        credit_history
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    low_risk = probability[0] * 100
    high_risk = probability[1] * 100

    st.markdown("---")

    # ================= RESULT =================
    if prediction == 1:
        st.error(f"🚨 HIGH RISK CUSTOMER\n\nProbability: {high_risk:.2f}%")
    else:
        st.success(f"✅ LOW RISK CUSTOMER\n\nProbability: {low_risk:.2f}%")

    # ================= DONUT CHART =================
    fig = go.Figure(data=[go.Pie(
        labels=['Low Risk', 'High Risk'],
        values=[low_risk, high_risk],
        hole=.6
    )])

    fig.update_layout(
        title="Risk Distribution",
        showlegend=True,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<center>Developed by Devendra Gangurde 🚀</center>", unsafe_allow_html=True)
