import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Credit Risk Intelligence",
    page_icon="💳",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("credit_risk_dataset.csv")

df = load_data()

num_features = df.select_dtypes(include=np.number).columns.tolist()
cat_features = df.select_dtypes(include="object").columns.tolist()

# ================= SESSION STATE =================
if "show_overview" not in st.session_state:
    st.session_state.show_overview = False
if "prediction_mode" not in st.session_state:
    st.session_state.prediction_mode = False


# ================= THEME =================
st.markdown("""
<style>
.main { background-color: #0f172a; color:white; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#111827,#1f2937);
}

.kpi-card {
    background: linear-gradient(135deg,#1e293b,#334155);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    color: white;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.4);
}

.kpi-title { font-size:14px; opacity:0.8; }
.kpi-value { font-size:26px; font-weight:bold; }

.insight-box {
    background: #1e293b;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #38bdf8;
    margin-top: 15px;
}

.highlight-box {
    background: #111827;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #10b981;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
# ================= SIDEBAR =================
st.sidebar.title("📊 Control Panel")

if st.sidebar.button("📂 Dataset Overview"):
    st.session_state.show_overview = True
    st.session_state.prediction_mode = False

if st.sidebar.button("📈 Analysis Dashboard"):
    st.session_state.show_overview = False
    st.session_state.prediction_mode = False

if st.sidebar.button("🔮 Predict Loan Risk"):
    st.session_state.show_overview = False
    st.session_state.prediction_mode = True

# 👇 Moved BELOW Predict Button (not removed)
analysis_type = st.sidebar.selectbox("Analysis Type", ["Univariate", "Bivariate"])


# ================= TITLE =================
st.title("💳 AI Credit Risk Intelligence Dashboard")
st.markdown("---")

# ============================================================
# ================= DATASET OVERVIEW ==========================
# ============================================================
if st.session_state.show_overview:

    st.header("📂 Dataset Overview")

    rows, cols = df.shape

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Rows</div><div class='kpi-value'>{rows}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Columns</div><div class='kpi-value'>{cols}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Numerical Columns</div><div class='kpi-value'>{len(num_features)}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Categorical Columns</div><div class='kpi-value'>{len(cat_features)}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("🔍 View Sample Dataset (30 Rows)"):
        st.dataframe(df.head(30), use_container_width=True, height=450)

    st.markdown("---")

    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # ===== Column Description Added Here =====
    st.subheader("📑 Column Description")

    column_desc = pd.DataFrame({
        "Feature Name": [
            "person_age","person_income","person_home_ownership",
            "person_emp_length","loan_intent","loan_grade",
            "loan_amnt","loan_int_rate","loan_status",
            "loan_percent_income","cb_person_default_on_file",
            "cb_person_cred_hist_length"
        ],
        "Description": [
            "Age",
            "Annual Income",
            "Home ownership",
            "Employment length (in years)",
            "Loan intent",
            "Loan grade",
            "Loan amount",
            "Interest rate",
            "Loan status (0 = Non Default, 1 = Default)",
            "Percent income",
            "Historical default",
            "Credit history length"
        ]
    })

    st.dataframe(column_desc, use_container_width=True)

    st.markdown("---")

    st.subheader("🎯 Project Objective")
    st.markdown("""
<div class="highlight-box">
This project analyzes financial attributes of applicants to identify patterns
that influence loan default risk and support data-driven lending decisions.
</div>
""", unsafe_allow_html=True)

    st.subheader("🚀 Project Highlights")
    st.markdown("""
<div class="highlight-box">
• Interactive credit risk dashboard  
• Multiple visualization types  
• Business-oriented insights  
• Executive-ready reporting  
• Scalable fintech structure  
</div>
""", unsafe_allow_html=True)

# ============================================================
# ================= ANALYSIS SECTION ==========================
# ============================================================
else:

    total_records = df.shape[0]
    avg_income = int(df["person_income"].mean())
    avg_loan = int(df["loan_amnt"].mean())
    default_rate = round(df["loan_status"].mean()*100,2)

    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Applicants</div><div class='kpi-value'>{total_records}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Avg Income</div><div class='kpi-value'>₹ {avg_income:,}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Avg Loan</div><div class='kpi-value'>₹ {avg_loan:,}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Default Rate</div><div class='kpi-value'>{default_rate}%</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📈 Interactive Visualization")

    insight_lines = []

    # ================= UNIVARIATE =================
    if analysis_type == "Univariate":

        feature = st.selectbox("Select Feature", num_features + cat_features)

        graph_type = st.selectbox("Graph Type",
                                  ["Histogram","Box Plot","Bar Chart","Pie Chart","Line Chart","Area Chart"])

        if graph_type == "Histogram":
            fig = px.histogram(df, x=feature, template="plotly_dark")

        elif graph_type == "Box Plot":
            fig = px.box(df, x=feature, template="plotly_dark")

        elif graph_type == "Bar Chart":
            counts = df[feature].value_counts().reset_index()
            counts.columns = [feature,"Count"]
            fig = px.bar(counts, x=feature, y="Count", template="plotly_dark")

        elif graph_type == "Pie Chart":
            fig = px.pie(df, names=feature, template="plotly_dark")

        elif graph_type == "Line Chart":
            fig = px.line(df.sort_values(feature), x=feature, template="plotly_dark")

        else:
            fig = px.area(df.sort_values(feature), x=feature, template="plotly_dark")

        if feature in num_features:
            insight_lines = [
                f"Average: {round(df[feature].mean(),2)}",
                f"Maximum: {round(df[feature].max(),2)}",
                f"Minimum: {round(df[feature].min(),2)}"
            ]
        else:
            insight_lines = [
                f"Most frequent: {df[feature].value_counts().idxmax()}",
                f"Unique categories: {df[feature].nunique()}",
                "Distribution varies across customers"
            ]

    # ================= BIVARIATE =================
    else:

        relation_type = st.selectbox("Relationship Type",
                                     ["Num vs Num","Num vs Cat","Cat vs Cat"])

        if relation_type == "Num vs Num":

            x = st.selectbox("Select X", num_features)
            y = st.selectbox("Select Y", num_features,index=1)

            graph_type = st.selectbox("Graph Type",
                                      ["Scatter Plot","Line Chart","Area Chart"])

            if graph_type == "Scatter Plot":
                fig = px.scatter(df, x=x, y=y, color="loan_status",
                                 template="plotly_dark")
            elif graph_type == "Line Chart":
                fig = px.line(df.sort_values(x), x=x, y=y,
                              template="plotly_dark")
            else:
                fig = px.area(df.sort_values(x), x=x, y=y,
                              template="plotly_dark")

            insight_lines = [
                f"Correlation: {round(df[x].corr(df[y]),2)}",
                "Financial relationship observed",
                "Risk clustering visible"
            ]

        elif relation_type == "Num vs Cat":

            num = st.selectbox("Numerical Feature", num_features)
            cat = st.selectbox("Categorical Feature", cat_features)

            graph_type = st.selectbox("Graph Type",
                                      ["Box Plot","Bar Chart"])

            if graph_type == "Box Plot":
                fig = px.box(df, x=cat, y=num, template="plotly_dark")
            else:
                avg_data = df.groupby(cat)[num].mean().reset_index()
                fig = px.bar(avg_data, x=cat, y=num, template="plotly_dark")

            insight_lines = [
                f"{num} varies across {cat}",
                "Median difference visible",
                "Category impacts financial risk"
            ]

        else:

            x = st.selectbox("Category 1", cat_features)
            y = st.selectbox("Category 2", cat_features)

            graph_type = st.selectbox("Graph Type",
                                      ["Grouped Bar Chart"])

            grouped = df.groupby([x,y]).size().reset_index(name="Count")
            fig = px.bar(grouped, x=x, y="Count", color=y,
                         barmode="group", template="plotly_dark")

            insight_lines = [
                f"Relationship between {x} and {y}",
                "Category distribution differs",
                "Some segments show higher risk"
            ]

    st.plotly_chart(fig, use_container_width=True)

    # ===== INSIGHTS (NEW LINE FIXED) =====
    st.markdown('<div class="insight-box">📊 Insights:<br>' +
                "<br>".join([f"• {line}" for line in insight_lines]) +
                "</div>", unsafe_allow_html=True)
