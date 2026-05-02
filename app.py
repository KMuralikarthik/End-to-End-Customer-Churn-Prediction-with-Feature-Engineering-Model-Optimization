import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import io

# Set page config for a wide layout
st.set_page_config(page_title="Telco Customer Churn Dashboard", page_icon="📊", layout="wide")

# Custom CSS for the dashboard aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #F8F5F0;
    }
    .stMetric {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .header-container {
        background-color: #2C5F6C;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
        color: white;
    }
    h1, h2, h3 {
        color: #2C5F6C;
    }
    .filter-container {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2C5F6C !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Colors
COLOR_TEAL = "#2C5F6C"
COLOR_GOLD = "#E1AD01"
COLOR_BROWN = "#A0522D"

@st.cache_data
def load_data():
    df = pd.read_csv("data/churn.csv")
    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # Create Tenure Group
    def tenure_group(t):
        if t <= 6: return "0-6 Months"
        elif t <= 12: return "7-12 Months"
        else: return "1+ Years"
    df['Tenure Group'] = df['tenure'].apply(tenure_group)
    
    return df

@st.cache_resource
def load_ml_model():
    if not os.path.exists("model/churn_model.pkl"):
        return None
    with open("model/churn_model.pkl", "rb") as file:
        return pickle.load(file)

# Load Data
df_raw = load_data()
model_data = load_ml_model()

# Header
st.markdown('<div class="header-container"><h1>Telco Customer Churn Analytics Platform</h1></div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_dashboard, tab_predict, tab_batch = st.tabs(["📈 Executive Dashboard", "🔮 Manual Prediction", "📂 Batch Processing"])

with tab_dashboard:
    # Sidebar Filters (Only relevant for dashboard)
    st.sidebar.title("Dashboard Filters")
    contract_filter = st.sidebar.multiselect("Contract", options=df_raw['Contract'].unique(), default=df_raw['Contract'].unique())
    payment_filter = st.sidebar.multiselect("Payment Method", options=df_raw['PaymentMethod'].unique(), default=df_raw['PaymentMethod'].unique())
    internet_filter = st.sidebar.multiselect("Internet Service", options=df_raw['InternetService'].unique(), default=df_raw['InternetService'].unique())
    gender_filter = st.sidebar.multiselect("Gender", options=df_raw['gender'].unique(), default=df_raw['gender'].unique())

    # Filter Data
    df = df_raw[
        (df_raw['Contract'].isin(contract_filter)) &
        (df_raw['PaymentMethod'].isin(payment_filter)) &
        (df_raw['InternetService'].isin(internet_filter)) &
        (df_raw['gender'].isin(gender_filter))
    ]

    # KPI Metrics
    total_customers = len(df)
    churn_count = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churn_count / total_customers * 100) if total_customers > 0 else 0
    retention_rate = 100 - churn_rate
    arpu = df['MonthlyCharges'].mean() if total_customers > 0 else 0
    total_revenue = df['TotalCharges'].sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Customers", f"{total_customers:,}")
    m2.metric("Churned Customers", f"{churn_count:,}")
    m3.metric("Churn Rate %", f"{churn_rate:.1f}%")
    m4.metric("Retention Rate %", f"{retention_rate:.1f}%")
    m5.metric("ARPU", f"${arpu:.2f}")
    m6.metric("Total Revenue", f"${total_revenue/1000:.1f}K")

    st.markdown("---")

    # Row 1: Main Charts
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Customer Churn by Contract & Churn Status")
        group_df = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig_main = px.bar(group_df, x='Count', y='Contract', color='Churn', orientation='h',
                         color_discrete_map={'No': COLOR_TEAL, 'Yes': COLOR_GOLD},
                         category_orders={"Contract": ["Month-to-month", "One year", "Two year"]})
        fig_main.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_main, use_container_width=True)

    with c2:
        st.subheader("Churn Rate (%) by Tenure Group")
        tenure_churn = df.groupby('Tenure Group')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index(name='Churn Rate %')
        tenure_churn['Sort'] = tenure_churn['Tenure Group'].map({"0-6 Months": 0, "7-12 Months": 1, "1+ Years": 2})
        tenure_churn = tenure_churn.sort_values('Sort')
        
        fig_tenure = px.bar(tenure_churn, x='Tenure Group', y='Churn Rate %',
                           color='Tenure Group', color_discrete_sequence=[COLOR_TEAL, COLOR_GOLD, COLOR_BROWN])
        fig_tenure.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_tenure, use_container_width=True)

    # Row 2: Secondary Charts
    c3, c4, c5 = st.columns(3)

    with c3:
        st.subheader("Churn Rate (%) by Partner")
        partner_churn = df.groupby('Partner')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index(name='Churn Rate %')
        fig_partner = px.bar(partner_churn, x='Churn Rate %', y='Partner', orientation='h',
                            color_discrete_sequence=[COLOR_TEAL])
        fig_partner.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_partner, use_container_width=True)

    with c4:
        st.subheader("Churn Rate (%) by Senior Citizen")
        senior_churn = df.groupby('SeniorCitizen')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index(name='Churn Rate %')
        senior_churn['SeniorCitizen'] = senior_churn['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
        fig_senior = px.pie(senior_churn, values='Churn Rate %', names='SeniorCitizen', hole=0.5,
                           color_discrete_sequence=[COLOR_TEAL, COLOR_GOLD])
        fig_senior.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_senior, use_container_width=True)

    with c5:
        st.subheader("Churn Rate (%) by Dependents")
        dept_churn = df.groupby('Dependents')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index(name='Churn Rate %')
        fig_dept = px.bar(dept_churn, x='Churn Rate %', y='Dependents', orientation='h',
                         color_discrete_sequence=[COLOR_TEAL])
        fig_dept.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_dept, use_container_width=True)

with tab_predict:
    st.subheader("🔮 Individual Customer Prediction")
    st.info("Fill in the customer details below to calculate the likelihood of them leaving your service.")
    
    if model_data:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            p_monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            p_contract = st.selectbox("Contract Type", options=df_raw['Contract'].unique())
        with col_p2:
            p_internet = st.selectbox("Internet Service Type", options=df_raw['InternetService'].unique())
            p_payment = st.selectbox("Payment Method Type", options=df_raw['PaymentMethod'].unique())
            p_total = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
        
        if st.button("Calculate Churn Probability", use_container_width=True):
            input_dict = {
                "tenure": p_tenure, "MonthlyCharges": p_monthly, "TotalCharges": p_total,
                "Contract": p_contract, "InternetService": p_internet, "PaymentMethod": p_payment,
                "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
                "PaperlessBilling": "Yes"
            }
            input_df = pd.DataFrame([input_dict])
            input_encoded = pd.get_dummies(input_df).reindex(columns=model_data["features"], fill_value=0)
            
            prob = model_data["model"].predict_proba(input_encoded)[0][1]
            
            # Gauge Chart for Probability
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability %", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': COLOR_GOLD if prob > 0.5 else COLOR_TEAL},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(44, 95, 108, 0.2)'},
                        {'range': [30, 70], 'color': 'rgba(225, 173, 1, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(160, 82, 45, 0.2)'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50}}))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prob > 0.5:
                st.error(f"⚠️ HIGH RISK: This customer has a {prob:.1%} chance of churning.")
            else:
                st.success(f"✅ LOW RISK: This customer has a {prob:.1%} chance of churning.")
    else:
        st.warning("Model not found. Please run the training script first.")

with tab_batch:
    st.subheader("📂 Batch Customer Analysis")
    st.write("Upload a CSV file containing customer data to get predictions for multiple clients at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None and model_data:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} customers for analysis.")
            
            # Preprocessing
            if 'customerID' in batch_df.columns:
                batch_df_clean = batch_df.drop('customerID', axis=1)
            else:
                batch_df_clean = batch_df.copy()
                
            if 'TotalCharges' in batch_df_clean.columns:
                batch_df_clean['TotalCharges'] = pd.to_numeric(batch_df_clean['TotalCharges'], errors='coerce').fillna(0)
            
            # Encoding
            batch_encoded = pd.get_dummies(batch_df_clean).reindex(columns=model_data["features"], fill_value=0)
            
            # Predictions
            probs = model_data["model"].predict_proba(batch_encoded)[:, 1]
            batch_df['Churn Probability (%)'] = (probs * 100).round(2)
            batch_df['Churn Prediction'] = np.where(probs > 0.5, "Yes", "No")
            
            # Summary Metrics
            at_risk = len(batch_df[batch_df['Churn Prediction'] == "Yes"])
            avg_prob = batch_df['Churn Probability (%)'].mean()
            
            col_b1, col_b2 = st.columns(2)
            col_b1.metric("Customers At Risk", f"{at_risk}")
            col_b2.metric("Average Churn Risk", f"{avg_prob:.1f}%")
            
            st.markdown("---")
            st.write("### Prediction Results")
            st.dataframe(batch_df, use_container_width=True)
            
            # Download button
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    elif uploaded_file and not model_data:
        st.warning("Model not found. Please run the training script first.")

st.markdown("---")
st.caption("Interactive Telco Churn Analytics Dashboard | Professional Edition")
