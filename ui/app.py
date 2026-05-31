"""
Streamlit Multi-Page Application for Credit Risk Intelligence Platform
Pages: EDA Dashboard, Risk Prediction, SHAP Explanation, Talk-to-Data Chatbot
Modern Dark Theme Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern Dark Theme
st.markdown("""
<style>
    /* Global Styles */
    :root {
        --primary: #00D4FF;
        --secondary: #7B2FBE;
        --accent: #FF6B6B;
        --success: #00C853;
        --warning: #FFD600;
        --background: #0A0E1A;
        --card-bg: #1A1F2E;
        --text-primary: #FFFFFF;
        --text-secondary: #A0AEC0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0A0E1A 0%, #1A1F2E 100%);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #1A1F2E 0%, #0A0E1A 100%);
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sidebar-version {
        text-align: center;
        color: #A0AEC0;
        font-size: 0.8rem;
        padding: 0.5rem;
        background: rgba(0, 212, 255, 0.1);
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Navigation Buttons */
    .stRadio > label {
        background: linear-gradient(135deg, #1A1F2E 0%, #2A2F3E 100%);
        border: 2px solid #00D4FF;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stRadio > label:hover {
        background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%);
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stRadio > label > div {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #2A2F3E 100%);
        border: 2px solid #00D4FF;
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.4);
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        font-size: 1rem;
        color: #A0AEC0;
        margin: 0;
    }
    
    /* Risk Cards */
    .risk-low { 
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        border-color: #00C853;
        box-shadow: 0 8px 32px rgba(0, 200, 83, 0.3);
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #FFD600 0%, #FFEB3B 100%);
        border-color: #FFD600;
        box-shadow: 0 8px 32px rgba(255, 214, 0, 0.3);
        color: #0A0E1A !important;
    }
    
    .risk-medium h2, .risk-medium p {
        color: #0A0E1A !important;
    }
    
    .risk-high { 
        background: linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%);
        border-color: #FF6B6B;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #2A2F3E 100%);
        border-left: 4px solid #00D4FF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.1);
    }
    
    .info-card h3 {
        color: #00D4FF;
        margin-top: 0;
    }
    
    .info-card p {
        color: #A0AEC0;
        margin-bottom: 0;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 212, 255, 0.5);
    }
    
    /* Input Styles */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background: #1A1F2E;
        border: 2px solid #00D4FF;
        border-radius: 8px;
        color: white;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #7B2FBE;
        box-shadow: 0 0 0 3px rgba(123, 47, 190, 0.2);
    }
    
    /* Chat Styles */
    .chat-container {
        background: #1A1F2E;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3);
    }
    
    .bot-message {
        background: #2A2F3E;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 2px solid #00D4FF;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.1);
    }
    
    /* SQL Code Block */
    .sql-code {
        background: #0A0E1A;
        border: 2px solid #7B2FBE;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        color: #00D4FF;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #00D4FF;
        border-top-color: transparent;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00D4FF 0%, #7B2FBE 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #A0AEC0;
        border-top: 2px solid #00D4FF;
        margin-top: 2rem;
    }
    
    /* Dataframe Styles */
    .stDataFrame {
        background: #1A1F2E;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame table {
        color: white;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%);
        color: white;
    }
    
    /* Chart Container */
    .chart-container {
        background: #1A1F2E;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #00D4FF;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.1);
    }
    
    /* Subheader */
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00D4FF;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #7B2FBE;
    }
    
    /* Example Question Chips */
    .question-chip {
        background: linear-gradient(135deg, #1A1F2E 0%, #2A2F3E 100%);
        border: 2px solid #00D4FF;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
        font-size: 0.9rem;
    }
    
    .question-chip:hover {
        background: linear-gradient(135deg, #00D4FF 0%, #7B2FBE 100%);
        transform: translateY(-2px);
    }
    
    /* Animated Gradient */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animated-gradient {
        background: linear-gradient(-45deg, #00D4FF, #7B2FBE, #FF6B6B, #00C853);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
def load_data_with_demo_mode():
    """Load data with demo mode fallback"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Try to load full dataset first
    data_path = data_dir / "application_train.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df, False  # False = not demo mode
    
    # Try to load sample dataset for demo mode
    sample_path = data_dir / "sample_data.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        return df, True  # True = demo mode
    
    # No dataset found
    return None, False


def load_model_with_demo_mode():
    """Load model with demo mode fallback"""
    models_dir = Path(__file__).parent.parent / "models"
    
    # Check if model exists
    if (models_dir / "lgbm_model.pkl").exists():
        with open(models_dir / "lgbm_model.pkl", 'rb') as f:
            model = pickle.load(f)
        return model, False  # False = not demo mode
    
    return None, False


def check_chart_exists(chart_name):
    """Check if a pre-generated chart exists"""
    chart_path = Path(__file__).parent.parent / "notebooks" / chart_name
    return chart_path.exists()


# ==================== PAGE 1: EDA DASHBOARD ====================
def eda_dashboard():
    st.markdown('<h1 class="main-header">📊 EDA Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data with demo mode
    df, is_demo = load_data_with_demo_mode()
    
    if df is None:
        st.error("🚫 No dataset found. Please download the dataset from Kaggle or use the included sample data.")
        st.info("📥 Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data")
        return
    
    if is_demo:
        st.warning("⚠️ Demo Mode: Using sample dataset (1000 rows). For full functionality, download the complete dataset.")
    
    with st.spinner("🔄 Loading data..."):
        pass  # Data already loaded
    
    # Dataset Summary - Big Metric Cards
    st.markdown('<h2 class="subheader">📈 Dataset Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h2>{len(df):,}</h2>
            <p>📊 Total Records</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h2>{df.shape[1]}</h2>
            <p>🔢 Total Features</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        default_rate = df['TARGET'].mean() * 100
        risk_class = "risk-low" if default_rate < 5 else "risk-medium" if default_rate < 10 else "risk-high"
        st.markdown(f'''
        <div class="metric-card {risk_class}">
            <h2>{default_rate:.2f}%</h2>
            <p>⚠️ Default Rate</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100
        st.markdown(f'''
        <div class="metric-card">
            <h2>{missing_pct:.2f}%</h2>
            <p>🔍 Missing Data</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Data Quality Summary Table
    st.markdown('<h2 class="subheader">✅ Data Quality Summary</h2>', unsafe_allow_html=True)
    
    quality_data = {
        'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values', 'Duplicate Rows'],
        'Value': [
            f"{len(df):,}",
            f"{df.shape[1]}",
            f"{df.select_dtypes(include=[np.number]).shape[1]}",
            f"{df.select_dtypes(include=['object']).shape[1]}",
            f"{df.isnull().sum().sum():,}",
            f"{df.duplicated().sum():,}"
        ],
        'Status': ['✅ Good', '✅ Good', '✅ Good', '✅ Good', '⚠️ Review' if df.isnull().sum().sum() > 0 else '✅ Good', '✅ Good']
    }
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    # Missing Values
    st.markdown('<h2 class="subheader">🔍 Missing Values Analysis</h2>', unsafe_allow_html=True)
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df.head(10), use_container_width=True)
    else:
        st.markdown('''
        <div class="info-card">
            <h3>✅ No Missing Values</h3>
            <p>The dataset is complete with no missing values.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Display Charts in Grid
    st.markdown('<h2 class="subheader">📊 Visualizations</h2>', unsafe_allow_html=True)
    
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    
    plots = {
        "Default Distribution": "default_distribution.png",
        "Income vs Default": "income_vs_default.png",
        "Age vs Default": "age_vs_default.png",
        "Loan Amount vs Default": "loan_amount_vs_default.png",
        "Gender vs Default": "gender_vs_default.png",
        "Education vs Default": "education_vs_default.png"
    }
    
    # Display in 2x3 grid
    for i, (plot_name, plot_file) in enumerate(plots.items()):
        plot_path = notebooks_dir / plot_file
        if plot_path.exists():
            with st.container():
                st.markdown(f'''
                <div class="chart-container">
                    <h3 style="color: #00D4FF; margin-top: 0;">📈 {plot_name}</h3>
                </div>
                ''', unsafe_allow_html=True)
                st.image(str(plot_path), use_container_width=True)
        else:
            st.warning(f"⚠️ Plot not found: {plot_file}. Run src/eda.py to generate.")
    
    # Business Insights as Colored Info Cards
    st.markdown('<h2 class="subheader">💡 Key Business Insights</h2>', unsafe_allow_html=True)
    
    insights = [
        ("🎯 Overall Default Rate", "Approximately 8% of applicants default on their loans.", "#00D4FF"),
        ("👶 Age Factor", "Younger applicants (under 35) have higher default rates.", "#7B2FBE"),
        ("💰 Income Impact", "Lower income correlates with higher default risk.", "#FF6B6B"),
        ("👥 Gender Difference", "Males tend to have slightly higher default rates than females.", "#00C853"),
        ("🎓 Education", "Higher education levels correlate with lower default risk.", "#FFD600")
    ]
    
    for title, insight, color in insights:
        st.markdown(f'''
        <div class="info-card" style="border-left-color: {color};">
            <h3 style="color: {color};">{title}</h3>
            <p>{insight}</p>
        </div>
        ''', unsafe_allow_html=True)


# ==================== PAGE 2: RISK PREDICTION ====================
def risk_prediction():
    st.markdown('<h1 class="main-header">🎯 Risk Prediction</h1>', unsafe_allow_html=True)
    
    # Load model with demo mode
    model, is_demo = load_model_with_demo_mode()
    
    if model is None:
        st.error("🚫 Model not found. Please run src/train.py first.")
        st.info("💡 Demo mode requires a pre-trained model. The model should be in the models/ folder.")
        return
    
    if is_demo:
        st.warning("⚠️ Demo Mode: Using pre-trained model.")
    
    # Load preprocessing artifacts
    feature_names_path = Path(__file__).parent.parent / "data" / "feature_names.pkl"
    metrics_path = Path(__file__).parent.parent / "models" / "metrics.json"
    
    if not feature_names_path.exists():
        st.error("🚫 Feature names not found. Please run src/preprocess.py first.")
        return
    
    # Load feature names
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load and display model metrics
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        st.markdown('<h2 class="subheader">📊 Model Performance Metrics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h2>{metrics.get("auc_roc", "N/A")}</h2>
                <p>📈 AUC-ROC</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h2>{metrics.get("accuracy", "N/A")}</h2>
                <p>🎯 Accuracy</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <h2>{metrics.get("precision", "N/A")}</h2>
                <p>🔍 Precision</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <h2>{metrics.get("recall", "N/A")}</h2>
                <p>📋 Recall</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col5:
            st.markdown(f'''
            <div class="metric-card">
                <h2>{metrics.get("f1", "N/A")}</h2>
                <p>⚖️ F1 Score</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Input Form - Two Column Layout with Icons
    st.markdown('<h2 class="subheader">📝 Enter Applicant Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card"><h3>💰 Financial Information</h3></div>', unsafe_allow_html=True)
        amt_income_total = st.number_input("💵 Total Income", min_value=0, value=100000, step=1000)
        amt_credit = st.number_input("🏦 Loan Amount", min_value=0, value=500000, step=10000)
        amt_annuity = st.number_input("📅 Annuity Amount", min_value=0, value=25000, step=500)
    
    with col2:
        st.markdown('<div class="info-card"><h3>👤 Personal Information</h3></div>', unsafe_allow_html=True)
        days_birth = st.number_input("🎂 Age in Days (negative)", value=-12000, step=-365)
        days_employed = st.number_input("💼 Employment Days (negative, 365243=unemployed)", value=-2000, step=-365)
        code_gender = st.selectbox("🚻 Gender", ["M", "F"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="info-card"><h3>🏠 Assets & Education</h3></div>', unsafe_allow_html=True)
        flag_own_car = st.selectbox("🚗 Owns Car", ["Y", "N"])
        flag_own_realty = st.selectbox("🏡 Owns Real Estate", ["Y", "N"])
        name_education_type = st.selectbox(
            "🎓 Education Level",
            ["Secondary / secondary special", "Higher education", "Incomplete higher", 
             "Lower secondary", "Academic degree"]
        )
    
    with col4:
        st.markdown('<div class="info-card"><h3>ℹ️ Information</h3></div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="info-card">
            <p><strong>💡 Tips:</strong></p>
            <ul>
                <li>Age in days: Use negative value (e.g., -12000 for ~33 years)</li>
                <li>Employment: -365243 means unemployed</li>
                <li>Higher income & education = Lower risk</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    # Predict Button
    if st.button("🔮 Predict Risk", type="primary"):
        with st.spinner("🔄 Analyzing applicant data..."):
            # Encode categorical variables
            gender_encoding = {'M': 1, 'F': 0, 'XNA': 2}
            car_encoding = {'Y': 1, 'N': 0}
            realty_encoding = {'Y': 1, 'N': 0}
            education_encoding = {
                'Secondary / secondary special': 1,
                'Higher education': 2,
                'Incomplete higher': 3,
                'Lower secondary': 4,
                'Academic degree': 5
            }
            
            # Create feature dictionary with encoded values
            features = {
                'AMT_INCOME_TOTAL': amt_income_total,
                'AMT_CREDIT': amt_credit,
                'AMT_ANNUITY': amt_annuity,
                'DAYS_BIRTH': days_birth,
                'DAYS_EMPLOYED': days_employed,
                'CODE_GENDER': gender_encoding.get(code_gender, 0),
                'FLAG_OWN_CAR': car_encoding.get(flag_own_car, 0),
                'FLAG_OWN_REALTY': realty_encoding.get(flag_own_realty, 0),
                'NAME_EDUCATION_TYPE': education_encoding.get(name_education_type, 1),
            }
            
            # Add engineered features
            features['AGE_YEARS'] = abs(days_birth) / 365.25
            features['EMPLOYMENT_YEARS'] = abs(days_employed) / 365.25 if days_employed != 365243 else 0
            features['INCOME_CREDIT_RATIO'] = amt_income_total / (amt_credit + 1)
            features['CREDIT_ANNUITY_RATIO'] = amt_credit / (amt_annuity + 1)
            features['INCOME_PER_PERSON'] = amt_income_total / 2  # Assuming 2 family members
            features['AGE_YEARS_SQUARED'] = features['AGE_YEARS'] ** 2
            features['LOG_INCOME'] = np.log1p(amt_income_total)
            features['LOG_CREDIT'] = np.log1p(amt_credit)
            features['PAYMENT_BURDEN'] = amt_annuity / (amt_income_total + 1)
            
            # Load feature names from models folder
            feature_names_path = Path(__file__).parent.parent / "data" / "feature_names.pkl"
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
            
            # Add missing features with default values
            missing_features = set(feature_names) - set(features.keys())
            for feat in missing_features:
                if 'EXT_SOURCE' in feat:
                    features[feat] = 0.5
                elif 'FLAG_DOCUMENT' in feat:
                    features[feat] = 0
                elif 'FLAG_PHONE' in feat or 'FLAG_CONT' in feat:
                    features[feat] = 0
                elif 'CNT' in feat:
                    features[feat] = 0
                else:
                    features[feat] = 0
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([features])
            input_df = input_df[feature_names]
            
            # Ensure all columns are numeric (int or float)
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                    input_df[col] = input_df[col].fillna(0)
            
            # Make prediction
            try:
                prediction_proba = model.predict(input_df, num_iteration=model.best_iteration)[0]
                risk_score = prediction_proba * 100
                
                # Determine risk band
                if risk_score < 30:
                    risk_band = "Low"
                    risk_class = "risk-low"
                    decision = "✅ APPROVE"
                    emoji = "🟢"
                elif risk_score < 60:
                    risk_band = "Medium"
                    risk_class = "risk-medium"
                    decision = "⚠️ REVIEW"
                    emoji = "🟡"
                else:
                    risk_band = "High"
                    risk_class = "risk-high"
                    decision = "❌ REJECT"
                    emoji = "🔴"
                
                # Display Results with Big Animated Gauge
                st.markdown('<h2 class="subheader">🎯 Prediction Results</h2>', unsafe_allow_html=True)
                
                # Risk Score Gauge
                st.markdown(f'''
                <div class="chart-container" style="text-align: center;">
                    <h3 style="color: #00D4FF; margin-top: 0;">Risk Score Gauge</h3>
                    <div style="font-size: 4rem; font-weight: bold; color: {get_risk_color(risk_score)};">
                        {risk_score:.1f}%
                    </div>
                    <div style="font-size: 1.5rem; margin-top: 1rem;">
                        {emoji} {risk_band} Risk
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Progress bar for risk score
                st.progress(risk_score / 100)
                
                # Decision Card
                st.markdown(f'''
                <div class="metric-card {risk_class}" style="margin: 1rem 0;">
                    <h2 style="font-size: 3rem;">{decision}</h2>
                    <p>Loan Decision Recommendation</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Additional Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <h2>{risk_score:.2f}%</h2>
                        <p>📊 Risk Score</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="metric-card {risk_class}">
                        <h2>{risk_band}</h2>
                        <p>🎯 Risk Band</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    confidence = (1 - abs(risk_score - 50) / 50) * 100
                    st.markdown(f'''
                    <div class="metric-card">
                        <h2>{confidence:.1f}%</h2>
                        <p>🔍 Model Confidence</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Interpretation Card
                st.markdown('''
                <div class="info-card">
                    <h3>📖 Interpretation Guide</h3>
                    <ul>
                        <li><strong>Risk Score</strong>: Probability of default (0-100%)</li>
                        <li><strong>Low Risk</strong> (<30%): ✅ High confidence approval</li>
                        <li><strong>Medium Risk</strong> (30-60%): ⚠️ Manual review recommended</li>
                        <li><strong>High Risk</strong> (>60%): ❌ High probability of default</li>
                    </ul>
                </div>
                ''', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"🚫 Error making prediction: {str(e)}")


def get_risk_color(score):
    if score < 30:
        return "#00C853"
    elif score < 60:
        return "#FFD600"
    else:
        return "#FF6B6B"


# ==================== PAGE 3: SHAP EXPLANATION ====================
def shap_explanation():
    st.markdown('<h1 class="main-header">🔍 SHAP Explanation</h1>', unsafe_allow_html=True)
    
    # Load model with demo mode
    model, is_demo = load_model_with_demo_mode()
    
    if model is None:
        st.error("🚫 Model not found. Please run src/train.py first.")
        st.info("💡 Displaying pre-generated SHAP charts if available.")
    
    if is_demo:
        st.warning("⚠️ Demo Mode: Using pre-trained model.")
    
    # Load explainer and data
    explainer_path = Path(__file__).parent.parent / "notebooks" / "shap_explainer.pkl"
    feature_importance_path = Path(__file__).parent.parent / "models" / "feature_importance.csv"
    
    # Display pre-generated charts
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    shap_charts = [
        ("SHAP Summary Plot", "shap_summary_plot.png"),
        ("SHAP Feature Importance", "shap_feature_importance.png"),
        ("SHAP Waterfall Sample", "shap_waterfall_sample_0.png")
    ]
    
    st.markdown('<h2 class="subheader">📊 Pre-Generated SHAP Charts</h2>', unsafe_allow_html=True)
    
    charts_found = False
    for chart_name, chart_file in shap_charts:
        chart_path = notebooks_dir / chart_file
        if chart_path.exists():
            charts_found = True
            with st.container():
                st.markdown(f'''
                <div class="chart-container">
                    <h3 style="color: #00D4FF; margin-top: 0;">📈 {chart_name}</h3>
                </div>
                ''', unsafe_allow_html=True)
                st.image(str(chart_path), use_container_width=True)
    
    if not charts_found:
        st.warning("⚠️ No pre-generated SHAP charts found. Run src/explain.py to generate.")
    
    # Load feature importance
    if feature_importance_path.exists():
        feature_importance = pd.read_csv(feature_importance_path)
        
        st.markdown('<h2 class="subheader">📊 Feature Importance</h2>', unsafe_allow_html=True)
        
        # Create colored progress bars for feature importance
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        for idx, row in feature_importance.head(10).iterrows():
            importance = row['importance']
            feature = row['feature']
            color = get_importance_color(idx, 10)
            st.markdown(f'''
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: white;">{feature}</span>
                    <span style="color: #00D4FF;">{importance:.4f}</span>
                </div>
                <div style="background: #2A2F3E; border-radius: 8px; overflow: hidden;">
                    <div style="background: {color}; height: 8px; border-radius: 8px; width: {(importance / feature_importance['importance'].max()) * 100}%;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plotly chart
        fig = px.bar(feature_importance.head(15), x='importance', y='feature', 
                     orientation='h', title='Top 15 Features by SHAP Importance')
        fig.update_layout(
            yaxis={'categoryorder':'total ascending', 'gridcolor': '#2A2F3E'},
            plot_bgcolor='#1A1F2E',
            paper_bgcolor='#1A1F2E',
            font={'color': '#FFFFFF'},
            xaxis={'gridcolor': '#2A2F3E'}
        )
        fig.update_traces(marker_color='#00D4FF')
        st.plotly_chart(fig, use_container_width=True)
    
    # Display SHAP plots in clean card backgrounds
    st.markdown('<h2 class="subheader">📈 SHAP Visualizations</h2>', unsafe_allow_html=True)
    
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    
    shap_plots = {
        ("SHAP Summary Plot", "Shows how each feature impacts predictions across the dataset", "shap_summary_plot.png"),
        ("SHAP Feature Importance", "Ranks features by their average impact on model output", "shap_feature_importance.png"),
        ("Sample Waterfall Plot", "Shows how each feature contributed to a specific prediction", "shap_waterfall_sample_0.png"),
        ("Sample Force Plot", "Visualizes the push and pull of features for a single prediction", "shap_force_sample_0.png")
    }
    
    for plot_name, description, plot_file in shap_plots:
        plot_path = notebooks_dir / plot_file
        if plot_path.exists():
            st.markdown(f'''
            <div class="chart-container">
                <h3 style="color: #00D4FF; margin-top: 0;">📊 {plot_name}</h3>
                <p style="color: #A0AEC0; margin-bottom: 1rem;">{description}</p>
            </div>
            ''', unsafe_allow_html=True)
            st.image(str(plot_path), use_container_width=True)
        else:
            st.warning(f"⚠️ Plot not found: {plot_file}")
    
    # Explanation card
    st.markdown('''
    <div class="info-card">
        <h3>📖 Understanding SHAP Plots</h3>
        <ul>
            <li><strong>Summary Plot</strong>: Shows how each feature impacts predictions across the dataset</li>
            <li><strong>Feature Importance</strong>: Ranks features by their average impact on model output</li>
            <li><strong>Waterfall Plot</strong>: Shows how each feature contributed to a specific prediction</li>
            <li><strong>Force Plot</strong>: Visualizes the push and pull of features for a single prediction</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)


def get_importance_color(idx, total):
    colors = ['#00D4FF', '#7B2FBE', '#FF6B6B', '#00C853', '#FFD600']
    return colors[idx % len(colors)]


# ==================== PAGE 4: TALK-TO-DATA CHATBOT ====================
def talk_to_data():
    st.markdown('<h1 class="main-header">💬 Talk-to-Data Chatbot</h1>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="subheader">🤖 Ask questions about the credit risk data in plain English</h2>', unsafe_allow_html=True)
    
    # Load data with demo mode
    df, is_demo = load_data_with_demo_mode()
    
    if df is None:
        st.error("🚫 No dataset found. Please download the dataset from Kaggle or use the included sample data.")
        st.info("📥 Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data")
        return
    
    if is_demo:
        st.warning("⚠️ Demo Mode: Using sample dataset (1000 rows). Chatbot will answer based on this limited data.")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        st.error("🚫 GROQ_API_KEY not found in .env file. Please add it.")
        st.info("💡 Get your API key from: https://console.groq.com/keys")
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Example questions as clickable chips
    st.markdown('<h3 style="color: #00D4FF; margin: 1rem 0;">💡 Example Questions</h3>', unsafe_allow_html=True)
    
    example_questions = [
        "How many people defaulted?",
        "What is the average income of defaulters?",
        "Which gender has higher default rate?",
        "What is the average loan amount?",
        "How many applicants own a car?"
    ]
    
    # Display example questions as chips
    chip_container = st.container()
    with chip_container:
        for question in example_questions:
            st.markdown(f'''
            <div class="question-chip" onclick="document.querySelector('input[data-testid=\"stTextInputInput\"]').value = '{question}';">
                {question}
            </div>
            ''', unsafe_allow_html=True)
    
    # User input
    st.markdown('<h3 style="color: #00D4FF; margin: 1rem 0;">💭 Your Question</h3>', unsafe_allow_html=True)
    user_question = st.text_input("", key="user_question", placeholder="Type your question here...")
    
    if st.button("🚀 Ask", type="primary") and user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        with st.spinner("🤖 Thinking..."):
            try:
                # Import chatbot
                from chatbot import CreditRiskChatbot
                
                # Initialize chatbot (cached)
                if 'chatbot' not in st.session_state:
                    st.session_state.chatbot = CreditRiskChatbot()
                
                # Get answer
                answer = st.session_state.chatbot.ask(user_question)
                
                # Add bot response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer
                })
                
            except Exception as e:
                st.error(f"🚫 Error: {str(e)}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"Sorry, I encountered an error: {str(e)}"
                })
    
    # Display chat history with bubble style
    if st.session_state.chat_history:
        st.markdown('<h3 style="color: #00D4FF; margin: 1rem 0;">💬 Conversation History</h3>', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'''
                <div class="user-message">
                    <strong>👤 You:</strong> {message['content']}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="bot-message">
                    <strong>🤖 Assistant:</strong> {message['content']}
                </div>
                ''', unsafe_allow_html=True)
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Info card
    st.markdown('''
    <div class="info-card">
        <h3>💡 Tips for Better Results</h3>
        <ul>
            <li>Ask specific questions about the data</li>
            <li>Use clear, simple language</li>
            <li>Try the example questions above</li>
            <li>The chatbot converts your question to SQL and queries the database</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)


# ==================== MAIN APP ====================
def main():
    # Sidebar with logo and gradient navigation
    st.sidebar.markdown('''
    <div class="sidebar-logo">
        🏦 Credit Risk Platform
    </div>
    ''', unsafe_allow_html=True)
    
    st.sidebar.markdown('''
    <div class="sidebar-version">
        📌 Version 1.0.0 | ✅ Online
    </div>
    ''', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Navigation with gradient buttons
    page = st.sidebar.radio(
        "🚀 Navigate to:",
        ["📊 EDA Dashboard", "🎯 Risk Prediction", "🔍 SHAP Explanation", "💬 Talk-to-Data Chatbot"],
        label_visibility="collapsed"
    )
    
    # Page routing
    if page == "📊 EDA Dashboard":
        eda_dashboard()
    elif page == "🎯 Risk Prediction":
        risk_prediction()
    elif page == "🔍 SHAP Explanation":
        shap_explanation()
    elif page == "💬 Talk-to-Data Chatbot":
        talk_to_data()
    
    # Footer with project info
    st.sidebar.markdown("---")
    st.sidebar.markdown('''
    <div style="text-align: center; color: #A0AEC0; padding: 1rem;">
        <h4 style="color: #00D4FF; margin-top: 0;">🛠️ Built With</h4>
        <p style="font-size: 0.9rem;">
            🤖 LightGBM<br>
            🔍 SHAP<br>
            💬 Groq API<br>
            🎨 Streamlit
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Main footer
    st.markdown('''
    <div class="footer">
        <p><strong>🏦 Credit Risk Intelligence Platform</strong></p>
        <p style="font-size: 0.9rem;">AI-Powered Loan Default Prediction System</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">Built with ❤️ using Machine Learning & Explainable AI</p>
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
