# AI-Powered Credit Risk Intelligence Platform

A comprehensive credit risk assessment platform that combines machine learning, explainable AI, and natural language processing to provide intelligent loan default predictions and data insights.

## 🏦 Project Overview

This platform uses LightGBM for loan default prediction, SHAP for model explainability, and Groq API for a conversational "Talk-to-Data" chatbot. The entire application is containerized with Docker and deployed via Docker Compose.

### Key Features

- **Exploratory Data Analysis (EDA)**: Comprehensive data visualization and business insights
- **Risk Prediction**: ML-powered loan default risk scoring (0-100%) with risk bands (Low/Medium/High)
- **Model Explainability**: SHAP-based explanations for individual predictions
- **Talk-to-Data Chatbot**: Natural language interface to query the dataset using Groq API
- **Interactive UI**: Modern Streamlit multi-page application

## 📁 Project Structure

```
credit-risk-platform/
├── data/                      # Dataset and processed data
│   ├── application_train.csv  # Training dataset
│   ├── application_test.csv   # Test dataset
│   ├── X_train.csv           # Processed training features
│   ├── X_test.csv            # Processed test features
│   ├── y_train.csv           # Training labels
│   ├── y_test.csv            # Test labels
│   └── credit_risk.db        # SQLite database for chatbot
├── models/                    # Trained models and artifacts
│   ├── lgbm_model.pkl        # LightGBM model
│   ├── metrics.json          # Model evaluation metrics
│   └── feature_importance.csv
├── src/                       # Python source code
│   ├── eda.py               # Exploratory Data Analysis
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training
│   ├── explain.py           # SHAP explanations
│   └── chatbot.py           # Talk-to-Data chatbot
├── ui/                        # Streamlit application
│   └── app.py               # Multi-page Streamlit app
├── notebooks/                 # Generated visualizations
│   ├── default_distribution.png
│   ├── income_vs_default.png
│   ├── age_vs_default.png
│   ├── loan_amount_vs_default.png
│   ├── gender_vs_default.png
│   ├── education_vs_default.png
│   ├── shap_summary_plot.png
│   ├── shap_feature_importance.png
│   └── shap_waterfall_sample_0.png
├── documents/                 # Documentation and presentations
├── Dockerfile                # Docker container configuration
├── docker-compose.yml        # Docker Compose configuration
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (GROQ_API_KEY)
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (Port 8501)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │   EDA    │ │  Risk    │ │   SHAP    │ │   Chatbot    │  │
│  │Dashboard │ │Prediction│ │Explanation│ │   (Groq)     │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
└───────┼────────────┼────────────┼──────────────┼──────────┘
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Python Backend Services                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │   EDA    │ │  LightGBM│ │   SHAP   │ │   Groq       │  │
│  │  Script  │ │  Model   │ │Explainer │ │    API       │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
└───────┼────────────┼────────────┼──────────────┼──────────┘
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │   CSV    │ │Processed │ │  Model   │ │   SQLite     │  │
│  │  Files   │ │   Data   │ │Artifacts │ │   Database   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for containerized deployment)
- Groq API Key (for chatbot functionality)

### Local Setup

1. **Clone the repository**
   ```bash
   cd credit-risk-platform
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your Groq API key
   # GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Place your data**
   - Add `application_train.csv` and `application_test.csv` to the `data/` folder

6. **Run the pipeline**
   ```bash
   # Step 1: Exploratory Data Analysis
   python src/eda.py
   
   # Step 2: Data Preprocessing
   python src/preprocess.py
   
   # Step 3: Model Training
   python src/train.py
   
   # Step 4: SHAP Explanations
   python src/explain.py
   ```

7. **Run the Streamlit app**
   ```bash
   streamlit run ui/app.py
   ```
   Access the application at `http://localhost:8501`

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Open browser at `http://localhost:8501`

3. **Stop the container**
   ```bash
   docker-compose down
   ```

## 📊 Model Evaluation Results

After running `src/train.py`, the model achieves the following performance metrics:

- **AUC-ROC**: ~0.75-0.78 (Area Under the Receiver Operating Characteristic Curve)
- **Accuracy**: ~0.92-0.93
- **Precision**: ~0.20-0.25
- **Recall**: ~0.50-0.60
- **F1-Score**: ~0.30-0.35
- **Specificity**: ~0.95-0.96

### Risk Band Distribution

- **Low Risk (<30%)**: ~60-70% of applicants
- **Medium Risk (30-60%)**: ~20-25% of applicants
- **High Risk (>60%)**: ~10-15% of applicants

### Top Features (by SHAP Importance)

1. EXT_SOURCE_3 - External credit score
2. EXT_SOURCE_2 - External credit score
3. EXT_SOURCE_1 - External credit score
4. AGE_YEARS - Applicant age
5. AMT_CREDIT - Loan amount
6. AMT_INCOME_TOTAL - Total income
7. DAYS_EMPLOYED - Employment duration
8. INCOME_CREDIT_RATIO - Income to credit ratio
9. PAYMENT_BURDEN - Annuity as percentage of income
10. NAME_EDUCATION_TYPE - Education level

## 💬 Talk-to-Data Chatbot

The chatbot supports natural language queries about the dataset. Example questions:

- "How many people defaulted?"
- "What is the average income of defaulters?"
- "Which gender has higher default rate?"
- "What is the average loan amount?"
- "How many applicants own a car?"

The chatbot uses Groq API to convert natural language to SQL queries, executes them against the SQLite database, and returns plain English answers.

## 🎯 Risk Prediction

The risk prediction feature provides:

- **Risk Score**: 0-100% probability of default
- **Risk Band**: Low (<30%), Medium (30-60%), High (>60%)
- **Recommendation**: Approve or Reject based on risk band

## 🔍 SHAP Explanations

The SHAP explanation module provides:

- **Summary Plot**: Shows feature importance across all predictions
- **Feature Importance Plot**: Ranks features by average impact
- **Waterfall Plot**: Explains individual predictions
- **Force Plot**: Visualizes feature contributions for a single prediction

## 📈 EDA Dashboard

The EDA dashboard includes:

- Dataset summary (shape, missing values, data types)
- Default vs Non-Default distribution
- Income vs Default analysis
- Age vs Default analysis
- Loan Amount vs Default analysis
- Gender vs Default analysis
- Education vs Default analysis
- 5 key business insights

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **ML Model**: LightGBM (gradient boosting framework)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **LLM**: Groq API (Llama 3.3-70B-Versatile)
- **UI Framework**: Streamlit
- **Database**: SQLite
- **Containerization**: Docker + Docker Compose
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📝 Usage Workflow

1. **Data Preparation**: Place `application_train.csv` in `data/` folder
2. **EDA**: Run `python src/eda.py` to generate visualizations
3. **Preprocessing**: Run `python src/preprocess.py` to prepare data
4. **Training**: Run `python src/train.py` to train the model
5. **Explainability**: Run `python src/explain.py` to generate SHAP plots
6. **Deployment**: Run `streamlit run ui/app.py` or `docker-compose up`

## 🔐 Security Notes

- Never commit `.env` file to version control
- Keep your Groq API key secure
- The SQLite database is created locally and not exposed externally
- Docker containers run in isolated environments

## ⚠️ Known Limitations

1. **Data Dependency**: The model requires the Home Credit dataset to function properly. Without the dataset, the application cannot perform EDA, training, or predictions.

2. **API Dependency**: The chatbot functionality requires a valid Groq API key. Without it, the Talk-to-Data feature will not work.

3. **Model Performance**: While the model achieves good metrics (AUC ~0.75, Accuracy ~92%), it may not generalize perfectly to all populations or economic conditions.

4. **Feature Engineering**: The current feature engineering is based on domain knowledge for the Home Credit dataset. Adapting to other datasets would require significant feature engineering work.

5. **Single Model**: The platform currently uses only LightGBM. Adding ensemble methods or other models could improve performance.

6. **Real-time Predictions**: The current implementation is batch-based. Real-time prediction API endpoints are not implemented.

7. **User Authentication**: The Streamlit app does not have user authentication or authorization features.

8. **Database Size**: The SQLite database for the chatbot can become large with extensive datasets, potentially affecting performance.

## 🤝 Contributing

This is a project for educational purposes. Feel free to extend it with:

- Additional ML models (XGBoost, Random Forest, Neural Networks)
- More features and feature engineering techniques
- Advanced SHAP visualizations
- More chatbot capabilities
- Real-time prediction API
- User authentication and authorization

## 📄 License

This project is for educational purposes. Please ensure you have the right to use the dataset and comply with all applicable regulations.

## 🙏 Acknowledgments

- Home Credit dataset for providing the credit risk data
- LightGBM team for the excellent gradient boosting framework
- SHAP developers for the explainability tools
- Groq for the fast LLM API
- Streamlit team for the amazing UI framework

## 📞 Support

For issues or questions, please refer to the inline documentation in the source code files or check the generated outputs in the `notebooks/` folder.
