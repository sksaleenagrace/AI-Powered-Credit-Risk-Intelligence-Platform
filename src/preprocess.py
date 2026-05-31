"""
Data Preprocessing for Credit Risk Model
Handles missing values, encoding, feature engineering, and train/test split
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the training dataset"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Try to load full dataset first
    data_path = data_dir / "application_train.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"✓ Loaded full dataset: {len(df)} rows")
        return df
    
    # Try to load sample dataset for demo mode
    sample_path = data_dir / "sample_data.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        print(f"⚠ Full dataset not found. Using sample dataset for demo: {len(df)} rows")
        print(f"⚠ For full functionality, download dataset from:")
        print(f"   https://www.kaggle.com/competitions/home-credit-default-risk/data")
        print(f"   Place application_train.csv in the data/ folder")
        return df
    
    # No dataset found
    print("❌ ERROR: No dataset found!")
    print("Please download the dataset from:")
    print("   https://www.kaggle.com/competitions/home-credit-default-risk/data")
    print("Place application_train.csv in the data/ folder")
    raise FileNotFoundError("Dataset not found. Please download from Kaggle and place in data/ folder.")


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("Handling missing values...")
    
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    
    print(f"✓ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
    return df


def encode_categorical_columns(df):
    """Encode categorical columns using Label Encoding"""
    print("Encoding categorical columns...")
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Store label encoders for later use
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle unseen categories by fitting on all unique values
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    print(f"✓ Encoded {len(categorical_cols)} categorical columns")
    return df, label_encoders


def feature_engineering(df):
    """Create new features from existing data"""
    print("Performing feature engineering...")
    
    # Feature 1: Age in years (from DAYS_BIRTH)
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25
    
    # Feature 2: Employment duration in years (from DAYS_EMPLOYED)
    df['EMPLOYMENT_YEARS'] = abs(df['DAYS_EMPLOYED']) / 365.25
    # Handle anomaly: DAYS_EMPLOYED = 365243 means unemployed
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].replace(365243/365.25, 0)  # Set to 0 for unemployed
    
    # Feature 3: Income to Credit ratio
    df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1)
    
    # Feature 4: Credit to Annuity ratio
    df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    
    # Feature 5: Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    
    # Feature 6: External source average
    ext_cols = [col for col in df.columns if 'EXT_SOURCE' in col]
    if ext_cols:
        df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    
    # Feature 7: Documents submitted count
    doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
    if doc_cols:
        df['DOCUMENTS_SUBMITTED'] = df[doc_cols].sum(axis=1)
    
    # Feature 8: Phone contact count
    phone_cols = [col for col in df.columns if 'FLAG_PHONE' in col or 'FLAG_CONT' in col]
    if phone_cols:
        df['PHONE_CONTACT_COUNT'] = df[phone_cols].sum(axis=1)
    
    # Feature 9: Age squared (non-linear relationship)
    df['AGE_YEARS_SQUARED'] = df['AGE_YEARS'] ** 2
    
    # Feature 10: Log of income (to handle skewness)
    df['LOG_INCOME'] = np.log1p(df['AMT_INCOME_TOTAL'])
    
    # Feature 11: Log of credit amount
    df['LOG_CREDIT'] = np.log1p(df['AMT_CREDIT'])
    
    # Feature 12: Payment burden (annuity as percentage of income)
    df['PAYMENT_BURDEN'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    
    print(f"✓ Created 12 new features. Total features: {df.shape[1]}")
    return df


def calculate_scale_pos_weight(df):
    """Calculate scale_pos_weight for handling class imbalance in LightGBM"""
    # scale_pos_weight = number of negative samples / number of positive samples
    negative_samples = (df['TARGET'] == 0).sum()
    positive_samples = (df['TARGET'] == 1).sum()
    scale_pos_weight = negative_samples / positive_samples
    
    print(f"✓ Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    print(f"  → Negative samples: {negative_samples:,}")
    print(f"  → Positive samples: {positive_samples:,}")
    
    return scale_pos_weight


def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print(f"Splitting data into train/test sets (test_size={test_size})...")
    
    # Separate features and target
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train set: {X_train.shape[0]:,} samples")
    print(f"✓ Test set: {X_test.shape[0]:,} samples")
    print(f"✓ Train default rate: {y_train.mean()*100:.2f}%")
    print(f"✓ Test default rate: {y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, label_encoders, scale_pos_weight):
    """Save processed data and metadata"""
    print("Saving processed data...")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed datasets
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    pd.Series(y_train).to_csv(output_dir / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(output_dir / "y_test.csv", index=False)
    
    # Save label encoders
    with open(output_dir / "label_encoders.pkl", 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save scale_pos_weight
    with open(output_dir / "scale_pos_weight.pkl", 'wb') as f:
        pickle.dump(scale_pos_weight, f)
    
    # Save feature names
    with open(output_dir / "feature_names.pkl", 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
    
    print(f"✓ Saved processed data to: {output_dir}")
    print("  → X_train.csv, X_test.csv")
    print("  → y_train.csv, y_test.csv")
    print("  → label_encoders.pkl")
    print("  → scale_pos_weight.pkl")
    print("  → feature_names.pkl")


def main():
    """Main preprocessing pipeline"""
    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load data
    df = load_data()
    print(f"✓ Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical columns
    df, label_encoders = encode_categorical_columns(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Calculate class imbalance weight
    scale_pos_weight = calculate_scale_pos_weight(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, label_encoders, scale_pos_weight)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python src/train.py")
    print("2. This will train the LightGBM model on the processed data")


if __name__ == "__main__":
    main()
