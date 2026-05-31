"""
Create simple sample dataset for demo mode
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    np.random.seed(42)
    
    # Create sample data with 1000 rows - very simple version
    n_samples = 1000
    
    # Core columns only - most important for demo
    data = {
        'SK_ID_CURR': range(100001, 100001 + n_samples),
        'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], size=n_samples),
        'CODE_GENDER': np.random.choice(['M', 'F'], size=n_samples),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], size=n_samples),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], size=n_samples),
        'CNT_CHILDREN': np.random.randint(0, 6, size=n_samples),
        'AMT_INCOME_TOTAL': np.random.uniform(50000, 500000, size=n_samples),
        'AMT_CREDIT': np.random.uniform(200000, 1000000, size=n_samples),
        'AMT_ANNUITY': np.random.uniform(10000, 100000, size=n_samples),
        'AMT_GOODS_PRICE': np.random.uniform(200000, 1000000, size=n_samples),
        'NAME_INCOME_TYPE': np.random.choice(['Working', 'State servant', 'Commercial associate', 'Pensioner'], size=n_samples),
        'NAME_EDUCATION_TYPE': np.random.choice(['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'], size=n_samples),
        'NAME_FAMILY_STATUS': np.random.choice(['Single / not married', 'Married', 'Civil marriage', 'Widow', 'Separated'], size=n_samples),
        'NAME_HOUSING_TYPE': np.random.choice(['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment'], size=n_samples),
        'REGION_POPULATION_RELATIVE': np.random.uniform(0.01, 0.1, size=n_samples),
        'DAYS_BIRTH': np.random.randint(-25000, -7000, size=n_samples),
        'DAYS_EMPLOYED': np.random.randint(-15000, -100, size=n_samples),
        'DAYS_REGISTRATION': np.random.randint(-18000, -100, size=n_samples),
        'DAYS_ID_PUBLISH': np.random.randint(-7000, -100, size=n_samples),
        'FLAG_MOBIL': np.random.choice([1, 0], size=n_samples, p=[0.999, 0.001]),
        'FLAG_EMP_PHONE': np.random.choice([1, 0], size=n_samples),
        'FLAG_WORK_PHONE': np.random.choice([1, 0], size=n_samples),
        'FLAG_CONT_MOBILE': np.random.choice([1, 0], size=n_samples),
        'FLAG_PHONE': np.random.choice([1, 0], size=n_samples),
        'FLAG_EMAIL': np.random.choice([1, 0], size=n_samples),
        'OCCUPATION_TYPE': np.random.choice(['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff'], size=n_samples),
        'CNT_FAM_MEMBERS': np.random.randint(1, 7, size=n_samples),
        'REGION_RATING_CLIENT': np.random.choice([1, 2, 3], size=n_samples),
        'REGION_RATING_CLIENT_W_CITY': np.random.choice([1, 2, 3], size=n_samples),
        'WEEKDAY_APPR_PROCESS_START': np.random.choice(['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'], size=n_samples),
        'HOUR_APPR_PROCESS_START': np.random.randint(0, 24, size=n_samples),
        'REG_REGION_NOT_LIVE_REGION': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'REG_REGION_NOT_WORK_REGION': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'LIVE_REGION_NOT_WORK_REGION': np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
        'REG_CITY_NOT_LIVE_CITY': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'REG_CITY_NOT_WORK_CITY': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'LIVE_CITY_NOT_WORK_CITY': np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
        'EXT_SOURCE_1': np.random.uniform(0.1, 0.9, size=n_samples),
        'EXT_SOURCE_2': np.random.uniform(0.1, 0.9, size=n_samples),
        'EXT_SOURCE_3': np.random.uniform(0.1, 0.9, size=n_samples),
        'APARTMENTS_AVG': np.random.uniform(0.01, 0.3, size=n_samples),
        'BASEMENTAREA_AVG': np.random.uniform(0.01, 0.2, size=n_samples),
        'YEARS_BEGINEXPLUATATION_AVG': np.random.uniform(0.5, 1.0, size=n_samples),
        'YEARS_BUILD_AVG': np.random.uniform(0.3, 1.0, size=n_samples),
        'COMMONAREA_AVG': np.random.uniform(0.01, 0.2, size=n_samples),
        'ELEVATORS_AVG': np.random.uniform(0.01, 0.3, size=n_samples),
        'ENTRANCES_AVG': np.random.uniform(0.01, 0.2, size=n_samples),
        'FLOORSMAX_AVG': np.random.uniform(0.01, 0.3, size=n_samples),
        'LANDAREA_AVG': np.random.uniform(0.01, 0.2, size=n_samples),
        'LIVINGAPARTMENTS_AVG': np.random.uniform(0.01, 0.2, size=n_samples),
        'NONLIVINGAPARTMENTS_AVG': np.random.uniform(0.01, 0.1, size=n_samples),
        'NONLIVINGAREA_AVG': np.random.uniform(0.01, 0.1, size=n_samples),
        'APARTMENTS_MODE': np.random.uniform(0.01, 0.3, size=n_samples),
        'BASEMENTAREA_MODE': np.random.uniform(0.01, 0.2, size=n_samples),
        'YEARS_BEGINEXPLUATATION_MODE': np.random.uniform(0.5, 1.0, size=n_samples),
        'YEARS_BUILD_MODE': np.random.uniform(0.3, 1.0, size=n_samples),
        'COMMONAREA_MODE': np.random.uniform(0.01, 0.2, size=n_samples),
        'ELEVATORS_MODE': np.random.uniform(0.01, 0.3, size=n_samples),
        'ENTRANCES_MODE': np.random.uniform(0.01, 0.2, size=n_samples),
        'FLOORSMAX_MODE': np.random.uniform(0.01, 0.3, size=n_samples),
        'LANDAREA_MODE': np.random.uniform(0.01, 0.2, size=n_samples),
        'LIVINGAPARTMENTS_MODE': np.random.uniform(0.01, 0.2, size=n_samples),
        'NONLIVINGAPARTMENTS_MODE': np.random.uniform(0.01, 0.1, size=n_samples),
        'NONLIVINGAREA_MODE': np.random.uniform(0.01, 0.1, size=n_samples),
        'FONDKAPREMONT_MODE': np.random.choice(['reg oper account', 'reg oper spec account', 'not specified', 'cash'], size=n_samples),
        'HOUSETYPE_MODE': np.random.choice(['block of flats', 'terraced house', 'specific housing'], size=n_samples),
        'TOTALAREA_MODE': np.random.uniform(0.05, 0.3, size=n_samples),
        'WALLSMATERIAL_MODE': np.random.choice(['Panel', 'Stone, brick', 'Wooden', 'Block', 'Mixed', 'Others'], size=n_samples),
        'EMERGENCYSTATE_MODE': np.random.choice(['No', 'Yes'], size=n_samples, p=[0.95, 0.05]),
        'OBS_30_CNT_SOCIAL_CIRCLE': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.8, 0.15, 0.04, 0.01]),
        'DEF_30_CNT_SOCIAL_CIRCLE': np.random.choice([0, 1, 2], size=n_samples, p=[0.9, 0.08, 0.02]),
        'OBS_60_CNT_SOCIAL_CIRCLE': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.8, 0.15, 0.04, 0.01]),
        'DEF_60_CNT_SOCIAL_CIRCLE': np.random.choice([0, 1, 2], size=n_samples, p=[0.9, 0.08, 0.02]),
        'DAYS_LAST_PHONE_CHANGE': np.random.randint(-4000, -100, size=n_samples),
        'FLAG_DOCUMENT_2': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'FLAG_DOCUMENT_3': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'FLAG_DOCUMENT_4': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'FLAG_DOCUMENT_5': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_6': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'FLAG_DOCUMENT_7': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_8': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_9': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_10': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_11': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_12': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_13': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_14': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_15': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_16': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_17': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_18': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_19': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_20': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'FLAG_DOCUMENT_21': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'AMT_REQ_CREDIT_BUREAU_HOUR': np.random.choice([0, 1, 2], size=n_samples, p=[0.95, 0.04, 0.01]),
        'AMT_REQ_CREDIT_BUREAU_DAY': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.9, 0.07, 0.02, 0.01]),
        'AMT_REQ_CREDIT_BUREAU_WEEK': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.85, 0.1, 0.04, 0.01]),
        'AMT_REQ_CREDIT_BUREAU_MON': np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.7, 0.2, 0.07, 0.02, 0.01]),
        'AMT_REQ_CREDIT_BUREAU_QRT': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.8, 0.15, 0.04, 0.01]),
        'AMT_REQ_CREDIT_BUREAU_YEAR': np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.6, 0.25, 0.1, 0.03, 0.015, 0.005]),
    }
    
    df = pd.DataFrame(data)
    
    # Save to data folder
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    df.to_csv(data_dir / "sample_data.csv", index=False)
    print(f"Sample data created with {len(df)} rows")
    print(f"Default rate: {df['TARGET'].mean() * 100:.2f}%")
    print(f"Saved to: {data_dir / 'sample_data.csv'}")

if __name__ == "__main__":
    create_sample_data()
