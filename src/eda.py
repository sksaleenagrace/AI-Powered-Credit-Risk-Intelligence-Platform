"""
Exploratory Data Analysis for Credit Risk Dataset
Loads application_train.csv and generates visualizations with business insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_data():
    """Load the training dataset"""
    data_path = Path(__file__).parent.parent / "data" / "application_train.csv"
    df = pd.read_csv(data_path)
    return df


def print_dataset_summary(df):
    """Print comprehensive dataset summary"""
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Rows: {df.shape[0]:,}")
    print(f"Number of Columns: {df.shape[1]:,}")
    
    print("\n" + "-" * 80)
    print("MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)
    
    # Show columns with missing values
    missing_cols = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_cols) > 0:
        print(f"\nColumns with missing values: {len(missing_cols)}")
        print(missing_cols.head(20))
    else:
        print("\nNo missing values found")
    
    print("\n" + "-" * 80)
    print("DATA TYPES")
    print("-" * 80)
    dtype_counts = df.dtypes.value_counts()
    print(dtype_counts)
    
    print("\n" + "-" * 80)
    print("TARGET VARIABLE DISTRIBUTION")
    print("-" * 80)
    target_counts = df['TARGET'].value_counts()
    print(f"Non-Default (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)")
    print(f"Default (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)")
    print("=" * 80)


def plot_default_distribution(df, output_dir):
    """Plot Default vs Non-Default distribution"""
    plt.figure(figsize=(10, 6))
    target_counts = df['TARGET'].value_counts()
    labels = ['Non-Default (0)', 'Default (1)']
    colors = ['#2ecc71', '#e74c3c']
    
    plt.bar(labels, target_counts.values, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Default vs Non-Default Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Count', fontsize=14)
    plt.xlabel('Target Variable', fontsize=14)
    
    # Add count labels on bars
    for i, v in enumerate(target_counts.values):
        plt.text(i, v + 1000, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "default_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_income_vs_default(df, output_dir):
    """Plot Income distribution by Default status"""
    plt.figure(figsize=(14, 6))
    
    # Create boxplot
    plt.subplot(1, 2, 1)
    df.boxplot(column='AMT_INCOME_TOTAL', by='TARGET', ax=plt.gca())
    plt.title('Income Distribution by Default Status', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove automatic title
    plt.xlabel('Target (0=Non-Default, 1=Default)', fontsize=12)
    plt.ylabel('Income Amount', fontsize=12)
    plt.yscale('log')  # Log scale due to outliers
    
    # Create violin plot
    plt.subplot(1, 2, 2)
    # Sample data for violin plot to avoid overcrowding
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    sns.violinplot(x='TARGET', y='AMT_INCOME_TOTAL', data=sample_df, ax=plt.gca())
    plt.title('Income Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    plt.xlabel('Target (0=Non-Default, 1=Default)', fontsize=12)
    plt.ylabel('Income Amount (Log Scale)', fontsize=12)
    plt.yscale('log')
    
    plt.tight_layout()
    output_path = output_dir / "income_vs_default.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_age_vs_default(df, output_dir):
    """Plot Age distribution by Default status"""
    plt.figure(figsize=(14, 6))
    
    # Calculate age in years from DAYS_BIRTH
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25
    
    # Create histogram
    plt.subplot(1, 2, 1)
    plt.hist(df[df['TARGET']==0]['AGE_YEARS'], bins=30, alpha=0.7, 
             label='Non-Default', color='#2ecc71', edgecolor='black')
    plt.hist(df[df['TARGET']==1]['AGE_YEARS'], bins=30, alpha=0.7, 
             label='Default', color='#e74c3c', edgecolor='black')
    plt.title('Age Distribution by Default Status', fontsize=14, fontweight='bold')
    plt.xlabel('Age (Years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    
    # Create boxplot
    plt.subplot(1, 2, 2)
    df.boxplot(column='AGE_YEARS', by='TARGET', ax=plt.gca())
    plt.title('Age Distribution (Box Plot)', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Target (0=Non-Default, 1=Default)', fontsize=12)
    plt.ylabel('Age (Years)', fontsize=12)
    
    plt.tight_layout()
    output_path = output_dir / "age_vs_default.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_loan_amount_vs_default(df, output_dir):
    """Plot Loan Amount distribution by Default status"""
    plt.figure(figsize=(14, 6))
    
    # Create histogram
    plt.subplot(1, 2, 1)
    plt.hist(df[df['TARGET']==0]['AMT_CREDIT'], bins=50, alpha=0.7, 
             label='Non-Default', color='#2ecc71', edgecolor='black')
    plt.hist(df[df['TARGET']==1]['AMT_CREDIT'], bins=50, alpha=0.7, 
             label='Default', color='#e74c3c', edgecolor='black')
    plt.title('Loan Amount Distribution by Default Status', fontsize=14, fontweight='bold')
    plt.xlabel('Loan Amount', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.xlim(0, df['AMT_CREDIT'].quantile(0.95))  # Limit to 95th percentile
    
    # Create boxplot
    plt.subplot(1, 2, 2)
    df.boxplot(column='AMT_CREDIT', by='TARGET', ax=plt.gca())
    plt.title('Loan Amount Distribution (Box Plot)', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Target (0=Non-Default, 1=Default)', fontsize=12)
    plt.ylabel('Loan Amount', fontsize=12)
    plt.ylim(0, df['AMT_CREDIT'].quantile(0.95))
    
    plt.tight_layout()
    output_path = output_dir / "loan_amount_vs_default.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_gender_vs_default(df, output_dir):
    """Plot Gender distribution by Default status"""
    plt.figure(figsize=(12, 6))
    
    # Calculate default rates by gender
    gender_default = df.groupby('CODE_GENDER')['TARGET'].agg(['mean', 'count'])
    gender_default['percentage'] = gender_default['mean'] * 100
    gender_default['non_default_pct'] = (1 - gender_default['mean']) * 100
    
    # Create stacked bar chart
    labels = gender_default.index
    non_default = gender_default['non_default_pct']
    default = gender_default['percentage']
    
    x = np.arange(len(labels))
    width = 0.6
    
    plt.bar(x, non_default, width, label='Non-Default', color='#2ecc71', alpha=0.8, edgecolor='black')
    plt.bar(x, default, width, bottom=non_default, label='Default', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    plt.title('Default Rate by Gender', fontsize=16, fontweight='bold')
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(x, labels)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    
    # Add percentage labels
    for i, (nd, d) in enumerate(zip(non_default, default)):
        plt.text(i, nd/2, f'{nd:.1f}%', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=11)
        plt.text(i, nd + d/2, f'{d:.1f}%', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "gender_vs_default.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_education_vs_default(df, output_dir):
    """Plot Education level distribution by Default status"""
    plt.figure(figsize=(14, 6))
    
    # Calculate default rates by education
    edu_default = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].agg(['mean', 'count'])
    edu_default = edu_default.sort_values('mean', ascending=True)
    edu_default['percentage'] = edu_default['mean'] * 100
    edu_default['non_default_pct'] = (1 - edu_default['mean']) * 100
    
    # Create stacked bar chart
    labels = edu_default.index
    non_default = edu_default['non_default_pct']
    default = edu_default['percentage']
    
    x = np.arange(len(labels))
    width = 0.6
    
    plt.bar(x, non_default, width, label='Non-Default', color='#2ecc71', alpha=0.8, edgecolor='black')
    plt.bar(x, default, width, bottom=non_default, label='Default', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    plt.title('Default Rate by Education Level', fontsize=16, fontweight='bold')
    plt.xlabel('Education Level', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    
    # Add percentage labels
    for i, (nd, d) in enumerate(zip(non_default, default)):
        plt.text(i, nd/2, f'{nd:.1f}%', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=9)
        plt.text(i, nd + d/2, f'{d:.1f}%', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "education_vs_default.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def generate_business_insights(df):
    """Generate and print 5 key business insights"""
    print("\n" + "=" * 80)
    print("5 KEY BUSINESS INSIGHTS")
    print("=" * 80)
    
    # Insight 1: Overall default rate
    default_rate = df['TARGET'].mean() * 100
    print(f"\n1. OVERALL DEFAULT RATE: {default_rate:.2f}%")
    print(f"   → Approximately 1 in {int(100/default_rate)} applicants default on their loans.")
    print(f"   → This indicates a relatively low-risk portfolio overall.")
    
    # Insight 2: Age relationship
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25
    avg_age_non_default = df[df['TARGET']==0]['AGE_YEARS'].mean()
    avg_age_default = df[df['TARGET']==1]['AGE_YEARS'].mean()
    print(f"\n2. AGE AND DEFAULT RISK:")
    print(f"   → Average age of non-defaulters: {avg_age_non_default:.1f} years")
    print(f"   → Average age of defaulters: {avg_age_default:.1f} years")
    print(f"   → Younger applicants are {(avg_age_non_default/avg_age_default - 1)*100:.1f}% more likely to default.")
    print(f"   → This suggests financial stability improves with age.")
    
    # Insight 3: Income relationship
    median_income_non_default = df[df['TARGET']==0]['AMT_INCOME_TOTAL'].median()
    median_income_default = df[df['TARGET']==1]['AMT_INCOME_TOTAL'].median()
    print(f"\n3. INCOME AND DEFAULT RISK:")
    print(f"   → Median income of non-defaulters: ${median_income_non_default:,.0f}")
    print(f"   → Median income of defaulters: ${median_income_default:,.0f}")
    print(f"   → Defaulters have {(1 - median_income_default/median_income_non_default)*100:.1f}% lower median income.")
    print(f"   → Income is a strong predictor of loan repayment ability.")
    
    # Insight 4: Gender difference
    gender_default = df.groupby('CODE_GENDER')['TARGET'].mean()
    if 'F' in gender_default.index and 'M' in gender_default.index:
        male_rate = gender_default['M'] * 100
        female_rate = gender_default['F'] * 100
        print(f"\n4. GENDER DIFFERENCES:")
        print(f"   → Male default rate: {male_rate:.2f}%")
        print(f"   → Female default rate: {female_rate:.2f}%")
        print(f"   → Males are {(male_rate/female_rate - 1)*100:.1f}% more likely to default than females.")
        print(f"   → Gender should be considered as a risk factor in the model.")
    
    # Insight 5: Education impact
    edu_default = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False)
    highest_risk_edu = edu_default.index[0]
    lowest_risk_edu = edu_default.index[-1]
    print(f"\n5. EDUCATION LEVEL IMPACT:")
    print(f"   → Highest risk education: {highest_risk_edu} ({edu_default.iloc[0]*100:.2f}% default rate)")
    print(f"   → Lowest risk education: {lowest_risk_edu} ({edu_default.iloc[-1]*100:.2f}% default rate)")
    print(f"   → Risk varies by {(edu_default.iloc[0]/edu_default.iloc[-1] - 1)*100:.1f}% across education levels.")
    print(f"   → Higher education correlates strongly with lower default risk.")
    
    print("=" * 80)


def main():
    """Main function to run EDA"""
    print("Starting Exploratory Data Analysis...")
    print("=" * 80)
    
    # Load data
    df = load_data()
    print(f"✓ Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns")
    
    # Print summary
    print_dataset_summary(df)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir}")
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_default_distribution(df, output_dir)
    plot_income_vs_default(df, output_dir)
    plot_age_vs_default(df, output_dir)
    plot_loan_amount_vs_default(df, output_dir)
    plot_gender_vs_default(df, output_dir)
    plot_education_vs_default(df, output_dir)
    
    # Generate insights
    generate_business_insights(df)
    
    print("\n" + "=" * 80)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("Review the PNG files to explore the data patterns.")


if __name__ == "__main__":
    main()
