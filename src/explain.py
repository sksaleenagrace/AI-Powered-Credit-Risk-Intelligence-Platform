"""
SHAP Explanations for Credit Risk Model
Generates SHAP summary plots and individual prediction explanations
"""

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_model_and_data():
    """Load trained model and test data"""
    models_dir = Path(__file__).parent.parent / "models"
    data_dir = Path(__file__).parent.parent / "data"
    
    # Check if model exists
    if not (models_dir / "lgbm_model.pkl").exists():
        print("❌ ERROR: Trained model not found!")
        print("Please train the model first:")
        print("  python src/train.py")
        raise FileNotFoundError("Model not found. Run training first.")
    
    # Check if test data exists
    if not (data_dir / "X_test.csv").exists() or not (data_dir / "y_test.csv").exists():
        print("❌ ERROR: Test data files not found!")
        print("Please run preprocessing first:")
        print("  python src/preprocess.py")
        raise FileNotFoundError("Test data not found. Run preprocessing first.")
    
    # Load model
    with open(models_dir / "lgbm_model.pkl", 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Loaded trained model")
    
    # Load test data
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")
    print(f"✓ Loaded test data: {X_test.shape}")
    
    # Load feature names
    with open(data_dir / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)
    
    # Ensure X_test has the same columns as feature_names
    X_test = X_test[feature_names]
    
    return model, X_test, y_test


def generate_shap_summary_plot(model, X_test, output_dir):
    """Generate SHAP summary plot showing feature importance"""
    print("Generating SHAP summary plot...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample of test data (for performance)
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # If binary classification, shap_values is a list of two arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use values for positive class
    
    # Create summary plot
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Default Risk', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = output_dir / "shap_summary_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved SHAP summary plot to: {output_path}")
    
    return explainer


def generate_shap_feature_importance_plot(model, X_test, output_dir):
    """Generate SHAP feature importance bar plot"""
    print("Generating SHAP feature importance plot...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample
    sample_size = min(1000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for plotting
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP Value|', fontsize=14)
    plt.title('Top 20 Features by SHAP Importance', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    output_path = output_dir / "shap_feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved SHAP feature importance plot to: {output_path}")
    
    return feature_importance


def explain_individual_prediction(model, X_test, y_test, explainer, output_dir, sample_idx=0):
    """Explain an individual prediction with top 5 features"""
    print(f"Explaining individual prediction (sample index: {sample_idx})...")
    
    # Get the sample
    sample = X_test.iloc[[sample_idx]]
    actual_target = y_test.iloc[sample_idx]
    
    # Calculate SHAP values for this sample
    shap_values = explainer.shap_values(sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Get prediction
    prediction_proba = model.predict(sample, num_iteration=model.best_iteration)[0]
    risk_score = prediction_proba * 100
    
    if risk_score < 30:
        risk_band = "Low"
    elif risk_score < 60:
        risk_band = "Medium"
    else:
        risk_band = "High"
    
    # Get top 5 features
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'shap_value': shap_values[0]
    }).sort_values('shap_value', key=abs, ascending=False)
    
    top_5_features = feature_importance.head(5)
    
    # Print explanation
    print("\n" + "=" * 80)
    print("INDIVIDUAL PREDICTION EXPLANATION")
    print("=" * 80)
    print(f"\nActual Target: {'Default (1)' if actual_target == 1 else 'Non-Default (0)'}")
    print(f"Predicted Default Probability: {prediction_proba:.4f} ({risk_score:.2f}%)")
    print(f"Risk Band: {risk_band}")
    print(f"\nTop 5 Features Driving This Prediction:")
    print("-" * 80)
    
    for idx, row in top_5_features.iterrows():
        direction = "increases" if row['shap_value'] > 0 else "decreases"
        print(f"{row['feature']}:")
        print(f"  → SHAP Value: {row['shap_value']:.4f}")
        print(f"  → This feature {direction} default risk")
        print(f"  → Feature Value: {sample[row['feature']].values[0]:.4f}")
        print()
    
    # Create waterfall plot
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                         data=sample.values[0],
                                         feature_names=X_test.columns),
                       show=False)
    plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"shap_waterfall_sample_{sample_idx}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved waterfall plot to: {output_path}")
    
    # Create force plot
    force_plot = shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                 shap_values[0],
                                 sample.iloc[0],
                                 feature_names=X_test.columns,
                                 matplotlib=True,
                                 show=False)
    plt.title(f'SHAP Force Plot - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"shap_force_sample_{sample_idx}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved force plot to: {output_path}")
    
    return top_5_features


def explain_multiple_samples(model, X_test, y_test, explainer, output_dir, num_samples=5):
    """Explain multiple samples to show different scenarios"""
    print(f"\nExplaining {num_samples} different samples...")
    
    # Get samples: some defaults, some non-defaults
    default_indices = y_test[y_test == 1].index[:num_samples//2 + 1]
    non_default_indices = y_test[y_test == 0].index[:num_samples//2]
    
    all_indices = list(default_indices) + list(non_default_indices)
    
    explanations = []
    for idx in all_indices[:num_samples]:
        sample = X_test.loc[[idx]]
        actual_target = y_test.loc[idx]
        
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        prediction_proba = model.predict(sample, num_iteration=model.best_iteration)[0]
        
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'shap_value': shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        explanations.append({
            'index': idx,
            'actual_target': actual_target,
            'predicted_proba': prediction_proba,
            'top_features': feature_importance.head(5)['feature'].tolist()
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("MULTIPLE SAMPLE EXPLANATIONS SUMMARY")
    print("=" * 80)
    for exp in explanations:
        actual = "Default" if exp['actual_target'] == 1 else "Non-Default"
        print(f"\nSample {exp['index']}: {actual}")
        print(f"  Predicted Probability: {exp['predicted_proba']:.4f}")
        print(f"  Top 5 Features: {', '.join(exp['top_features'])}")
    
    return explanations


def save_explainer(explainer, output_dir):
    """Save SHAP explainer for later use"""
    explainer_path = output_dir / "shap_explainer.pkl"
    with open(explainer_path, 'wb') as f:
        pickle.dump(explainer, f)
    print(f"✓ Saved SHAP explainer to: {explainer_path}")


def main():
    """Main SHAP explanation pipeline"""
    print("=" * 80)
    print("SHAP EXPLANATION GENERATION")
    print("=" * 80)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Generate SHAP summary plot
    explainer = generate_shap_summary_plot(model, X_test, output_dir)
    
    # Generate feature importance plot
    feature_importance = generate_shap_feature_importance_plot(model, X_test, output_dir)
    
    # Explain individual predictions
    explain_individual_prediction(model, X_test, y_test, explainer, output_dir, sample_idx=0)
    
    # Explain multiple samples
    explain_multiple_samples(model, X_test, y_test, explainer, output_dir, num_samples=5)
    
    # Save explainer
    save_explainer(explainer, output_dir)
    
    print("\n" + "=" * 80)
    print("SHAP EXPLANATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  → shap_summary_plot.png")
    print("  → shap_feature_importance.png")
    print("  → shap_waterfall_sample_0.png")
    print("  → shap_force_sample_0.png")
    print("  → shap_explainer.pkl")
    print("\nNext steps:")
    print("1. Run: python src/chatbot.py")
    print("2. This will create the Talk-to-Data chatbot")


if __name__ == "__main__":
    main()
