"""
LightGBM Model Training for Credit Risk Prediction
Trains model, evaluates performance, and saves model with metrics
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_processed_data():
    data_dir = Path(__file__).parent.parent / "data"
    
    # Check if processed files exist
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "scale_pos_weight.pkl"]
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    
    if missing_files:
        print("❌ ERROR: Processed data files not found!")
        print(f"Missing files: {', '.join(missing_files)}")
        print("Please run preprocessing first:")
        print("  python src/preprocess.py")
        raise FileNotFoundError("Processed data files not found. Run preprocessing first.")
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")
    with open(data_dir / "scale_pos_weight.pkl", 'rb') as f:
        scale_pos_weight = pickle.load(f)
    
    print(f"✓ Loaded processed data")
    print(f"  → X_train: {X_train.shape}")
    print(f"  → X_test: {X_test.shape}")
    print(f"  → y_train: {y_train.shape}")
    print(f"  → y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scale_pos_weight


def calculate_risk_score(probability):
    """Convert probability to risk score (0-100%)"""
    return probability * 100


def calculate_risk_band(risk_score):
    """Convert risk score to risk band (Low/Medium/High)"""
    if risk_score < 30:
        return "Low"
    elif risk_score < 60:
        return "Medium"
    else:
        return "High"


def train_lightgbm(X_train, y_train, X_test, y_test, scale_pos_weight):
    """Train LightGBM model with optimized parameters"""
    print("Training LightGBM model...")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'scale_pos_weight': scale_pos_weight,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1
    }
    
    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print(f"✓ Model trained successfully")
    print(f"  → Best iteration: {model.best_iteration}")
    print(f"  → Number of trees: {model.num_trees()}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and calculate metrics"""
    print("Evaluating model performance...")
    
    # Get predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp)
    
    print(f"✓ Model Evaluation Metrics:")
    print(f"  → AUC-ROC: {auc_roc:.4f}")
    print(f"  → Accuracy: {accuracy:.4f}")
    print(f"  → Precision: {precision:.4f}")
    print(f"  → Recall: {recall:.4f}")
    print(f"  → F1-Score: {f1:.4f}")
    print(f"  → Specificity: {specificity:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")
    
    # Calculate risk score distribution
    risk_scores = calculate_risk_score(y_pred_proba)
    print(f"\n  Risk Score Distribution:")
    print(f"    Mean: {risk_scores.mean():.2f}%")
    print(f"    Std: {risk_scores.std():.2f}%")
    print(f"    Min: {risk_scores.min():.2f}%")
    print(f"    Max: {risk_scores.max():.2f}%")
    
    # Calculate risk band distribution
    risk_bands = [calculate_risk_band(score) for score in risk_scores]
    low_risk = sum(1 for band in risk_bands if band == "Low")
    medium_risk = sum(1 for band in risk_bands if band == "Medium")
    high_risk = sum(1 for band in risk_bands if band == "High")
    print(f"\n  Risk Band Distribution:")
    print(f"    Low Risk: {low_risk:,} ({low_risk/len(risk_bands)*100:.1f}%)")
    print(f"    Medium Risk: {medium_risk:,} ({medium_risk/len(risk_bands)*100:.1f}%)")
    print(f"    High Risk: {high_risk:,} ({high_risk/len(risk_bands)*100:.1f}%)")
    
    metrics = {
        'auc_roc': float(auc_roc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'risk_score_distribution': {
            'mean': float(risk_scores.mean()),
            'std': float(risk_scores.std()),
            'min': float(risk_scores.min()),
            'max': float(risk_scores.max())
        },
        'risk_band_distribution': {
            'low': int(low_risk),
            'medium': int(medium_risk),
            'high': int(high_risk)
        }
    }
    
    return metrics


def save_model_and_metrics(model, metrics):
    """Save trained model and evaluation metrics"""
    print("Saving model and metrics...")
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / "lgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model to: {model_path}")
    
    # Save metrics
    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    feature_importance_path = models_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"✓ Saved feature importance to: {feature_importance_path}")
    
    print(f"\n  Top 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.2f}")


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("LIGHTGBM MODEL TRAINING")
    print("=" * 80)
    
    # Load processed data
    X_train, X_test, y_train, y_test, scale_pos_weight = load_processed_data()
    
    # Train model
    model = train_lightgbm(X_train, y_train, X_test, y_test, scale_pos_weight)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and metrics
    save_model_and_metrics(model, metrics)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python src/explain.py")
    print("2. This will generate SHAP explanations for the model")


# Fix the typo in the function call
def calculate_risk_band(risk_score):
    """Convert risk score to risk band (Low/Medium/High)"""
    if risk_score < 30:
        return "Low"
    elif risk_score < 60:
        return "Medium"
    else:
        return "High"


if __name__ == "__main__":
    main()
