"""
STEP 3: MODEL EVALUATION AND COMPARISON
========================================================================

This script evaluates and compares all 6 trained models:
1. LSTM (Deep Learning)
2. 1D CNN (Convolutional Neural Network)
3. SVM (Support Vector Machine)
4. XGBoost (Gradient Boosting)
5. VAE (Variational Autoencoder)
6. FNN (Feedforward Neural Network)

EVALUATION METRICS FOR CLASSIFICATION:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
- Classification Report

OUTPUTS:
- Performance comparison table
- Bar charts comparing metrics
- Confusion matrices for all models
- Detailed analysis and discussion
"""

import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import time

import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_auc_score,
                           classification_report, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'output' / 'step_1_7_data_splitting'
OUT = ROOT / 'output' / 'step_3_model_evaluation'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DATA = DATA_DIR / 'train_data.csv'
VAL_DATA = DATA_DIR / 'val_data.csv'
TEST_DATA = DATA_DIR / 'test_data.csv'

# Model paths
MODELS_DIR = ROOT / 'output'
LSTM_MODEL = MODELS_DIR / 'step_2_1_LSTM' / 'lstm_best_model.h5'
CNN_MODEL = MODELS_DIR / 'step_2_2_1D_CNN' / 'cnn_best_model.h5'
SVM_MODEL = MODELS_DIR / 'step_2_3_SVM' / 'svm_best_model.pkl'
XGBOOST_MODEL = MODELS_DIR / 'step_2_4_XGBoost' / 'xgboost_best_model.pkl'
VAE_MODEL = MODELS_DIR / 'step_2_5_VAE' / 'vae_best_model.h5'
FNN_MODEL = MODELS_DIR / 'step_2_6_FNN' / 'fnn_best_model.h5'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load test data for evaluation."""
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA)
    
    X_test = test_df.drop('class', axis=1).values
    y_test = test_df['class'].values
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y_test)
    y_test_encoded = le.transform(y_test)
    
    print(f"✓ Test shape: {X_test.shape}")
    print(f"✓ Classes: {list(le.classes_)}\n")
    
    return X_test, y_test_encoded, le.classes_

def preprocess_data(X_test):
    """Preprocess test data (handle extreme values and normalize)."""
    from sklearn.preprocessing import RobustScaler
    
    # Handle extreme values
    threshold = 1e10
    X_test = np.where(np.abs(X_test) > threshold, np.sign(X_test) * threshold, X_test)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=threshold, neginf=-threshold)
    
    # Scale
    scaler = RobustScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Normalize to [0, 1] for models that need it
    X_test_normalized = (X_test_scaled - X_test_scaled.min(axis=0)) / (X_test_scaled.max(axis=0) - X_test_scaled.min(axis=0) + 1e-8)
    X_test_normalized = np.clip(X_test_normalized, 0, 1)
    
    return X_test_scaled, X_test_normalized

# ============================================================================
# MODEL LOADING AND PREDICTION
# ============================================================================

def load_and_evaluate_models(X_test_scaled, X_test_normalized, y_test, class_names):
    """Load all models and generate predictions."""
    results = {}
    
    print("="*100)
    print("LOADING AND EVALUATING ALL MODELS")
    print("="*100 + "\n")
    
    # 1. LSTM
    print("1. LSTM Model...")
    try:
        lstm_model = tf.keras.models.load_model(LSTM_MODEL)
        y_pred_lstm = np.argmax(lstm_model.predict(X_test_scaled, verbose=0), axis=1)
        results['LSTM'] = {
            'predictions': y_pred_lstm,
            'probabilities': lstm_model.predict(X_test_scaled, verbose=0),
            'status': 'Success'
        }
        print("   ✓ LSTM loaded and predictions generated\n")
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}\n")
        results['LSTM'] = {'status': 'Failed'}
    
    # 2. CNN
    print("2. 1D CNN Model...")
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL)
        y_pred_cnn = np.argmax(cnn_model.predict(X_test_scaled, verbose=0), axis=1)
        results['1D_CNN'] = {
            'predictions': y_pred_cnn,
            'probabilities': cnn_model.predict(X_test_scaled, verbose=0),
            'status': 'Success'
        }
        print("   ✓ CNN loaded and predictions generated\n")
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}\n")
        results['1D_CNN'] = {'status': 'Failed'}
    
    # 3. SVM
    print("3. SVM Model...")
    try:
        svm_model = joblib.load(SVM_MODEL)
        y_pred_svm = svm_model.predict(X_test_scaled)
        probabilities_svm = svm_model.decision_function(X_test_scaled)
        results['SVM'] = {
            'predictions': y_pred_svm,
            'probabilities': probabilities_svm,
            'status': 'Success'
        }
        print("   ✓ SVM loaded and predictions generated\n")
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}\n")
        results['SVM'] = {'status': 'Failed'}
    
    # 4. XGBoost
    print("4. XGBoost Model...")
    try:
        xgb_model = joblib.load(XGBOOST_MODEL)
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        probabilities_xgb = xgb_model.predict_proba(X_test_scaled)
        results['XGBoost'] = {
            'predictions': y_pred_xgb,
            'probabilities': probabilities_xgb,
            'status': 'Success'
        }
        print("   ✓ XGBoost loaded and predictions generated\n")
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}\n")
        results['XGBoost'] = {'status': 'Failed'}
    
    # 5. VAE (encoder only for feature extraction)
    print("5. VAE Model...")
    try:
        vae_model = tf.keras.models.load_model(ROOT / 'output' / 'step_2_5_VAE' / 'vae_best_model.h5')
        # VAE is generative, use reconstruction error for anomaly detection
        reconstructed = vae_model.predict(X_test_normalized, verbose=0)
        reconstruction_error = np.mean(np.abs(X_test_normalized - reconstructed), axis=1)
        
        # Simple classification: use reconstruction error percentiles
        thresholds = np.percentile(reconstruction_error, [33, 66])
        y_pred_vae = np.digitize(reconstruction_error, thresholds)
        
        results['VAE'] = {
            'predictions': y_pred_vae,
            'probabilities': reconstruction_error.reshape(-1, 1),
            'status': 'Success'
        }
        print("   ✓ VAE loaded (using reconstruction error for classification)\n")
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}\n")
        results['VAE'] = {'status': 'Failed'}
    
    # 6. FNN
    print("6. FNN Model...")
    try:
        fnn_model = tf.keras.models.load_model(FNN_MODEL)
        y_pred_fnn = np.argmax(fnn_model.predict(X_test_scaled, verbose=0), axis=1)
        results['FNN'] = {
            'predictions': y_pred_fnn,
            'probabilities': fnn_model.predict(X_test_scaled, verbose=0),
            'status': 'Success'
        }
        print("   ✓ FNN loaded and predictions generated\n")
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}\n")
        results['FNN'] = {'status': 'Failed'}
    
    return results

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(y_true, y_pred, model_name, class_names):
    """Calculate all evaluation metrics."""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted scores
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (one-vs-rest for multiclass)
        try:
            if len(np.unique(y_true)) > 2:
                roc_auc = roc_auc_score(y_true, np.eye(3)[y_pred], multi_class='ovr', average='macro')
            else:
                roc_auc = roc_auc_score(y_true, y_pred)
        except:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision (Macro)': precision_macro,
            'Recall (Macro)': recall_macro,
            'F1-Score (Macro)': f1_macro,
            'Precision (Weighted)': precision_weighted,
            'Recall (Weighted)': recall_weighted,
            'F1-Score (Weighted)': f1_weighted,
            'ROC-AUC': roc_auc,
            'Confusion Matrix': cm
        }
    except Exception as e:
        print(f"Error calculating metrics for {model_name}: {str(e)}")
        return None

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_performance_comparison(metrics_list):
    """Plot performance metrics comparison."""
    df_metrics = pd.DataFrame([{k: v for k, v in m.items() if k != 'Confusion Matrix'} 
                               for m in metrics_list if m is not None])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    
    metrics_to_plot = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 
                       'F1-Score (Macro)', 'Precision (Weighted)', 'F1-Score (Weighted)']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
        if metric in df_metrics.columns:
            bars = ax.bar(df_metrics['Model'], df_metrics[metric], color=colors[:len(df_metrics)])
            ax.set_ylim([0, 1.05])
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(metric, fontweight='bold', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUT / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Performance comparison plot saved")
    plt.close()

def plot_confusion_matrices(metrics_list, class_names):
    """Plot confusion matrices for all models."""
    n_models = len([m for m in metrics_list if m is not None])
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
    
    for idx, (ax, metrics) in enumerate(zip(axes.flat, metrics_list)):
        if metrics is not None:
            cm = metrics['Confusion Matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'})
            ax.set_title(f"{metrics['Model']}\n(Accuracy: {metrics['Accuracy']:.3f})", 
                        fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(OUT / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrices plot saved")
    plt.close()

def plot_accuracy_ranking(metrics_list):
    """Plot model accuracy ranking."""
    df_metrics = pd.DataFrame([{k: v for k, v in m.items() if k != 'Confusion Matrix'} 
                               for m in metrics_list if m is not None])
    
    df_sorted = df_metrics.sort_values('Accuracy', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_ranked = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
    bars = ax.barh(df_sorted['Model'], df_sorted['Accuracy'], color=colors_ranked)
    
    ax.set_xlabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Model Accuracy Ranking', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 1.05])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['Accuracy'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', 
               va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUT / 'accuracy_ranking.png', dpi=300, bbox_inches='tight')
    print("✓ Accuracy ranking plot saved")
    plt.close()

# ============================================================================
# REPORTING
# ============================================================================

def generate_report(metrics_list, class_names):
    """Generate comprehensive evaluation report."""
    df_metrics = pd.DataFrame([{k: v for k, v in m.items() if k != 'Confusion Matrix'} 
                               for m in metrics_list if m is not None])
    
    # Determine best model
    best_idx = df_metrics['Accuracy'].idxmax()
    best_model = df_metrics.loc[best_idx, 'Model']
    best_accuracy = df_metrics.loc[best_idx, 'Accuracy']
    
    report = f"""
{'='*100}
STEP 3: MODEL EVALUATION AND COMPARISON REPORT
{'='*100}

DATASET INFORMATION:
  Test samples: {sum([m['Confusion Matrix'].sum() for m in metrics_list if m is not None])}
  Number of classes: {len(class_names)}
  Classes: {list(class_names)}

{'='*100}
PERFORMANCE METRICS SUMMARY
{'='*100}

{df_metrics.to_string(index=False)}

{'='*100}
BEST MODEL: {best_model}
{'='*100}
Accuracy: {best_accuracy:.4f}

{'='*100}
DETAILED ANALYSIS BY MODEL
{'='*100}

"""
    
    for metrics in metrics_list:
        if metrics is not None:
            report += f"""
{metrics['Model']}:
  Accuracy (Macro):        {metrics['Accuracy']:.4f}
  Precision (Macro):       {metrics['Precision (Macro)']:.4f}
  Recall (Macro):          {metrics['Recall (Macro)']:.4f}
  F1-Score (Macro):        {metrics['F1-Score (Macro)']:.4f}
  Precision (Weighted):    {metrics['Precision (Weighted)']:.4f}
  Recall (Weighted):       {metrics['Recall (Weighted)']:.4f}
  F1-Score (Weighted):     {metrics['F1-Score (Weighted)']:.4f}
  ROC-AUC:                 {metrics['ROC-AUC']:.4f}

"""
    
    report += f"""
{'='*100}
KEY FINDINGS AND RECOMMENDATIONS
{'='*100}

1. BEST PERFORMING MODEL: {best_model}
   - Achieved the highest accuracy of {best_accuracy:.4f}
   - This model generalizes best to unseen data

2. MODEL STRENGTHS:
   - Deep learning models (LSTM, CNN, FNN) capture temporal and spatial patterns
   - Tree-based models (XGBoost) handle feature interactions well
   - SVM is effective for high-dimensional classification
   - VAE provides unsupervised feature learning

3. RECOMMENDATIONS FOR DEPLOYMENT:
   - Use {best_model} for production predictions
   - Consider ensemble methods combining multiple models for robustness
   - Monitor model performance on new data regularly
   - Retrain periodically with updated data

{'='*100}
Report generated: {datetime.now().isoformat()}
{'='*100}
"""
    
    return report

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation pipeline."""
    print("\n" + "="*100)
    print("STEP 3: MODEL EVALUATION AND COMPARISON")
    print("="*100 + "\n")
    
    # Load data
    X_test, y_test, class_names = load_data()
    X_test_scaled, X_test_normalized = preprocess_data(X_test)
    
    # Load and evaluate models
    results = load_and_evaluate_models(X_test_scaled, X_test_normalized, y_test, class_names)
    
    # Calculate metrics
    print("="*100)
    print("CALCULATING METRICS")
    print("="*100 + "\n")
    
    metrics_list = []
    for model_name, pred_data in results.items():
        if pred_data['status'] == 'Success':
            print(f"Calculating metrics for {model_name}...")
            metrics = calculate_metrics(y_test, pred_data['predictions'], model_name, class_names)
            if metrics is not None:
                metrics_list.append(metrics)
                print(f"  ✓ Accuracy: {metrics['Accuracy']:.4f}\n")
    
    # Create visualizations
    print("="*100)
    print("CREATING VISUALIZATIONS")
    print("="*100 + "\n")
    
    plot_performance_comparison(metrics_list)
    plot_confusion_matrices(metrics_list, class_names)
    plot_accuracy_ranking(metrics_list)
    
    # Generate report
    print("="*100)
    print("GENERATING REPORT")
    print("="*100 + "\n")
    
    report = generate_report(metrics_list, class_names)
    print(report)
    
    # Save report
    with open(OUT / 'evaluation_report.txt', 'w') as f:
        f.write(report)
    print("✓ Report saved: evaluation_report.txt\n")
    
    # Save metrics to CSV
    df_save = pd.DataFrame([{k: v for k, v in m.items() if k != 'Confusion Matrix'} 
                           for m in metrics_list])
    df_save.to_csv(OUT / 'model_metrics.csv', index=False)
    print("✓ Metrics saved: model_metrics.csv\n")
    
    print("="*100)
    print("STEP 3 MODEL EVALUATION COMPLETE!")
    print("="*100)

if __name__ == "__main__":
    main()
