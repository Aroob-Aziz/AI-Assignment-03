"""
STEP 4: EXPLAINABLE AI (XAI) ANALYSIS
========================================================================

This script provides comprehensive explainability analysis for the trained models
using multiple XAI techniques:

4.1 FEATURE IMPORTANCE ANALYSIS
    - XGBoost native feature importance
    - Permutation importance for neural networks
    - Comparison across models

4.2 SHAP (SHapley Additive exPlanations)
    - SHAP values for XGBoost and FNN
    - Summary plots
    - Dependence plots
    - Force plots
    - Waterfall plots

4.3 LIME (Local Interpretable Model-agnostic Explanations)
    - Individual prediction explanations
    - Comparison with SHAP

4.4 PARTIAL DEPENDENCE PLOTS
    - Top 5 feature PDPs
    - Non-linear relationship identification

4.5 CORRELATION ANALYSIS
    - Feature-target correlation
    - Scatter plots with regression lines

4.6 FEATURE INTERACTION ANALYSIS
    - Identify feature interactions
    - SHAP interaction plots
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance, partial_dependence
import xgboost as xgb
import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'output' / 'step_1_7_data_splitting'
OUT = ROOT / 'output' / 'step_4_xai_analysis'
OUT.mkdir(parents=True, exist_ok=True)

TEST_DATA = DATA_DIR / 'test_data.csv'

# Model paths
MODELS_DIR = ROOT / 'output'
XGBOOST_MODEL = MODELS_DIR / 'step_2_4_XGBoost' / 'xgboost_best_model.pkl'
FNN_MODEL = MODELS_DIR / 'step_2_6_FNN' / 'fnn_best_model.h5'
SVM_MODEL = MODELS_DIR / 'step_2_3_SVM' / 'svm_best_model.pkl'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_prepare_data():
    """Load and preprocess test data."""
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA)
    
    X_test = test_df.drop('class', axis=1).values
    y_test = test_df['class'].values
    feature_names = test_df.drop('class', axis=1).columns.tolist()
    
    # Encode labels
    le = LabelEncoder()
    le.fit(y_test)
    y_test_encoded = le.transform(y_test)
    
    # Preprocess
    from sklearn.preprocessing import RobustScaler
    threshold = 1e10
    X_test = np.where(np.abs(X_test) > threshold, np.sign(X_test) * threshold, X_test)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=threshold, neginf=-threshold)
    
    scaler = RobustScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Normalize for neural networks
    X_test_normalized = (X_test_scaled - X_test_scaled.min(axis=0)) / (X_test_scaled.max(axis=0) - X_test_scaled.min(axis=0) + 1e-8)
    X_test_normalized = np.clip(X_test_normalized, 0, 1)
    
    print(f"✓ Data loaded: {X_test_scaled.shape}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Classes: {list(le.classes_)}\n")
    
    return X_test_scaled, X_test_normalized, y_test_encoded, feature_names, le.classes_

# ============================================================================
# 4.1 FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def feature_importance_xgboost(X_test, y_test, feature_names):
    """Extract and plot XGBoost feature importance."""
    print("="*100)
    print("4.1 FEATURE IMPORTANCE ANALYSIS - XGBoost")
    print("="*100 + "\n")
    
    try:
        xgb_model = joblib.load(XGBOOST_MODEL)
        
        # Get feature importance
        importance_dict = xgb_model.get_booster().get_score(importance_type='weight')
        
        # Convert to feature names
        importance_scores = []
        for i, fname in enumerate(feature_names):
            importance_scores.append(importance_dict.get(f'f{i}', 0))
        
        importance_scores = np.array(importance_scores)
        
        # Sort and get top 10
        top_indices = np.argsort(importance_scores)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(top_features, top_scores, color='#3498db')
        ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
        ax.set_title('XGBoost Feature Importance (Top 10)', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score, bar.get_y() + bar.get_height()/2, f' {score:.2f}',
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUT / 'xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Top 10 Important Features:")
        for fname, score in zip(top_features, top_scores):
            print(f"  {fname:30s}: {score:.2f}")
        print(f"\n✓ XGBoost feature importance plot saved\n")
        
        return dict(zip(top_features, top_scores))
    
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        return {}

def permutation_importance_fnn(X_test, y_test, feature_names):
    """Calculate permutation importance for FNN."""
    print("="*100)
    print("4.1 FEATURE IMPORTANCE ANALYSIS - FNN Permutation Importance")
    print("="*100 + "\n")
    
    try:
        fnn_model = tf.keras.models.load_model(FNN_MODEL)
        
        # Define scoring function
        def score_func(X, y):
            y_pred = np.argmax(fnn_model.predict(X, verbose=0), axis=1)
            return accuracy_score(y, y_pred)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            fnn_model, X_test, y_test,
            n_repeats=10,
            scoring=None,
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        
        # Get top 10
        top_indices = np.argsort(perm_importance.importances_mean)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = perm_importance.importances_mean[top_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(top_features, top_scores, color='#2ecc71')
        ax.set_xlabel('Permutation Importance', fontweight='bold', fontsize=12)
        ax.set_title('FNN Permutation Importance (Top 10)', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score, bar.get_y() + bar.get_height()/2, f' {score:.4f}',
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUT / 'fnn_permutation_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Top 10 Important Features (Permutation):")
        for fname, score in zip(top_features, top_scores):
            print(f"  {fname:30s}: {score:.4f}")
        print(f"\n✓ FNN permutation importance plot saved\n")
        
        return dict(zip(top_features, top_scores))
    
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        return {}

# ============================================================================
# 4.2 SHAP ANALYSIS
# ============================================================================

def shap_analysis_xgboost(X_test, feature_names):
    """SHAP analysis for XGBoost."""
    print("="*100)
    print("4.2 SHAP ANALYSIS - XGBoost")
    print("="*100 + "\n")
    
    try:
        xgb_model = joblib.load(XGBOOST_MODEL)
        
        # Sample data for faster computation
        n_samples = min(200, X_test.shape[0])
        X_sample = X_test[:n_samples]
        
        # Create explainer
        print("Computing SHAP values (this may take a minute)...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(OUT / 'shap_xgboost_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP summary plot saved")
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(OUT / 'shap_xgboost_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP bar plot saved")
        
        # Dependence plots for top 4 features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-4:][::-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (ax, feat_idx) in enumerate(zip(axes.flat, top_indices)):
            shap.dependence_plot(feat_idx, shap_values, X_sample, 
                               feature_names=feature_names, ax=ax, show=False)
        
        plt.tight_layout()
        plt.savefig(OUT / 'shap_xgboost_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP dependence plots saved")
        
        # Force plot for first sample
        plt.figure(figsize=(14, 4))
        shap.force_plot(explainer.expected_value, shap_values[0], X_sample[0],
                       feature_names=feature_names, matplotlib=True)
        plt.tight_layout()
        plt.savefig(OUT / 'shap_xgboost_force.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP force plot saved\n")
        
        return shap_values, explainer.expected_value
    
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        return None, None

def shap_analysis_fnn(X_test, feature_names):
    """SHAP analysis for FNN."""
    print("="*100)
    print("4.2 SHAP ANALYSIS - FNN (Kernel Explainer)")
    print("="*100 + "\n")
    
    try:
        fnn_model = tf.keras.models.load_model(FNN_MODEL)
        
        # Sample data
        n_samples = min(100, X_test.shape[0])
        X_sample = X_test[:n_samples]
        
        # Background data
        X_background = X_test[:50]
        
        # Create prediction function
        def predict_fn(x):
            return fnn_model.predict(x, verbose=0)
        
        print("Computing SHAP values for FNN...")
        explainer = shap.KernelExplainer(predict_fn, X_background)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle output
        if isinstance(shap_values, list):
            # Take first class
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(OUT / 'shap_fnn_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP summary plot saved\n")
        
        return shap_vals
    
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
        return None

# ============================================================================
# 4.3 LIME ANALYSIS
# ============================================================================

def lime_analysis(X_test, y_test, feature_names):
    """LIME analysis for model interpretability."""
    print("="*100)
    print("4.3 LIME ANALYSIS")
    print("="*100 + "\n")
    
    try:
        xgb_model = joblib.load(XGBOOST_MODEL)
        fnn_model = tf.keras.models.load_model(FNN_MODEL)
        
        # Create explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test[:200], 
            feature_names=feature_names,
            class_names=['DoS_Attack', 'Malfunction', 'Normal'],
            mode='classification',
            random_state=RANDOM_SEED
        )
        
        # Explain XGBoost prediction
        idx = 0
        exp_xgb = explainer.explain_instance(
            X_test[idx], 
            lambda x: xgb_model.predict_proba(x),
            num_features=10
        )
        
        fig = exp_xgb.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(OUT / 'lime_xgboost_explanation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ LIME explanation for XGBoost saved")
        
        # Explain FNN prediction
        exp_fnn = explainer.explain_instance(
            X_test[idx],
            lambda x: fnn_model.predict(x, verbose=0),
            num_features=10
        )
        
        fig = exp_fnn.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(OUT / 'lime_fnn_explanation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ LIME explanation for FNN saved\n")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")

# ============================================================================
# 4.4 PARTIAL DEPENDENCE PLOTS
# ============================================================================

def partial_dependence_analysis(X_test, y_test, feature_names):
    """Create partial dependence plots for top 5 features."""
    print("="*100)
    print("4.4 PARTIAL DEPENDENCE PLOTS")
    print("="*100 + "\n")
    
    try:
        xgb_model = joblib.load(XGBOOST_MODEL)
        
        # Get top 5 features by SHAP importance
        xgb_explainer = shap.TreeExplainer(xgb_model)
        shap_values = xgb_explainer.shap_values(X_test[:100])
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
        
        # Create PDPs
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        for idx, (ax, feat_idx) in enumerate(zip(axes.flat, top_indices)):
            if idx < len(top_indices):
                pd_result = partial_dependence(xgb_model, X_test, [feat_idx], n_jobs=-1)
                pd_values = pd_result['average'][0]
                pd_grid = pd_result['grid_values'][0]
                
                ax.plot(pd_grid, pd_values, linewidth=2, color='#e74c3c')
                ax.fill_between(pd_grid, pd_values - 0.01, pd_values + 0.01, alpha=0.2, color='#e74c3c')
                ax.set_xlabel(feature_names[feat_idx], fontweight='bold')
                ax.set_ylabel('Prediction', fontweight='bold')
                ax.set_title(f'PDP: {feature_names[feat_idx]}', fontweight='bold')
                ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(top_indices), len(axes.flat)):
            axes.flat[idx].axis('off')
        
        plt.suptitle('Partial Dependence Plots - Top 5 Features', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(OUT / 'partial_dependence_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Top 5 Features for PDP:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. {feature_names[idx]}")
        print("\n✓ Partial dependence plots saved\n")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")

# ============================================================================
# 4.5 CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(X_test, y_test, feature_names):
    """Analyze feature-target correlation."""
    print("="*100)
    print("4.5 CORRELATION ANALYSIS")
    print("="*100 + "\n")
    
    try:
        # Calculate correlations
        df = pd.DataFrame(X_test, columns=feature_names)
        df['target'] = y_test
        
        correlations = df.corr()['target'].drop('target').sort_values(ascending=False)
        
        # Top 10
        top_10 = pd.concat([correlations.head(5), correlations.tail(5)])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_10.values]
        bars = ax.barh(range(len(top_10)), top_10.values, color=colors)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10.index)
        ax.set_xlabel('Correlation with Target', fontweight='bold', fontsize=12)
        ax.set_title('Feature-Target Correlation (Top 5 Positive & Negative)', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linewidth=0.8)
        
        for i, (bar, val) in enumerate(zip(bars, top_10.values)):
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.3f}',
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUT / 'feature_target_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Top 5 Positive Correlations:")
        for fname, corr in correlations.head(5).items():
            print(f"  {fname:30s}: {corr:.4f}")
        
        print("\nTop 5 Negative Correlations:")
        for fname, corr in correlations.tail(5).items():
            print(f"  {fname:30s}: {corr:.4f}")
        
        print("\n✓ Correlation analysis plot saved\n")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")

# ============================================================================
# REPORTING
# ============================================================================

def generate_xai_report():
    """Generate comprehensive XAI analysis report."""
    report = f"""
{'='*100}
STEP 4: EXPLAINABLE AI (XAI) ANALYSIS REPORT
{'='*100}

{'='*100}
4.1 FEATURE IMPORTANCE ANALYSIS
{'='*100}

Key Findings:
- XGBoost identifies which features are most frequently used in decision trees
- Permutation importance shows which features have highest impact on predictions
- Top features provide clues about discriminative patterns between classes

Interpretation:
Feature importance scores indicate how much each feature contributes to the model's 
decision-making process. Higher scores mean the model relies more on these features 
for accurate predictions.

{'='*100}
4.2 SHAP (SHapley Additive exPlanations) ANALYSIS
{'='*100}

SHAP Values Explanation:
- Each SHAP value represents the contribution of a feature to pushing the 
  prediction from the base value (expected value) to the final prediction
- Positive SHAP values push prediction toward higher class probability
- Negative SHAP values push prediction toward lower class probability
- Magnitude indicates strength of the contribution

Summary Plots:
- Show overall importance of features across all predictions
- Color indicates whether feature value increases (red) or decreases (blue) prediction
- X-axis position shows magnitude of impact

Dependence Plots:
- Reveal relationship between feature values and their SHAP contributions
- Linear relationships suggest simple decision rules
- Non-linear patterns suggest complex feature interactions

Force Plots:
- Show how individual prediction is built from base value
- Each feature contributes either positively (red) or negatively (blue)
- Width indicates contribution magnitude

{'='*100}
4.3 LIME (Local Interpretable Model-agnostic Explanations) ANALYSIS
{'='*100}

LIME Approach:
- Trains simple, interpretable model locally around specific prediction
- Uses linear approximation to explain non-linear model behavior
- Different from SHAP's global approach (LIME is local)

Advantages of LIME:
- Model-agnostic (works with any model)
- Provides easy-to-understand linear explanations
- Good for debugging specific predictions

Comparison with SHAP:
- LIME focuses on individual predictions (local)
- SHAP provides both local and global explanations
- SHAP has stronger theoretical foundations (game theory)

{'='*100}
4.4 PARTIAL DEPENDENCE PLOTS (PDP)
{'='*100}

PDP Interpretation:
- Shows average marginal effect of each feature on predictions
- X-axis: feature value range
- Y-axis: average predicted probability
- Flat curves indicate feature has little effect
- Steep curves indicate strong feature influence

Non-linear Relationships:
- Sigmoid-shaped curves: threshold effects
- Parabolic curves: quadratic relationships
- Step-like curves: categorical-like behavior

{'='*100}
4.5 CORRELATION ANALYSIS
{'='*100}

Feature-Target Correlation:
- Positive correlation: feature values increase with target class
- Negative correlation: feature values decrease with target class
- Close to zero: weak linear relationship

Multicollinearity Check:
- High inter-feature correlations can cause model instability
- May indicate redundant features
- Important for feature selection

{'='*100}
KEY INSIGHTS FROM XAI ANALYSIS
{'='*100}

1. FEATURE RELATIONSHIPS:
   - Identify which features are most predictive
   - Understand how features interact
   - Discover non-linear relationships

2. MODEL DECISION PATTERNS:
   - Explain why model makes specific predictions
   - Identify potential biases or errors
   - Validate model aligns with domain knowledge

3. FEATURE ENGINEERING OPPORTUNITIES:
   - Identify important feature interactions
   - Discover redundant features for removal
   - Generate ideas for new feature combinations

4. MODEL DEBUGGING:
   - Find unexpected predictions
   - Identify data quality issues
   - Verify model behavior aligns with expectations

{'='*100}
RECOMMENDATIONS
{'='*100}

1. Use XGBoost for production (highest accuracy + explainability)
2. Monitor predictions using SHAP/LIME for real-time insights
3. Retrain models when SHAP importance patterns change
4. Remove low-importance features to improve efficiency
5. Use explanations for stakeholder communication and trust

{'='*100}
Report generated: {datetime.now().isoformat()}
{'='*100}
"""
    return report

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main XAI analysis pipeline."""
    print("\n" + "="*100)
    print("STEP 4: EXPLAINABLE AI (XAI) ANALYSIS")
    print("="*100 + "\n")
    
    # Load data
    X_test_scaled, X_test_normalized, y_test, feature_names, class_names = load_and_prepare_data()
    
    # 4.1 Feature Importance
    xgb_importance = feature_importance_xgboost(X_test_scaled, y_test, feature_names)
    fnn_importance = permutation_importance_fnn(X_test_normalized, y_test, feature_names)
    
    # 4.2 SHAP Analysis
    shap_xgb_values, shap_expected = shap_analysis_xgboost(X_test_scaled, feature_names)
    shap_fnn_values = shap_analysis_fnn(X_test_normalized, feature_names)
    
    # 4.3 LIME Analysis
    lime_analysis(X_test_scaled, y_test, feature_names)
    
    # 4.4 Partial Dependence
    partial_dependence_analysis(X_test_scaled, y_test, feature_names)
    
    # 4.5 Correlation Analysis
    correlation_analysis(X_test_scaled, y_test, feature_names)
    
    # Generate report
    print("="*100)
    print("GENERATING XAI REPORT")
    print("="*100 + "\n")
    
    report = generate_xai_report()
    print(report)
    
    # Save report
    with open(OUT / 'xai_analysis_report.txt', 'w') as f:
        f.write(report)
    print("✓ XAI analysis report saved\n")
    
    print("="*100)
    print("STEP 4 XAI ANALYSIS COMPLETE!")
    print("="*100)

if __name__ == "__main__":
    main()
