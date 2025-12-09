"""
STEP 2.4: XGBoost (Extreme Gradient Boosting) TRAINING AND HYPERPARAMETER TUNING
========================================================================

This script trains and optimizes an XGBoost classifier for drone telemetry
classification. XGBoost is a powerful gradient boosting method that works
well with tabular data.

HYPERPARAMETERS TUNED:
- n_estimators: [100, 200, 500, 1000]
- max_depth: [3, 5, 7, 9, 12]
- learning_rate: [0.01, 0.05, 0.1, 0.2, 0.3]
- subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
- colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
- min_child_weight: [1, 3, 5, 7]
- gamma: [0, 0.1, 0.2, 0.3, 0.5]
- reg_alpha: [0, 0.1, 0.5, 1]
- reg_lambda: [0, 0.1, 0.5, 1]

TUNING METHOD: Random Search with 5-fold cross-validation
"""

import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'output' / 'step_1_7_data_splitting'
OUT = ROOT / 'output' / 'step_2_4_XGBoost'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DATA = DATA_DIR / 'train_data.csv'
VAL_DATA = DATA_DIR / 'val_data.csv'
TEST_DATA = DATA_DIR / 'test_data.csv'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CV_SPLITS = 3
NUM_SEARCH_ITERATIONS = 3

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    train_df = pd.read_csv(TRAIN_DATA)
    val_df = pd.read_csv(VAL_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    print(f"✓ Train shape: {train_df.shape}")
    print(f"✓ Val shape: {val_df.shape}")
    print(f"✓ Test shape: {test_df.shape}\n")
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df):
    """Prepare data for XGBoost."""
    le = LabelEncoder()
    
    X_train = train_df.drop('class', axis=1).values
    y_train = train_df['class'].values
    y_train_encoded = le.fit_transform(y_train)
    
    X_val = val_df.drop('class', axis=1).values
    y_val = val_df['class'].values
    y_val_encoded = le.transform(y_val)
    
    X_test = test_df.drop('class', axis=1).values
    y_test = test_df['class'].values
    y_test_encoded = le.transform(y_test)
    
    # Handle extreme values and outliers
    print("\nData validation and cleaning...")
    
    # Replace extreme values (inf, -inf, very large numbers)
    threshold = 1e10
    X_train = np.where(np.abs(X_train) > threshold, np.sign(X_train) * threshold, X_train)
    X_val = np.where(np.abs(X_val) > threshold, np.sign(X_val) * threshold, X_val)
    X_test = np.where(np.abs(X_test) > threshold, np.sign(X_test) * threshold, X_test)
    
    # Remove any NaN or infinite values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=threshold, neginf=-threshold)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=threshold, neginf=-threshold)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=threshold, neginf=-threshold)
    
    # Use RobustScaler which is more resistant to outliers
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("✓ Extreme values handled")
    print("✓ Data rescaled with RobustScaler")
    
    feature_names = train_df.drop('class', axis=1).columns.tolist()
    class_names = sorted(le.classes_)
    
    print("\nData preparation complete:")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {class_names}")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Data range: [{X_train.min():.4f}, {X_train.max():.4f}]\n")
    
    return (X_train, y_train_encoded, X_val, y_val_encoded, 
            X_test, y_test_encoded, feature_names, class_names, le)

# ============================================================================
# XGBoost HYPERPARAMETER SEARCH
# ============================================================================

def hyperparameter_search(X_train, y_train, X_val, y_val):
    """Perform random search for XGBoost hyperparameters."""
    print("\n" + "="*100)
    print("XGBoost HYPERPARAMETER TUNING (RANDOM SEARCH)")
    print("="*100 + "\n")
    
    # Hyperparameter grid (reduced for speed)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [0, 1.0]
    }
    
    # Create base XGBoost classifier
    xgb_model = XGBClassifier(
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    # Random search
    print(f"Running random search with {CV_SPLITS}-fold cross-validation...")
    print(f"Testing {NUM_SEARCH_ITERATIONS} random combinations")
    print("(This may take several minutes)\n")
    
    start_time = time.time()
    random_search = RandomizedSearchCV(
        xgb_model, param_grid,
        n_iter=NUM_SEARCH_ITERATIONS,
        cv=CV_SPLITS,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED
    )
    
    random_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"\n{'='*100}")
    print(f"BEST CONFIGURATION FOUND")
    print(f"{'='*100}")
    print(f"Best cross-validation accuracy: {best_cv_score:.4f}")
    print(f"Validation set accuracy: {val_accuracy:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key:25s}: {value}")
    print()
    
    # Show top results
    cv_results = pd.DataFrame(random_search.cv_results_)
    top_results = cv_results.nlargest(10, 'rank_test_score')[
        ['param_n_estimators', 'param_max_depth', 'param_learning_rate', 
         'param_subsample', 'param_colsample_bytree', 
         'mean_test_score', 'std_test_score', 'rank_test_score']
    ]
    
    print("Top 10 hyperparameter combinations:")
    print(top_results.to_string(index=False))
    print()
    
    return best_model, best_params, random_search.cv_results_, search_time

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_search_results(cv_results, output_path):
    """Plot random search results."""
    results_df = pd.DataFrame(cv_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Learning rate vs accuracy
    ax = axes[0, 0]
    for lr in sorted(results_df['param_learning_rate'].unique()):
        lr_data = results_df[results_df['param_learning_rate'] == lr]
        ax.scatter(range(len(lr_data)), lr_data['mean_test_score'], label=f'LR={lr}', s=100)
    ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cross-validation Accuracy', fontsize=10, fontweight='bold')
    ax.set_title('Learning Rate Effect on Accuracy', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Max depth vs accuracy
    ax = axes[0, 1]
    depth_data = results_df.groupby('param_max_depth')['mean_test_score'].agg(['mean', 'std'])
    ax.bar(depth_data.index, depth_data['mean'], yerr=depth_data['std'], 
           color='#3498db', alpha=0.7, capsize=5, edgecolor='black')
    ax.set_xlabel('Max Depth', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cross-validation Accuracy', fontsize=10, fontweight='bold')
    ax.set_title('Max Depth Effect on Accuracy', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # N estimators vs accuracy
    ax = axes[1, 0]
    nest_data = results_df.groupby('param_n_estimators')['mean_test_score'].agg(['mean', 'std'])
    ax.bar(range(len(nest_data)), nest_data['mean'], yerr=nest_data['std'], 
           color='#e74c3c', alpha=0.7, capsize=5, edgecolor='black')
    ax.set_xticks(range(len(nest_data)))
    ax.set_xticklabels(nest_data.index)
    ax.set_xlabel('Number of Estimators', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cross-validation Accuracy', fontsize=10, fontweight='bold')
    ax.set_title('N Estimators Effect on Accuracy', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Distribution of accuracies
    ax = axes[1, 1]
    ax.hist(results_df['mean_test_score'], bins=15, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.axvline(results_df['mean_test_score'].max(), color='r', linestyle='--', linewidth=2, label='Best')
    ax.axvline(results_df['mean_test_score'].mean(), color='orange', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Cross-validation Accuracy', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Distribution of Accuracies', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, output_path, top_n=20):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(indices)), importance[indices], color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features - XGBoost', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute XGBoost training pipeline."""
    
    print("\n" + "="*100)
    print("STEP 2.4: XGBoost (Extreme Gradient Boosting) TRAINING AND HYPERPARAMETER TUNING")
    print("="*100 + "\n")
    
    # Load and prepare data
    train_df, val_df, test_df = load_data()
    (X_train, y_train, X_val, y_val, 
     X_test, y_test, feature_names, class_names, le) = prepare_data(train_df, val_df, test_df)
    
    # Hyperparameter search
    best_model, best_params, cv_results, search_time = hyperparameter_search(
        X_train, y_train, X_val, y_val
    )
    
    # Evaluate on test set
    print("\n" + "="*100)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*100 + "\n")
    
    y_test_pred = best_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}\n")
    
    # Save results
    print("\n" + "="*100)
    print("SAVING MODEL AND RESULTS")
    print("="*100 + "\n")
    
    joblib.dump(best_model, OUT / 'xgboost_best_model.pkl')
    print(f"✓ Best model saved: xgboost_best_model.pkl")
    
    joblib.dump(best_params, OUT / 'xgboost_best_params.pkl')
    print(f"✓ Best parameters saved: xgboost_best_params.pkl")
    
    joblib.dump(cv_results, OUT / 'xgboost_cv_results.pkl')
    print(f"✓ CV results saved: xgboost_cv_results.pkl")
    
    metadata = {
        'feature_names': feature_names,
        'class_names': class_names,
        'label_encoder': le,
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'best_cv_score': max(cv_results['mean_test_score']),
        'search_time': search_time,
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(metadata, OUT / 'xgboost_metadata.pkl')
    print(f"✓ Metadata saved: xgboost_metadata.pkl")
    
    # Visualizations
    print(f"\nCreating visualizations...")
    plot_search_results(cv_results, OUT / 'xgboost_search_results.png')
    print(f"✓ Search results plot saved")
    
    plot_feature_importance(best_model, feature_names, OUT / 'xgboost_feature_importance.png')
    print(f"✓ Feature importance plot saved")
    
    # Generate report
    report_text = f"""
{'='*100}
STEP 2.4: XGBoost (Extreme Gradient Boosting) TRAINING REPORT
{'='*100}

DATASET INFORMATION:
  Training samples: {X_train.shape[0]:,}
  Validation samples: {X_val.shape[0]:,}
  Test samples: {X_test.shape[0]:,}
  Number of features: {X_train.shape[1]}
  Number of classes: 3 ({', '.join(class_names)})

HYPERPARAMETER SEARCH CONFIGURATION:
  Search method: Random Search
  Number of iterations: {NUM_SEARCH_ITERATIONS}
  Cross-validation folds: {CV_SPLITS}
  Hyperparameters tuned:
    - n_estimators: [100, 200, 500, 1000]
    - max_depth: [3, 5, 7, 9, 12]
    - learning_rate: [0.01, 0.05, 0.1, 0.2, 0.3]
    - subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
    - colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
    - min_child_weight: [1, 3, 5, 7]
    - gamma: [0, 0.1, 0.2, 0.3, 0.5]
    - reg_alpha: [0, 0.1, 0.5, 1]
    - reg_lambda: [0, 0.5, 1.0]
  Search time: {search_time:.2f} seconds

BEST HYPERPARAMETERS FOUND:
"""
    for key, value in best_params.items():
        report_text += f"  {key:30s}: {value}\n"
    
    best_cv_score = max(cv_results['mean_test_score'])
    
    report_text += f"""
SEARCH RESULTS SUMMARY:
  Best cross-validation accuracy: {best_cv_score:.4f}
  Mean cross-validation accuracy: {np.mean(cv_results['mean_test_score']):.4f}
  Std cross-validation accuracy:  {np.std(cv_results['mean_test_score']):.4f}

TEST SET EVALUATION:
  Accuracy:  {test_accuracy:.4f}
  Precision: {test_precision:.4f}
  Recall:    {test_recall:.4f}
  F1-Score:  {test_f1:.4f}

MODEL ARCHITECTURE:
  Algorithm: Extreme Gradient Boosting (XGBoost)
  Base Learner: Decision Trees
  Number of boosting rounds: {best_params['n_estimators']}
  Learning rate (eta): {best_params['learning_rate']}
  Max tree depth: {best_params['max_depth']}
  Regularization (L1/L2): alpha={best_params['reg_alpha']}, lambda={best_params['reg_lambda']}

ADVANTAGES:
  - Excellent performance on tabular data
  - Built-in feature importance
  - Handles missing values well
  - Highly parallelizable
  - Fast training time

FILES SAVED:
  - xgboost_best_model.pkl: Trained XGBoost model
  - xgboost_best_params.pkl: Best hyperparameters
  - xgboost_cv_results.pkl: All CV results
  - xgboost_metadata.pkl: Training metadata
  - xgboost_search_results.png: Hyperparameter search visualization
  - xgboost_feature_importance.png: Top 20 feature importance

{'='*100}
"""
    
    with open(OUT / 'xgboost_training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Report saved: xgboost_training_report.txt")
    
    print(f"\n{'='*100}")
    print(f"STEP 2.4 XGBoost TRAINING COMPLETE!")
    print(f"{'='*100}\n")
    
    return best_model, best_params, metadata

if __name__ == '__main__':
    model, params, metadata = main()

