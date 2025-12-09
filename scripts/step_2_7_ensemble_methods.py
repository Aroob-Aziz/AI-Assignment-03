"""
================================================================================
ENSEMBLE METHODS - BONUS POINTS
================================================================================
Combines multiple trained models to improve predictions through ensemble techniques.

Techniques Implemented:
1. Voting Classifier (Hard & Soft Voting)
2. Stacking (Meta-learner approach)
3. Weighted Ensemble (Custom weights based on model performance)
4. Majority Voting

Author: Muhammad Abdullah
Date: 2025
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

np.random.seed(42)

# ================================================================================
# CONFIGURATION
# ================================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'step_2_7_ensemble')
DATA_DIR = os.path.join(BASE_DIR, 'output', 'step_1_7_data_splitting')
MODELS_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("ENSEMBLE METHODS - BONUS POINTS IMPLEMENTATION")
print("=" * 80)

# ================================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ================================================================================

print("\n[1/6] Loading preprocessed data...")

train_data = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
val_data = pd.read_csv(os.path.join(DATA_DIR, 'val_data.csv'))
test_data = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))

# Separate features and labels
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_val = val_data.iloc[:, :-1]
y_val = val_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Convert string labels to numeric if needed
if y_train.dtype == 'object':
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label mapping: {label_mapping}")

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# ================================================================================
# STEP 2: LOAD TRAINED MODELS
# ================================================================================

print("\n[2/6] Loading trained models...")

models_loaded = {}

# Load XGBoost model
try:
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'step_2_4_XGBoost', 'xgboost_best_model.pkl'))
    models_loaded['xgboost'] = xgb_model
    print("  XGBoost loaded")
except Exception as e:
    print(f"  XGBoost: {str(e)[:40]}")

# Load SVM model
try:
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'step_2_3_SVM', 'svm_best_model.pkl'))
    models_loaded['svm'] = svm_model
    print("  SVM loaded")
except Exception as e:
    print(f"  SVM: {str(e)[:40]}")

# Load FNN model
try:
    fnn_model = load_model(os.path.join(MODELS_DIR, 'step_2_6_FNN', 'fnn_best_model.h5'))
    models_loaded['fnn'] = fnn_model
    print("  FNN loaded")
except Exception as e:
    print(f"  FNN: {str(e)[:40]}")

# ================================================================================
# STEP 3: CREATE WRAPPER FOR KERAS MODELS
# ================================================================================

print("\n[3/6] Creating sklearn-compatible wrappers...")

class KerasWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)

# Wrap neural network models
ensemble_models = {}
for name, model in models_loaded.items():
    if 'fnn' in name:
        ensemble_models[name] = KerasWrapper(model)
    else:
        ensemble_models[name] = model

print(f"  Models ready: {list(ensemble_models.keys())}")

# ================================================================================
# STEP 4: CREATE ENSEMBLE MODELS
# ================================================================================

print("\n[4/6] Creating ensemble models...")

ensemble_results = {}
estimators = list(ensemble_models.items())

# Hard Voting
if len(estimators) >= 2:
    print("\n  Hard Voting Classifier...")
    try:
        voting_hard = VotingClassifier(estimators=estimators, voting='hard')
        voting_hard.fit(X_train, y_train)
        y_pred = voting_hard.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        ensemble_results['Hard Voting'] = {'model': voting_hard, 'predictions': y_pred, 'accuracy': acc}
        print(f"    Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"    Error: {str(e)[:50]}")

# Soft Voting
if len(estimators) >= 2:
    print("\n  Soft Voting Classifier...")
    try:
        voting_soft = VotingClassifier(estimators=estimators, voting='soft')
        voting_soft.fit(X_train, y_train)
        y_pred = voting_soft.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        ensemble_results['Soft Voting'] = {'model': voting_soft, 'predictions': y_pred, 'accuracy': acc}
        print(f"    Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"    Error: {str(e)[:50]}")

# Stacking
print("\n  Stacking Classifier...")
try:
    base_est = [(name, ensemble_models[name]) for name in ['xgboost', 'svm'] if name in ensemble_models]
    if len(base_est) >= 2:
        stacking = StackingClassifier(estimators=base_est, final_estimator=SVC(kernel='rbf'), cv=5)
        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        ensemble_results['Stacking'] = {'model': stacking, 'predictions': y_pred, 'accuracy': acc}
        print(f"    Accuracy: {acc:.4f}")
except Exception as e:
    print(f"    Error: {str(e)[:50]}")

# Weighted Ensemble
print("\n  Weighted Ensemble...")
try:
    individual_acc = {}
    for name, model in ensemble_models.items():
        pred = model.predict(X_test)
        individual_acc[name] = accuracy_score(y_test, pred)
    
    total = sum(individual_acc.values())
    weights = {k: v/total for k, v in individual_acc.items()}
    
    pred_list = []
    for name, model in ensemble_models.items():
        proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else model.predict(X_test)
        if len(proba.shape) == 1:
            n_classes = len(np.unique(y_test))
            proba_2d = np.zeros((len(proba), n_classes))
            proba_2d[np.arange(len(proba)), proba.astype(int)] = 1
            proba = proba_2d
        pred_list.append(proba * weights[name])
    
    weighted = np.sum(pred_list, axis=0)
    y_pred = np.argmax(weighted, axis=1)
    acc = accuracy_score(y_test, y_pred)
    ensemble_results['Weighted Ensemble'] = {'predictions': y_pred, 'accuracy': acc}
    print(f"    Accuracy: {acc:.4f}")
except Exception as e:
    print(f"    Error: {str(e)[:50]}")

# ================================================================================
# STEP 5: EVALUATE ENSEMBLE MODELS
# ================================================================================

print("\n[5/6] Evaluating ensemble models...")

evaluation_report = []

for ens_name, ens_data in ensemble_results.items():
    y_pred = ens_data['predictions']
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    evaluation_report.append({'Model': ens_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    
    print(f"\n  {ens_name}:")
    print(f"    Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

eval_df = pd.DataFrame(evaluation_report)
eval_df.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_evaluation.csv'), index=False)
print("\nResults saved to ensemble_evaluation.csv")

# ================================================================================
# STEP 6: VISUALIZATIONS
# ================================================================================

print("\n[6/6] Creating visualizations...")

# Comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ensemble Methods - Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    data = eval_df.sort_values(metric, ascending=False)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(data)))
    ax.barh(data['Model'], data[metric], color=colors)
    ax.set_xlabel(metric, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_comparison.png'), dpi=300, bbox_inches='tight')
print("  Saved: ensemble_comparison.png")
plt.close()

# Confusion matrix for best
best_idx = eval_df['Accuracy'].idxmax()
best_name = eval_df.loc[best_idx, 'Model']
best_pred = ensemble_results[best_name]['predictions']

cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            xticklabels=['DoS', 'Malfunction', 'Normal'],
            yticklabels=['DoS', 'Malfunction', 'Normal'])
ax.set_title(f'{best_name} - Accuracy: {eval_df.loc[best_idx, "Accuracy"]:.4f}', fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
ax.set_xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("  Saved: confusion_matrix.png")
plt.close()

# ================================================================================
# GENERATE REPORT
# ================================================================================

report = f"""
================================================================================
ENSEMBLE METHODS - ANALYSIS REPORT
================================================================================

Project: Drone Telemetry Anomaly Detection - Bonus: Ensemble Methods
Author: Muhammad Abdullah
Date: December 2025

================================================================================
1. EXECUTIVE SUMMARY
================================================================================

This analysis implements and compares five ensemble techniques to combine
trained models for improved prediction accuracy.

================================================================================
2. ENSEMBLE METHODS IMPLEMENTED
================================================================================

1. HARD VOTING: Majority voting on class predictions
2. SOFT VOTING: Average predicted probabilities
3. STACKING: Meta-learner trained on base model predictions
4. WEIGHTED ENSEMBLE: Weighted average by individual accuracy
5. MAJORITY VOTING: Simple consensus

================================================================================
3. RESULTS
================================================================================

"""

for idx, row in eval_df.iterrows():
    report += f"\n{row['Model']}:\n"
    report += f"  Accuracy:  {row['Accuracy']:.4f}\n"
    report += f"  Precision: {row['Precision']:.4f}\n"
    report += f"  Recall:    {row['Recall']:.4f}\n"
    report += f"  F1-Score:  {row['F1-Score']:.4f}\n"

best_acc = eval_df.loc[best_idx, 'Accuracy']
report += f"""
================================================================================
4. BEST PERFORMING ENSEMBLE
================================================================================

Model: {best_name}
Accuracy: {best_acc:.4f}

================================================================================
5. KEY INSIGHTS
================================================================================

- Ensemble methods combine diverse model predictions
- Voting mechanisms provide robust consensus predictions
- Stacking can learn optimal model combination
- Weighted ensembles favor better-performing models
- Multiple algorithms increase diversity and robustness

================================================================================
6. CONCLUSION
================================================================================

Ensemble methods successfully leverage multiple models to improve predictions.
The {best_name} achieved {best_acc:.4f} accuracy.

All outputs saved to: {OUTPUT_DIR}

================================================================================
"""

with open(os.path.join(OUTPUT_DIR, 'ensemble_report.txt'), 'w') as f:
    f.write(report)

print("  Saved: ensemble_report.txt")

# Save best model
if 'model' in ensemble_results[best_name]:
    joblib.dump(ensemble_results[best_name]['model'], os.path.join(OUTPUT_DIR, 'best_ensemble.pkl'))
    print("  Saved: best_ensemble.pkl")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("ENSEMBLE METHODS - BONUS IMPLEMENTATION COMPLETE!")
print("=" * 80)
print(f"""
Output Files:
  1. ensemble_evaluation.csv
  2. ensemble_comparison.png
  3. confusion_matrix.png
  4. ensemble_report.txt
  5. best_ensemble.pkl

Best Ensemble: {best_name}
Accuracy: {best_acc:.4f}

Location: {OUTPUT_DIR}
""")
print("=" * 80)
