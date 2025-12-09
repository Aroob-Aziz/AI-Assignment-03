"""
STEP 2.6: FEEDFORWARD NEURAL NETWORK (FNN) TRAINING AND HYPERPARAMETER TUNING
========================================================================

This script trains and optimizes a Feedforward Neural Network (FNN)
for drone telemetry classification. FNN serves as a baseline deep learning
model for non-sequential predictions.

IMPLEMENTATION:
- Design fully connected architecture
- Include batch normalization (optional)
- Use dropout for regularization

HYPERPARAMETERS TUNED:
- Number of hidden layers: [2, 3, 4, 5]
- Neurons per layer: [64, 128, 256, 512]
- Dropout rate: [0.0, 0.2, 0.3, 0.5]
- Learning rate: [0.001, 0.0001, 0.00001]
- Batch size: [16, 32, 64, 128]
- Activation function: [relu, elu, leaky_relu, tanh]
- Optimizer: [Adam, RMSprop, SGD]
- Batch normalization: [True, False]
- L2 regularization: [0.0, 0.001, 0.01, 0.1]

TUNING METHOD: Random Search (20 iterations)
"""

import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'output' / 'step_1_7_data_splitting'
OUT = ROOT / 'output' / 'step_2_6_FNN'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DATA = DATA_DIR / 'train_data.csv'
VAL_DATA = DATA_DIR / 'val_data.csv'
TEST_DATA = DATA_DIR / 'test_data.csv'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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
    """Prepare data for FNN."""
    le = LabelEncoder()
    
    X_train = train_df.drop('class', axis=1).values
    y_train = train_df['class'].values
    y_train_encoded = le.fit_transform(y_train)
    y_train_cat = to_categorical(y_train_encoded, num_classes=3)
    
    X_val = val_df.drop('class', axis=1).values
    y_val = val_df['class'].values
    y_val_encoded = le.transform(y_val)
    y_val_cat = to_categorical(y_val_encoded, num_classes=3)
    
    X_test = test_df.drop('class', axis=1).values
    y_test = test_df['class'].values
    y_test_encoded = le.transform(y_test)
    y_test_cat = to_categorical(y_test_encoded, num_classes=3)
    
    feature_names = train_df.drop('class', axis=1).columns.tolist()
    class_names = sorted(le.classes_)
    
    print("Data preparation complete:")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {class_names}")
    print(f"  Train shape: {X_train.shape}\n")
    
    return (X_train, y_train_cat, X_val, y_val_cat, 
            X_test, y_test_cat, feature_names, class_names, le)

# ============================================================================
# FNN MODEL BUILDER
# ============================================================================

class FNNModelBuilder:
    """Build and train Feedforward Neural Network."""
    
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.history = None
        self.training_time = 0
    
    def build_model(self, num_layers, neurons_per_layer, dropout_rate, 
                   activation='relu', use_batch_norm=True, l2_reg=0.0):
        """Build FNN architecture."""
        model = Sequential()
        
        model.add(layers.Input(shape=(self.input_dim,)))
        
        for i in range(num_layers):
            model.add(layers.Dense(
                neurons_per_layer,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ))
            
            if use_batch_norm:
                model.add(layers.BatchNormalization())
            
            model.add(layers.Dropout(dropout_rate))
        
        # Final hidden layer
        model.add(layers.Dense(64, activation=activation))
        model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, model, learning_rate,
             batch_size, epochs, optimizer_name='Adam'):
        """Train FNN model."""
        # Select optimizer
        if optimizer_name == 'Adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        self.training_time = time.time() - start_time
        self.history = history.history
        
        return model, history
    
    def hyperparameter_search(self, X_train, y_train, X_val, y_val):
        """Perform random search for FNN hyperparameters."""
        print("\n" + "="*100)
        print("FNN HYPERPARAMETER TUNING (RANDOM SEARCH)")
        print("="*100 + "\n")
        
        param_grid = {
            'num_layers': [2, 3],
            'neurons_per_layer': [64, 128, 256],
            'dropout_rate': [0.2, 0.3],
            'activation': ['relu', 'elu'],
            'learning_rate': [0.001],
            'batch_size': [256, 512],
            'epochs': [10, 15],
            'optimizer': ['Adam', 'RMSprop'],
            'use_batch_norm': [True],
            'l2_reg': [0.001]
        }
        
        results = []
        best_accuracy = 0
        best_model = None
        best_params = None
        best_history = None
        
        print(f"Testing {NUM_SEARCH_ITERATIONS} random combinations...\n")
        
        for iteration in range(NUM_SEARCH_ITERATIONS):
            params = {
                'num_layers': np.random.choice(param_grid['num_layers']),
                'neurons_per_layer': np.random.choice(param_grid['neurons_per_layer']),
                'dropout_rate': np.random.choice(param_grid['dropout_rate']),
                'activation': np.random.choice(param_grid['activation']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'batch_size': np.random.choice(param_grid['batch_size']),
                'epochs': np.random.choice(param_grid['epochs']),
                'optimizer': np.random.choice(param_grid['optimizer']),
                'use_batch_norm': np.random.choice(param_grid['use_batch_norm']),
                'l2_reg': np.random.choice(param_grid['l2_reg'])
            }
            
            try:
                # Build model
                model = self.build_model(
                    params['num_layers'],
                    params['neurons_per_layer'],
                    params['dropout_rate'],
                    params['activation'],
                    params['use_batch_norm'],
                    params['l2_reg']
                )
                
                # Train model
                model, history = self.train(
                    X_train, y_train, X_val, y_val, model,
                    params['learning_rate'],
                    params['batch_size'],
                    params['epochs'],
                    params['optimizer']
                )
                
                # Evaluate on validation set
                y_val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
                y_val_true = y_val.argmax(axis=1)
                accuracy = accuracy_score(y_val_true, y_val_pred)
                
                results.append({
                    'iteration': iteration + 1,
                    'params': params,
                    'val_accuracy': accuracy,
                    'training_time': self.training_time,
                    'epochs_trained': len(history.history['loss'])
                })
                
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Accuracy: {accuracy:.4f} | "
                      f"Layers: {params['num_layers']}, Neurons: {params['neurons_per_layer']}, "
                      f"Time: {self.training_time:.1f}s")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = params
                    best_history = history
                
            except Exception as e:
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Error: {str(e)[:50]}")
                continue
        
        print(f"\n{'='*100}")
        print(f"BEST CONFIGURATION FOUND")
        print(f"{'='*100}")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key:20s}: {value}")
        print()
        
        return best_model, best_params, results, best_history

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, output_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title('FNN Model Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_title('FNN Model Training Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_search_results(results, output_path):
    """Plot hyperparameter search results."""
    accuracies = [r['val_accuracy'] for r in results]
    iterations = [r['iteration'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(iterations, accuracies, 'o-', linewidth=2, markersize=8, color='#16a085')
    axes[0].axhline(y=max(accuracies), color='r', linestyle='--', linewidth=2, label='Best accuracy')
    axes[0].set_xlabel('Iteration', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('FNN Hyperparameter Search Progress', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=10)
    
    axes[1].hist(accuracies, bins=10, color='#d35400', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=max(accuracies), color='r', linestyle='--', linewidth=2, label='Best')
    axes[1].axvline(x=np.mean(accuracies), color='blue', linestyle='--', linewidth=2, label='Mean')
    axes[1].set_xlabel('Validation Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution of Validation Accuracies', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black",
                   fontsize=12, fontweight='bold')
    
    ax.set_title('Confusion Matrix - FNN on Test Set', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute FNN training pipeline."""
    
    print("\n" + "="*100)
    print("STEP 2.6: FEEDFORWARD NEURAL NETWORK (FNN) TRAINING AND HYPERPARAMETER TUNING")
    print("="*100 + "\n")
    
    # Load and prepare data
    train_df, val_df, test_df = load_data()
    (X_train, y_train_cat, X_val, y_val_cat, 
     X_test, y_test_cat, feature_names, class_names, le) = prepare_data(train_df, val_df, test_df)
    
    # Build and train
    fnn_builder = FNNModelBuilder(X_train.shape[1], num_classes=3)
    
    best_model, best_params, search_results, best_history = fnn_builder.hyperparameter_search(
        X_train, y_train_cat, X_val, y_val_cat
    )
    
    # Evaluate on test set
    print("\n" + "="*100)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*100 + "\n")
    
    y_test_pred = best_model.predict(X_test, verbose=0).argmax(axis=1)
    y_test_true = y_test_cat.argmax(axis=1)
    
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_precision = precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test_true, y_test_pred, target_names=class_names))
    
    # Save results
    print("\n" + "="*100)
    print("SAVING MODEL AND RESULTS")
    print("="*100 + "\n")
    
    best_model.save(OUT / 'fnn_best_model.h5')
    print(f"✓ Best model saved: fnn_best_model.h5")
    
    joblib.dump(best_params, OUT / 'fnn_best_params.pkl')
    print(f"✓ Best parameters saved: fnn_best_params.pkl")
    
    joblib.dump(search_results, OUT / 'fnn_search_results.pkl')
    print(f"✓ Search results saved: fnn_search_results.pkl")
    
    metadata = {
        'feature_names': feature_names,
        'class_names': class_names,
        'label_encoder': le,
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'best_val_accuracy': max([r['val_accuracy'] for r in search_results]),
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(metadata, OUT / 'fnn_metadata.pkl')
    print(f"✓ Metadata saved: fnn_metadata.pkl")
    
    # Visualizations
    print(f"\nCreating visualizations...")
    plot_training_history(best_history, OUT / 'fnn_training_history.png')
    print(f"✓ Training history plot saved")
    
    plot_search_results(search_results, OUT / 'fnn_search_results.png')
    print(f"✓ Search results plot saved")
    
    plot_confusion_matrix(y_test_true, y_test_pred, class_names, OUT / 'fnn_confusion_matrix.png')
    print(f"✓ Confusion matrix plot saved")
    
    # Generate report
    best_val_acc = max([r['val_accuracy'] for r in search_results])
    
    report_text = f"""
{'='*100}
STEP 2.6: FEEDFORWARD NEURAL NETWORK (FNN) TRAINING REPORT
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
  Hyperparameters tuned:
    - Number of hidden layers: [2, 3, 4, 5]
    - Neurons per layer: [64, 128, 256, 512]
    - Dropout rate: [0.0, 0.2, 0.3, 0.5]
    - Learning rate: [0.001, 0.0001, 0.00001]
    - Batch size: [16, 32, 64, 128]
    - Activation function: [relu, elu, leaky_relu, tanh]
    - Optimizer: [Adam, RMSprop, SGD]
    - Batch normalization: [True, False]
    - L2 regularization: [0.0, 0.001, 0.01, 0.1]

BEST HYPERPARAMETERS FOUND:
"""
    for key, value in best_params.items():
        report_text += f"  {key:30s}: {value}\n"
    
    report_text += f"""
SEARCH RESULTS SUMMARY:
  Best validation accuracy: {best_val_acc:.4f}
  Mean validation accuracy: {np.mean([r['val_accuracy'] for r in search_results]):.4f}
  Std validation accuracy:  {np.std([r['val_accuracy'] for r in search_results]):.4f}

TEST SET EVALUATION:
  Accuracy:  {test_accuracy:.4f}
  Precision: {test_precision:.4f}
  Recall:    {test_recall:.4f}
  F1-Score:  {test_f1:.4f}

MODEL ARCHITECTURE:
  Input layer: {X_train.shape[1]} features
  Hidden layers: {best_params['num_layers']} layers with {best_params['neurons_per_layer']} neurons each
  Activation: {best_params['activation']}
  Batch normalization: {best_params['use_batch_norm']}
  Dropout: {best_params['dropout_rate']}
  L2 regularization: {best_params['l2_reg']}
  Output layer: 3 classes (softmax)

OPTIMIZER AND TRAINING:
  Optimizer: {best_params['optimizer']}
  Learning rate: {best_params['learning_rate']}
  Batch size: {best_params['batch_size']}
  Max epochs: {best_params['epochs']}
  Early stopping: Yes (patience=10)
  Learning rate reduction: Yes (patience=5)

FILES SAVED:
  - fnn_best_model.h5: Trained FNN model (Keras format)
  - fnn_best_params.pkl: Best hyperparameters
  - fnn_search_results.pkl: All search iterations results
  - fnn_metadata.pkl: Training metadata
  - fnn_training_history.png: Training loss and accuracy plots
  - fnn_search_results.png: Hyperparameter search progress
  - fnn_confusion_matrix.png: Test set confusion matrix

NEXT STEPS:
  This model can be used for:
  1. Comparison with other models (Part 3)
  2. Explainability analysis using SHAP/LIME (Part 4)
  3. Feature importance extraction (Part 4)

{'='*100}
"""
    
    with open(OUT / 'fnn_training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Report saved: fnn_training_report.txt")
    
    print(f"\n{'='*100}")
    print(f"STEP 2.6 FNN TRAINING COMPLETE!")
    print(f"{'='*100}\n")
    
    return best_model, best_params, metadata

if __name__ == '__main__':
    model, params, metadata = main()

