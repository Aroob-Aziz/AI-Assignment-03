"""
STEP 2.2: 1D-CNN MODEL TRAINING AND HYPERPARAMETER TUNING
========================================================================

This script trains and optimizes a 1D Convolutional Neural Network for
drone telemetry classification. 1D-CNN extracts local patterns and
features from sequential sensor data.

IMPLEMENTATION REQUIREMENTS:
- Define 1D convolutional layers
- Include pooling layers (MaxPooling1D or AveragePooling1D)
- Flatten and add dense layers for final prediction
- Hyperparameter tuning via random search

HYPERPARAMETERS TUNED:
- Number of convolutional layers: [1, 2, 3, 4]
- Number of filters: [32, 64, 128, 256]
- Kernel size: [3, 5, 7, 9]
- Pooling size: [2, 3, 4]
- Dropout rate: [0.0, 0.2, 0.3, 0.5]
- Learning rate: [0.001, 0.0001]
- Batch size: [16, 32, 64]
- Activation function: [relu, elu, leaky_relu]
- Dense layer units: [64, 128, 256]

TUNING METHOD: Random Search (20 iterations)
"""

import os
import time
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'output' / 'step_1_7_data_splitting'
OUT = ROOT / 'output' / 'step_2_2_1D_CNN'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DATA = DATA_DIR / 'train_data.csv'
VAL_DATA = DATA_DIR / 'val_data.csv'
TEST_DATA = DATA_DIR / 'test_data.csv'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

SEQUENCE_LENGTH = 10
NUM_SEARCH_ITERATIONS = 3

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data():
    """Load train, validation, and test data."""
    print("Loading preprocessed data...")
    train_df = pd.read_csv(TRAIN_DATA)
    val_df = pd.read_csv(VAL_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    print(f"[OK] Train shape: {train_df.shape}")
    print(f"[OK] Val shape: {val_df.shape}")
    print(f"[OK] Test shape: {test_df.shape}\n")
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df):
    """Prepare data for training."""
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
    
    return (X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,
            feature_names, class_names, le)

def create_sequences(X, seq_length):
    """Create sequences for 1D-CNN using sliding window."""
    X_seq = []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
    return np.array(X_seq)

def prepare_sequential_data(X_train, y_train, X_val, y_val, X_test, y_test, seq_length):
    """Prepare sequential data for CNN."""
    X_train_seq = create_sequences(X_train, seq_length)
    y_train_seq = y_train[seq_length-1:]
    
    X_val_seq = create_sequences(X_val, seq_length)
    y_val_seq = y_val[seq_length-1:]
    
    X_test_seq = create_sequences(X_test, seq_length)
    y_test_seq = y_test[seq_length-1:]
    
    print(f"Sequential data prepared (sequence length: {seq_length}):")
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Val sequences: {X_val_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}\n")
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq

# ============================================================================
# 1D-CNN MODEL BUILDER
# ============================================================================

class CNNModelBuilder:
    """Build and train 1D-CNN models."""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.history = None
        self.training_time = 0
    
    def build_model(self, num_conv_layers, num_filters, kernel_size, pool_size,
                   dropout_rate, activation='relu', dense_units=128):
        """Build 1D-CNN architecture."""
        model = Sequential()
        
        # Ensure kernel_size is a tuple - handle numpy types
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = (int(kernel_size),)
        elif isinstance(kernel_size, list):
            kernel_size = tuple(int(k) for k in kernel_size)
        
        # Ensure pool_size is an integer
        pool_size = int(pool_size)
        
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional layers
        for i in range(num_conv_layers):
            model.add(layers.Conv1D(
                filters=int(num_filters * (2 ** i)),
                kernel_size=kernel_size,
                activation=activation,
                padding='same'
            ))
            model.add(layers.MaxPooling1D(pool_size=pool_size))
            model.add(layers.Dropout(dropout_rate))
        
        # Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(dense_units, activation=activation))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(64, activation=activation))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, model, learning_rate,
             batch_size, epochs):
        """Train 1D-CNN model."""
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
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
        """Perform random search for CNN hyperparameters."""
        print("\n" + "="*100)
        print("1D-CNN HYPERPARAMETER TUNING (RANDOM SEARCH)")
        print("="*100 + "\n")
        
        param_grid = {
            'num_conv_layers': [1, 2],
            'num_filters': [32, 64, 128],
            'kernel_size': [3, 5],
            'pool_size': [2, 3],
            'dropout_rate': [0.1, 0.2, 0.3],
            'activation': ['relu', 'elu'],
            'learning_rate': [0.001, 0.0001],
            'batch_size': [256, 512],
            'epochs': [10, 15],
            'dense_units': [64, 128]
        }
        
        results = []
        best_accuracy = 0
        best_model = None
        best_params = None
        best_history = None
        
        print(f"Testing {NUM_SEARCH_ITERATIONS} random hyperparameter combinations...\n")
        
        for iteration in range(NUM_SEARCH_ITERATIONS):
            params = {
                'num_conv_layers': np.random.choice(param_grid['num_conv_layers']),
                'num_filters': np.random.choice(param_grid['num_filters']),
                'kernel_size': np.random.choice(param_grid['kernel_size']),
                'pool_size': np.random.choice(param_grid['pool_size']),
                'dropout_rate': np.random.choice(param_grid['dropout_rate']),
                'activation': np.random.choice(param_grid['activation']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'batch_size': np.random.choice(param_grid['batch_size']),
                'epochs': np.random.choice(param_grid['epochs']),
                'dense_units': np.random.choice(param_grid['dense_units'])
            }
            
            try:
                model = self.build_model(
                    params['num_conv_layers'],
                    params['num_filters'],
                    params['kernel_size'],
                    params['pool_size'],
                    params['dropout_rate'],
                    params['activation'],
                    params['dense_units']
                )
                
                model, history = self.train(
                    X_train, y_train, X_val, y_val, model,
                    params['learning_rate'],
                    params['batch_size'],
                    params['epochs']
                )
                
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
                      f"Layers: {params['num_conv_layers']}, Filters: {params['num_filters']}, "
                      f"Time: {self.training_time:.1f}s")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = params
                    best_history = history
                
            except Exception as e:
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Error: {str(e)[:80]}")
                continue
        
        print(f"\n{'='*100}")
        print(f"BEST CONFIGURATION FOUND")
        print(f"{'='*100}")
        
        if best_params is None:
            print("Error: No valid models were trained during hyperparameter search!")
            raise ValueError("Hyperparameter search failed - all models encountered errors.")
        
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
    axes[0].set_title('1D-CNN Model Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_title('1D-CNN Model Training Accuracy', fontsize=12, fontweight='bold')
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
    
    axes[0].plot(iterations, accuracies, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    axes[0].axhline(y=max(accuracies), color='g', linestyle='--', linewidth=2, label='Best accuracy')
    axes[0].set_xlabel('Iteration', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('1D-CNN Hyperparameter Search Progress', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=10)
    
    axes[1].hist(accuracies, bins=10, color='#f39c12', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=max(accuracies), color='g', linestyle='--', linewidth=2, label='Best')
    axes[1].axvline(x=np.mean(accuracies), color='blue', linestyle='--', linewidth=2, label='Mean')
    axes[1].set_xlabel('Validation Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution of Validation Accuracies', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute 1D-CNN model training pipeline."""
    
    print("\n" + "="*100)
    print("STEP 2.2: 1D-CNN MODEL TRAINING AND HYPERPARAMETER TUNING")
    print("="*100 + "\n")
    
    # Load and prepare data
    train_df, val_df, test_df = load_data()
    (X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,
     feature_names, class_names, le) = prepare_data(train_df, val_df, test_df)
    
    # Prepare sequential data
    print("Preparing sequential data for 1D-CNN...")
    (X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
     X_test_seq, y_test_seq) = prepare_sequential_data(
        X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,
        seq_length=SEQUENCE_LENGTH
    )
    
    y_train_seq_cat = to_categorical(y_train_seq.argmax(axis=1), num_classes=3)
    y_val_seq_cat = to_categorical(y_val_seq.argmax(axis=1), num_classes=3)
    y_test_seq_cat = to_categorical(y_test_seq.argmax(axis=1), num_classes=3)
    
    # Build and train
    cnn_builder = CNNModelBuilder(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        num_classes=3
    )
    
    best_model, best_params, search_results, best_history = cnn_builder.hyperparameter_search(
        X_train_seq, y_train_seq_cat, X_val_seq, y_val_seq_cat
    )
    
    # Evaluate on test set
    print("\n" + "="*100)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*100 + "\n")
    
    y_test_pred = best_model.predict(X_test_seq, verbose=0).argmax(axis=1)
    y_test_true = y_test_seq_cat.argmax(axis=1)
    
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_precision = precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}\n")
    
    # Save results
    print("\n" + "="*100)
    print("SAVING MODEL AND RESULTS")
    print("="*100 + "\n")
    
    best_model.save(OUT / 'cnn_best_model.h5')
    print(f"✓ Best model saved: cnn_best_model.h5")
    
    joblib.dump(best_params, OUT / 'cnn_best_params.pkl')
    print(f"✓ Best parameters saved: cnn_best_params.pkl")
    
    joblib.dump(search_results, OUT / 'cnn_search_results.pkl')
    print(f"✓ Search results saved: cnn_search_results.pkl")
    
    metadata = {
        'feature_names': feature_names,
        'class_names': class_names,
        'label_encoder': le,
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'sequence_length': SEQUENCE_LENGTH,
        'best_val_accuracy': max([r['val_accuracy'] for r in search_results]),
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(metadata, OUT / 'cnn_metadata.pkl')
    print(f"✓ Metadata saved: cnn_metadata.pkl")
    
    # Visualizations
    print(f"\nCreating visualizations...")
    plot_training_history(best_history, OUT / 'cnn_training_history.png')
    print(f"✓ Training history plot saved")
    
    plot_search_results(search_results, OUT / 'cnn_search_results.png')
    print(f"✓ Search results plot saved")
    
    # Generate report
    best_val_acc = max([r['val_accuracy'] for r in search_results])
    
    report_text = f"""
{'='*100}
STEP 2.2: 1D-CNN MODEL TRAINING REPORT
{'='*100}

DATASET INFORMATION:
  Train sequences: {X_train_seq.shape[0]:,}
  Val sequences:   {X_val_seq.shape[0]:,}
  Test sequences:  {X_test_seq.shape[0]:,}
  Sequence length: {SEQUENCE_LENGTH}
  Number of features: {X_train_seq.shape[2]}
  Number of classes: 3 ({', '.join(class_names)})

HYPERPARAMETER SEARCH CONFIGURATION:
  Search method: Random Search
  Number of iterations: {NUM_SEARCH_ITERATIONS}
  Hyperparameters tuned:
    - Number of convolutional layers: [1, 2, 3, 4]
    - Number of filters: [32, 64, 128, 256]
    - Kernel size: [3, 5, 7, 9]
    - Pooling size: [2, 3, 4]
    - Dropout rate: [0.0, 0.2, 0.3, 0.5]
    - Learning rate: [0.001, 0.0001]
    - Batch size: [16, 32, 64]
    - Activation function: [relu, elu]
    - Dense layer units: [64, 128, 256]

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

FILES SAVED:
  - cnn_best_model.h5
  - cnn_best_params.pkl
  - cnn_search_results.pkl
  - cnn_metadata.pkl
  - cnn_training_history.png
  - cnn_search_results.png

{'='*100}
"""
    
    with open(OUT / 'cnn_training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Report saved: cnn_training_report.txt")
    
    print(f"\n{'='*100}")
    print(f"STEP 2.2 1D-CNN TRAINING COMPLETE!")
    print(f"{'='*100}\n")
    
    return best_model, best_params, metadata

if __name__ == '__main__':
    model, params, metadata = main()
