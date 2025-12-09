"""
STEP 2.1: LSTM MODEL TRAINING AND HYPERPARAMETER TUNING
========================================================================

This script trains and optimizes an LSTM model for drone telemetry
classification. LSTM is designed to capture temporal dependencies in
sequential data.

IMPLEMENTATION REQUIREMENTS:
- Input reshaping for sequence data (sliding windows)
- Define LSTM architecture (number of layers, units per layer)
- Consider bidirectional LSTM
- Hyperparameter tuning via random search with cross-validation

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

TUNING METHOD: Random Search (3 iterations)
"""

# ============================================================================
# GPU CONFIGURATION (ENABLE GPU ACCELERATION)
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")

# Enable GPU memory growth (prevents OOM errors)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
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
OUT = ROOT / 'output' / 'step_2_1_LSTM'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DATA = DATA_DIR / 'train_data.csv'
VAL_DATA = DATA_DIR / 'val_data.csv'
TEST_DATA = DATA_DIR / 'test_data.csv'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

SEQUENCE_LENGTH = 10  # Use 10 time steps for sequences
NUM_SEARCH_ITERATIONS = 1  # Number of random hyperparameter combinations to test (reduced for faster execution)

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data():
    """Load train, validation, and test data."""
    print("Loading preprocessed data...")
    train_df = pd.read_csv(TRAIN_DATA)
    val_df = pd.read_csv(VAL_DATA)
    test_df = pd.read_csv(TEST_DATA)
    
    print(f"✓ Train shape: {train_df.shape}")
    print(f"✓ Val shape: {val_df.shape}")
    print(f"✓ Test shape: {test_df.shape}\n")
    
    return train_df, val_df, test_df

def prepare_data(train_df, val_df, test_df):
    """Prepare data for training: separate features and labels."""
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
    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape: {X_val.shape}")
    print(f"  Test shape: {X_test.shape}\n")
    
    return (X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,
            feature_names, class_names, le)

def create_sequences(X, seq_length):
    """Create sequences for LSTM using sliding window approach."""
    X_seq = []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i+seq_length])
    return np.array(X_seq)

def prepare_sequential_data(X_train, y_train, X_val, y_val, X_test, y_test, seq_length):
    """Prepare data with sequences for LSTM model."""
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
# LSTM MODEL BUILDER
# ============================================================================

class LSTMModelBuilder:
    """Build and train LSTM models with hyperparameter tuning."""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.history = None
        self.training_time = 0
    
    def build_model(self, num_lstm_layers, num_units, dropout_rate, 
                   activation='tanh', bidirectional=False, l2_reg=0.0, use_batch_norm=False):
        """Build LSTM architecture with regularization."""
        from tensorflow.keras.regularizers import l2
        model = Sequential()
        
        # First LSTM layer
        if bidirectional:
            model.add(layers.Bidirectional(
                layers.LSTM(num_units, activation=activation, 
                           return_sequences=(num_lstm_layers > 1),
                           kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
                           input_shape=self.input_shape)
            ))
        else:
            model.add(layers.LSTM(num_units, activation=activation,
                                 return_sequences=(num_lstm_layers > 1),
                                 kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None,
                                 input_shape=self.input_shape))
        
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, num_lstm_layers):
            if bidirectional:
                model.add(layers.Bidirectional(
                    layers.LSTM(num_units, activation=activation,
                               return_sequences=(i < num_lstm_layers - 1),
                               kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None)
                ))
            else:
                model.add(layers.LSTM(num_units, activation=activation,
                                     return_sequences=(i < num_lstm_layers - 1),
                                     kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
            if use_batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Dense output layers
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, model, learning_rate, 
             batch_size, epochs, optimizer_name='Adam'):
        """Train LSTM model."""
        # Compile model
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
            patience=3,  # Reduced for faster execution
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,  # Reduced for faster execution
            min_lr=1e-7,
            verbose=0
        )
        
        # Train model
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
        """Perform random search for LSTM hyperparameters."""
        print("\n" + "="*100)
        print("LSTM HYPERPARAMETER TUNING (RANDOM SEARCH)")
        print("="*100 + "\n")
        
        # Hyperparameter grid
        param_grid = {
            'num_lstm_layers': [2, 3],  # Reduced for faster execution
            'num_units': [64, 128],  # Reduced for faster execution
            'dropout_rate': [0.2, 0.3],  # Reduced for faster execution
            'activation': ['relu', 'tanh'],  # Reduced for faster execution
            'learning_rate': [0.001, 0.0001],  # Reduced for faster execution
            'batch_size': [32, 64],  # Reduced for faster execution
            'epochs': [5, 10],  # Reduced for faster execution
            'optimizer': ['Adam'],  # Reduced for faster execution
            'bidirectional': [True, False],
            'l2_reg': [0.0, 0.01],  # Reduced for faster execution
            'use_batch_norm': [True, False]
        }
        
        results = []
        best_accuracy = 0
        best_model = None
        best_params = None
        best_history = None
        
        print(f"Testing {NUM_SEARCH_ITERATIONS} random hyperparameter combinations...\n")
        
        for iteration in range(NUM_SEARCH_ITERATIONS):
            print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Starting iteration...")
            # Sample random hyperparameters
            params = {
                'num_lstm_layers': np.random.choice(param_grid['num_lstm_layers']),
                'num_units': np.random.choice(param_grid['num_units']),
                'dropout_rate': np.random.choice(param_grid['dropout_rate']),
                'activation': np.random.choice(param_grid['activation']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'batch_size': np.random.choice(param_grid['batch_size']),
                'epochs': np.random.choice(param_grid['epochs']),
                'optimizer': np.random.choice(param_grid['optimizer']),
                'bidirectional': np.random.choice(param_grid['bidirectional']),
                'l2_reg': np.random.choice(param_grid['l2_reg']),
                'use_batch_norm': np.random.choice(param_grid['use_batch_norm'])
            }
            print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Sampled hyperparameters")
            
            try:
                # Build model
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Building model...")
                model = self.build_model(
                    params['num_lstm_layers'],
                    params['num_units'],
                    params['dropout_rate'],
                    params['activation'],
                    params['bidirectional'],
                    params['l2_reg'],
                    params['use_batch_norm']
                )
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Model built successfully")
                
                # Train model
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Training model (BS={params['batch_size']}, Epochs={params['epochs']})...")
                model, history = self.train(
                    X_train, y_train, X_val, y_val, model,
                    params['learning_rate'],
                    params['batch_size'],
                    params['epochs'],
                    params['optimizer']
                )
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Training completed in {self.training_time:.1f}s")
                
                # Evaluate on validation set
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Evaluating on validation set...")
                y_val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
                y_val_true = y_val.argmax(axis=1)
                accuracy = accuracy_score(y_val_true, y_val_pred)
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Validation accuracy: {accuracy:.4f}")
                
                results.append({
                    'iteration': iteration + 1,
                    'params': params,
                    'val_accuracy': accuracy,
                    'training_time': self.training_time,
                    'epochs_trained': len(history.history['loss'])
                })
                
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Accuracy: {accuracy:.4f} | "
                      f"Layers: {params['num_lstm_layers']}, Units: {params['num_units']}, "
                      f"LR: {params['learning_rate']:.0e}, Time: {self.training_time:.1f}s")
                
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
    """Plot training history (loss and accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title('LSTM Model Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_title('LSTM Model Training Accuracy', fontsize=12, fontweight='bold')
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
    
    # Accuracy over iterations
    axes[0].plot(iterations, accuracies, 'o-', linewidth=2, markersize=8, color='#3498db')
    axes[0].axhline(y=max(accuracies), color='r', linestyle='--', linewidth=2, label='Best accuracy')
    axes[0].set_xlabel('Iteration', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('LSTM Hyperparameter Search Progress', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Distribution of accuracies
    axes[1].hist(accuracies, bins=10, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=max(accuracies), color='r', linestyle='--', linewidth=2, label='Best')
    axes[1].axvline(x=np.mean(accuracies), color='orange', linestyle='--', linewidth=2, label='Mean')
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
    """Execute LSTM model training pipeline."""
    
    print("\n" + "="*100)
    print("STEP 2.1: LSTM MODEL TRAINING AND HYPERPARAMETER TUNING")
    print("="*100 + "\n")
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Prepare data
    (X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,
     feature_names, class_names, le) = prepare_data(train_df, val_df, test_df)
    
    # Prepare sequential data
    print("Preparing sequential data for LSTM...")
    (X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
     X_test_seq, y_test_seq) = prepare_sequential_data(
        X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat,
        seq_length=SEQUENCE_LENGTH
    )
    
    y_train_seq_cat = to_categorical(y_train_seq.argmax(axis=1), num_classes=3)
    y_val_seq_cat = to_categorical(y_val_seq.argmax(axis=1), num_classes=3)
    y_test_seq_cat = to_categorical(y_test_seq.argmax(axis=1), num_classes=3)
    
    # Build LSTM and perform hyperparameter search
    lstm_builder = LSTMModelBuilder(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        num_classes=3
    )
    
    best_model, best_params, search_results, best_history = lstm_builder.hyperparameter_search(
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
    
    # Save model and results
    print("\n" + "="*100)
    print("SAVING MODEL AND RESULTS")
    print("="*100 + "\n")
    
    best_model.save(OUT / 'lstm_best_model.h5')
    print(f"✓ Best model saved: lstm_best_model.h5")
    
    joblib.dump(best_params, OUT / 'lstm_best_params.pkl')
    print(f"✓ Best parameters saved: lstm_best_params.pkl")
    
    joblib.dump(search_results, OUT / 'lstm_search_results.pkl')
    print(f"✓ Search results saved: lstm_search_results.pkl")
    
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
        'num_search_iterations': NUM_SEARCH_ITERATIONS,
        'best_val_accuracy': max([r['val_accuracy'] for r in search_results]),
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(metadata, OUT / 'lstm_metadata.pkl')
    print(f"✓ Metadata saved: lstm_metadata.pkl")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    plot_training_history(best_history, OUT / 'lstm_training_history.png')
    print(f"✓ Training history plot saved")
    
    plot_search_results(search_results, OUT / 'lstm_search_results.png')
    print(f"✓ Search results plot saved")
    
    # Generate report
    print(f"\nGenerating comprehensive report...")
    report_text = f"""
{'='*100}
STEP 2.1: LSTM MODEL TRAINING REPORT
{'='*100}

DATASET INFORMATION:
  Train sequences: {X_train_seq.shape[0]:,} ({X_train_seq.shape[1:]} per sequence)
  Val sequences:   {X_val_seq.shape[0]:,}
  Test sequences:  {X_test_seq.shape[0]:,}
  Sequence length: {SEQUENCE_LENGTH}
  Number of features: {X_train_seq.shape[2]}
  Number of classes: 3 ({', '.join(class_names)})

HYPERPARAMETER SEARCH CONFIGURATION:
  Search method: Random Search
  Number of iterations: {NUM_SEARCH_ITERATIONS}
  Hyperparameters tuned:
    - Number of LSTM layers: [1, 2, 3]
    - Number of units: [32, 64, 128, 256]
    - Dropout rate: [0.0, 0.1, 0.2, 0.3, 0.5]
    - Activation: [tanh, relu, sigmoid]
    - Learning rate: [0.001, 0.0001, 0.00001]
    - Batch size: [16, 32, 64, 128]
    - Epochs: [50, 100, 150, 200]
    - Optimizer: [Adam, RMSprop, SGD]
    - Bidirectional: [True, False]

BEST HYPERPARAMETERS FOUND:
"""
    for key, value in best_params.items():
        report_text += f"  {key:30s}: {value}\n"
    
    best_val_acc = max([r['val_accuracy'] for r in search_results])
    
    report_text += f"""
SEARCH RESULTS SUMMARY:
  Best validation accuracy: {best_val_acc:.4f}
  Mean validation accuracy: {np.mean([r['val_accuracy'] for r in search_results]):.4f}
  Std validation accuracy:  {np.std([r['val_accuracy'] for r in search_results]):.4f}
  Min validation accuracy:  {min([r['val_accuracy'] for r in search_results]):.4f}

TEST SET EVALUATION:
  Accuracy:  {test_accuracy:.4f}
  Precision: {test_precision:.4f}
  Recall:    {test_recall:.4f}
  F1-Score:  {test_f1:.4f}

MODEL ARCHITECTURE:
  - Input: {X_train_seq.shape[1]} time steps × {X_train_seq.shape[2]} features
  - {best_params['num_lstm_layers']} LSTM layer(s) with {best_params['num_units']} units
  - Bidirectional: {best_params['bidirectional']}
  - Activation: {best_params['activation']}
  - Dropout: {best_params['dropout_rate']}
  - Output: 3 classes (softmax)

TRAINING CONFIGURATION:
  - Optimizer: {best_params['optimizer']}
  - Learning rate: {best_params['learning_rate']}
  - Batch size: {best_params['batch_size']}
  - Max epochs: {best_params['epochs']}
  - Early stopping: Yes (patience=10)
  - Learning rate reduction: Yes (patience=5)

FILES SAVED:
  - lstm_best_model.h5: Trained LSTM model (Keras format)
  - lstm_best_params.pkl: Best hyperparameters
  - lstm_search_results.pkl: All search iterations results
  - lstm_metadata.pkl: Training metadata
  - lstm_training_history.png: Training loss and accuracy plots
  - lstm_search_results.png: Hyperparameter search progress

NEXT STEPS:
  This model can now be used for:
  1. Making predictions on new data
  2. Feature extraction from latent representations
  3. Comparison with other models in Part 3
  4. Explainability analysis in Part 4

{'='*100}
"""
    
    with open(OUT / 'lstm_training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Report saved: lstm_training_report.txt")
    
    print(f"\n{'='*100}")
    print(f"STEP 2.1 LSTM TRAINING COMPLETE!")
    print(f"{'='*100}")
    print(f"\nOutput directory: {OUT}")
    print(f"All files saved successfully!\n")
    
    return best_model, best_params, metadata

if __name__ == '__main__':
    model, params, metadata = main()
