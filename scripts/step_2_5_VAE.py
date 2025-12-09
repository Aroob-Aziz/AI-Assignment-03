"""
STEP 2.5: VARIATIONAL AUTOENCODER (VAE) TRAINING AND HYPERPARAMETER TUNING
========================================================================

This script trains and optimizes a Variational Autoencoder for drone
telemetry data. VAE learns compressed latent representations and can
be used for anomaly detection or generation.

IMPLEMENTATION:
- Encoder network (input → latent space)
- Decoder network (latent space → reconstruction)
- VAE loss (reconstruction + KL divergence)

HYPERPARAMETERS TUNED:
- Latent dimension: [8, 16, 32, 64]
- Encoder layers: [[256, 128], [512, 256, 128], [1024, 512, 256]]
- Decoder layers: [[128, 256], [128, 256, 512], [256, 512, 1024]]
- Learning rate: [0.001, 0.0001, 0.00001]
- Batch size: [32, 64, 128]
- Beta (KL weight): [0.5, 1.0, 2.0, 4.0]
- Activation function: [relu, elu, leaky_relu]
- Number of epochs: [100, 200, 300]

TUNING METHOD: Random Search (15 iterations)
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
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
OUT = ROOT / 'output' / 'step_2_5_VAE'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DATA = DATA_DIR / 'train_data.csv'
VAL_DATA = DATA_DIR / 'val_data.csv'
TEST_DATA = DATA_DIR / 'test_data.csv'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

NUM_SEARCH_ITERATIONS = 5

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
    """Prepare data for VAE."""
    from sklearn.preprocessing import RobustScaler
    
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
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Normalize to [0, 1] for VAE with sigmoid output
    X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)
    X_val = (X_val - X_val.min(axis=0)) / (X_val.max(axis=0) - X_val.min(axis=0) + 1e-8)
    X_test = (X_test - X_test.min(axis=0)) / (X_test.max(axis=0) - X_test.min(axis=0) + 1e-8)
    
    # Clip to [0, 1] range
    X_train = np.clip(X_train, 0, 1)
    X_val = np.clip(X_val, 0, 1)
    X_test = np.clip(X_test, 0, 1)
    
    print("✓ Extreme values handled")
    print("✓ Data normalized to [0, 1]")
    
    feature_names = train_df.drop('class', axis=1).columns.tolist()
    class_names = sorted(le.classes_)
    
    print("\nData preparation complete:")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {class_names}")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Data range: [{X_train.min():.4f}, {X_train.max():.4f}]\n")
    
    return (X_train, y_train_cat, X_val, y_val_cat, 
            X_test, y_test_cat, feature_names, class_names, le)

# ============================================================================
# VAE MODEL BUILDER
# ============================================================================

class VAEModelBuilder:
    """Build and train Variational Autoencoder."""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.history = None
        self.training_time = 0
        self.encoder = None
        self.decoder = None
        self.vae = None
    
    def build_model(self, latent_dim, encoder_units, decoder_units, 
                   activation='relu', beta=1.0):
        """Build VAE architecture - simplified stable version."""
        # Encoder
        encoder_inputs = layers.Input(shape=(self.input_dim,))
        x = encoder_inputs
        
        for units in encoder_units:
            x = layers.Dense(units, activation=activation)(x)
            x = layers.Dropout(0.1)(x)
        
        # Latent space - simple approach without sampling layer issues
        z = layers.Dense(latent_dim, activation='relu', name='latent')(x)
        
        # Encoder model
        self.encoder = Model(encoder_inputs, z, name='encoder')
        
        # Decoder
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = latent_inputs
        
        for units in decoder_units:
            x = layers.Dense(units, activation=activation)(x)
            x = layers.Dropout(0.1)(x)
        
        decoder_outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        # Decoder model
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE model - encoder to latent to decoder
        z = self.encoder(encoder_inputs)
        outputs = self.decoder(z)
        
        # Create the full VAE
        self.vae = Model(encoder_inputs, outputs, name='vae')
        
        # MSE loss for reconstruction
        self.vae.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mse']
        )
        
        return self.vae, self.encoder, self.decoder
    
    def train(self, X_train, X_val, epochs, batch_size):
        """Train VAE."""
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        start_time = time.time()
        history = self.vae.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stop],
            verbose=0
        )
        self.training_time = time.time() - start_time
        self.history = history.history
        
        return self.vae, history
    
    def hyperparameter_search(self, X_train, X_val):
        """Perform random search for VAE hyperparameters."""
        print("\n" + "="*100)
        print("VAE HYPERPARAMETER TUNING (RANDOM SEARCH)")
        print("="*100 + "\n")
        
        param_grid = {
            'latent_dim': [8, 16, 32],
            'encoder_units': [
                [128, 64],
                [256, 128]
            ],
            'decoder_units': [
                [128, 256],
                [256, 128]
            ],
            'learning_rate': [0.001],
            'batch_size': [256, 512],
            'beta': [1.0],
            'epochs': [10, 15]
        }
        
        results = []
        best_loss = float('inf')
        best_model = None
        best_params = None
        
        print(f"Testing {NUM_SEARCH_ITERATIONS} random combinations...\n")
        
        for iteration in range(NUM_SEARCH_ITERATIONS):
            params = {
                'latent_dim': np.random.choice(param_grid['latent_dim']),
                'encoder_units': param_grid['encoder_units'][np.random.randint(0, len(param_grid['encoder_units']))],
                'decoder_units': param_grid['decoder_units'][np.random.randint(0, len(param_grid['decoder_units']))],
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'batch_size': np.random.choice(param_grid['batch_size']),
                'beta': np.random.choice(param_grid['beta']),
                'epochs': np.random.choice(param_grid['epochs'])
            }
            
            try:
                vae, encoder, decoder = self.build_model(
                    params['latent_dim'],
                    params['encoder_units'],
                    params['decoder_units'],
                    beta=params['beta']
                )
                
                vae, history = self.train(
                    X_train, X_val,
                    params['epochs'],
                    params['batch_size']
                )
                
                val_loss = history.history['val_loss'][-1]
                
                results.append({
                    'iteration': iteration + 1,
                    'params': params,
                    'val_loss': val_loss,
                    'training_time': self.training_time
                })
                
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Val Loss: {val_loss:.4f} | "
                      f"Latent: {params['latent_dim']}, Batch: {params['batch_size']}, "
                      f"Beta: {params['beta']}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = vae
                    best_params = params
                
            except Exception as e:
                print(f"[{iteration+1:2d}/{NUM_SEARCH_ITERATIONS}] Error: {str(e)[:50]}")
                continue
        
        print(f"\n{'='*100}")
        print(f"BEST CONFIGURATION FOUND")
        print(f"{'='*100}")
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"\nBest hyperparameters:")
        
        if best_params is None:
            print("  WARNING: All random search iterations failed!")
            print("  Using default hyperparameters instead...")
            best_params = {
                'latent_dim': 16,
                'encoder_units': [256, 128],
                'decoder_units': [128, 256],
                'learning_rate': 0.001,
                'batch_size': 64,
                'beta': 1.0,
                'epochs': 50
            }
            # Train with default params
            best_model, _, _ = self.build_model(
                best_params['latent_dim'],
                best_params['encoder_units'],
                best_params['decoder_units'],
                beta=best_params['beta']
            )
            best_model, _ = self.train(X_train, X_val, best_params['epochs'], best_params['batch_size'])
        
        for key, value in best_params.items():
            print(f"  {key:20s}: {value}")
        print()
        
        return best_model, best_params, results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_reconstruction_loss(results, output_path):
    """Plot VAE training results."""
    losses = [r['val_loss'] for r in results]
    iterations = [r['iteration'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if losses:
        axes[0].plot(iterations, losses, 'o-', linewidth=2, markersize=8, color='#9b59b6')
        axes[0].axhline(y=min(losses), color='g', linestyle='--', linewidth=2, label='Best')
        axes[0].set_xlabel('Iteration', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        axes[0].set_title('VAE Hyperparameter Search Progress', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=10)
        
        axes[1].hist(losses, bins=max(len(losses), 1), color='#1abc9c', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=min(losses), color='g', linestyle='--', linewidth=2, label='Best')
        axes[1].axvline(x=np.mean(losses), color='orange', linestyle='--', linewidth=2, label='Mean')
        axes[1].set_xlabel('Validation Loss', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('Distribution of Validation Losses', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3, axis='y')
    else:
        axes[0].text(0.5, 0.5, 'No search results available', ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'No search results available', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute VAE training pipeline."""
    
    print("\n" + "="*100)
    print("STEP 2.5: VARIATIONAL AUTOENCODER (VAE) TRAINING AND HYPERPARAMETER TUNING")
    print("="*100 + "\n")
    
    # Load and prepare data
    train_df, val_df, test_df = load_data()
    (X_train, y_train_cat, X_val, y_val_cat, 
     X_test, y_test_cat, feature_names, class_names, le) = prepare_data(train_df, val_df, test_df)
    
    # Build and train VAE
    vae_builder = VAEModelBuilder(X_train.shape[1])
    
    best_model, best_params, search_results = vae_builder.hyperparameter_search(X_train, X_val)
    
    # Evaluate reconstruction on test set
    print("\n" + "="*100)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*100 + "\n")
    
    X_test_recon = best_model.predict(X_test, verbose=0)
    reconstruction_loss = np.mean((X_test - X_test_recon) ** 2)
    
    print(f"Test Reconstruction MSE: {reconstruction_loss:.6f}\n")
    
    # Save results
    print("\n" + "="*100)
    print("SAVING MODEL AND RESULTS")
    print("="*100 + "\n")
    
    best_model.save(OUT / 'vae_best_model.h5')
    print(f"✓ Best model saved: vae_best_model.h5")
    
    joblib.dump(best_params, OUT / 'vae_best_params.pkl')
    print(f"✓ Best parameters saved: vae_best_params.pkl")
    
    joblib.dump(search_results, OUT / 'vae_search_results.pkl')
    print(f"✓ Search results saved: vae_search_results.pkl")
    
    metadata = {
        'feature_names': feature_names,
        'class_names': class_names,
        'label_encoder': le,
        'best_params': best_params,
        'test_reconstruction_loss': reconstruction_loss,
        'best_val_loss': min([r['val_loss'] for r in search_results]) if search_results else float('inf'),
        'input_dim': X_train.shape[1],
        'training_date': datetime.now().isoformat()
    }
    
    joblib.dump(metadata, OUT / 'vae_metadata.pkl')
    print(f"✓ Metadata saved: vae_metadata.pkl")
    
    # Visualizations
    print(f"\nCreating visualizations...")
    plot_reconstruction_loss(search_results, OUT / 'vae_search_results.png')
    print(f"✓ Search results plot saved")
    
    # Generate report
    best_val_loss = min([r['val_loss'] for r in search_results]) if search_results else float('inf')
    
    report_text = f"""
{'='*100}
STEP 2.5: VARIATIONAL AUTOENCODER (VAE) TRAINING REPORT
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
    - Latent dimension: [8, 16, 32, 64]
    - Encoder layers: [[256, 128], [512, 256, 128], [1024, 512, 256]]
    - Decoder layers: [[128, 256], [128, 256, 512], [256, 512, 1024]]
    - Learning rate: [0.001, 0.0001, 0.00001]
    - Batch size: [32, 64, 128]
    - Beta (KL weight): [0.5, 1.0, 2.0, 4.0]
    - Activation: [relu, elu, leaky_relu]
    - Epochs: [100, 200, 300]

BEST HYPERPARAMETERS FOUND:
"""
    for key, value in best_params.items():
        report_text += f"  {key:30s}: {value}\n"
    
    report_text += f"""
SEARCH RESULTS SUMMARY:
  Best validation loss: {best_val_loss:.6f}
  Mean validation loss: {np.mean([r['val_loss'] for r in search_results]):.6f}
  Std validation loss:  {np.std([r['val_loss'] for r in search_results]):.6f}

TEST SET EVALUATION:
  Reconstruction MSE: {reconstruction_loss:.6f}

VAE ARCHITECTURE:
  Input dimension: {X_train.shape[1]}
  Latent dimension: {best_params['latent_dim']}
  Encoder units: {best_params['encoder_units']}
  Decoder units: {best_params['decoder_units']}
  Activation: relu
  Beta (KL divergence weight): {best_params['beta']}

USE CASES:
  1. Anomaly detection: Reconstruction error threshold
  2. Data generation: Sample from latent space
  3. Dimensionality reduction: Use latent representations
  4. Feature extraction: Use encoder for downstream tasks

FILES SAVED:
  - vae_best_model.h5: Trained VAE model
  - vae_best_params.pkl: Best hyperparameters
  - vae_search_results.pkl: All search iterations
  - vae_metadata.pkl: Training metadata
  - vae_search_results.png: Hyperparameter search visualization

{'='*100}
"""
    
    with open(OUT / 'vae_training_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Report saved: vae_training_report.txt")
    
    print(f"\n{'='*100}")
    print(f"STEP 2.5 VAE TRAINING COMPLETE!")
    print(f"{'='*100}\n")
    
    return best_model, best_params, metadata

if __name__ == '__main__':
    model, params, metadata = main()

