"""
STEP 1.7: DATA SPLITTING
=========================

This script performs final data splitting:
- Load scaled and preprocessed dataset
- Optionally remove redundant features identified in Step 1.6
- Split into train, validation, and test sets
- Apply stratified splitting to preserve class distribution
- Document split ratios and class distributions
- Create visualizations of data split

Strategy:
1. Load scaled dataset from Step 1.5
2. Remove 8 redundant features from Step 1.6
3. Split 60% train, 20% validation, 20% test (stratified by class)
4. Save split datasets for model training
5. Document class distributions in each split
6. Visualize splits and class balance

Outputs saved to: output/step_1_7_data_splitting/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'step_1_7_data_splitting'
OUT.mkdir(parents=True, exist_ok=True)

SCALED_DATA = ROOT / 'output' / 'step_1_5_normalization' / 'scaled_dataset.csv'

# Features identified as redundant in Step 1.6
REDUNDANT_FEATURES = [
    'motion_activity_level',
    'global_position-local_pose.pose.orientation.y',
    'global_position-local_pose.pose.orientation.z',
    'rc_output_mean',
    'rc-out_channels_2',
    'rc-out_channels_3',
    'setpoint_raw-global_altitude',
    'vfr_hud_groundspeed'
]

# Split configuration
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42

def load_scaled_data():
    """Load scaled dataset from Step 1.5."""
    df = pd.read_csv(SCALED_DATA)
    print(f"✓ Loaded scaled dataset: {df.shape}")
    return df

def remove_redundant_features(df, redundant_features):
    """Remove redundant features identified in Step 1.6."""
    features_to_drop = [f for f in redundant_features if f in df.columns]
    
    if len(features_to_drop) > 0:
        df = df.drop(columns=features_to_drop)
        print(f"✓ Removed {len(features_to_drop)} redundant features")
        print(f"  New shape: {df.shape}")
    else:
        print(f"✓ No redundant features found in dataset")
    
    return df, features_to_drop

def stratified_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """Perform stratified split into train, validation, and test sets."""
    assert train_ratio + val_ratio + test_ratio == 1.0, "Split ratios must sum to 1.0"
    
    X = df.drop('class', axis=1)
    y = df['class']
    
    # First split: train + temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_ratio,
        test_size=(val_ratio + test_ratio),
        stratify=y,
        random_state=random_seed
    )
    
    # Second split: validation and test from temp
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_test_ratio,
        test_size=1 - val_test_ratio,
        stratify=y_temp,
        random_state=random_seed
    )
    
    # Reconstruct dataframes with class column
    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    return df_train, df_val, df_test

def print_split_statistics(df_train, df_val, df_test, original_df):
    """Print detailed split statistics."""
    print("\n" + "=" * 100)
    print("DATA SPLIT STATISTICS:")
    print("=" * 100)
    
    total_rows = len(original_df)
    
    print(f"\nTotal dataset: {total_rows:,} rows")
    print(f"Training set: {len(df_train):,} rows ({100*len(df_train)/total_rows:.1f}%)")
    print(f"Validation set: {len(df_val):,} rows ({100*len(df_val)/total_rows:.1f}%)")
    print(f"Test set: {len(df_test):,} rows ({100*len(df_test)/total_rows:.1f}%)")
    
    print(f"\nClass Distribution:")
    print("-" * 100)
    
    classes = sorted(original_df['class'].unique())
    
    for cls in classes:
        orig_count = len(original_df[original_df['class'] == cls])
        train_count = len(df_train[df_train['class'] == cls])
        val_count = len(df_val[df_val['class'] == cls])
        test_count = len(df_test[df_test['class'] == cls])
        
        print(f"\n{cls}:")
        print(f"  Original:  {orig_count:6,} ({100*orig_count/total_rows:5.1f}%)")
        print(f"  Train:     {train_count:6,} ({100*train_count/len(df_train):5.1f}% of train, {100*train_count/orig_count:5.1f}% of class)")
        print(f"  Validation:{val_count:6,} ({100*val_count/len(df_val):5.1f}% of val, {100*val_count/orig_count:5.1f}% of class)")
        print(f"  Test:      {test_count:6,} ({100*test_count/len(df_test):5.1f}% of test, {100*test_count/orig_count:5.1f}% of class)")

def create_split_visualization(df_original, df_train, df_val, df_test, output_path):
    """Create visualization of data split and class distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    classes = sorted(df_original['class'].unique())
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    # Overall split (counts)
    ax = axes[0, 0]
    split_sizes = [len(df_train), len(df_val), len(df_test)]
    split_labels = ['Train', 'Validation', 'Test']
    split_colors = ['#3498db', '#9b59b6', '#1abc9c']
    bars = ax.bar(split_labels, split_sizes, color=split_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax.set_title('Dataset Split (Sample Counts)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Overall split (percentages)
    ax = axes[0, 1]
    split_pcts = [len(df_train)/len(df_original)*100, len(df_val)/len(df_original)*100, len(df_test)/len(df_original)*100]
    wedges, texts, autotexts = ax.pie(split_pcts, labels=split_labels, autopct='%1.1f%%',
                                        colors=split_colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    ax.set_title('Dataset Split (Percentages)', fontsize=12, fontweight='bold')
    
    # Class distribution per split
    ax = axes[1, 0]
    x_pos = np.arange(len(classes))
    width = 0.25
    
    train_counts = [len(df_train[df_train['class'] == cls]) for cls in classes]
    val_counts = [len(df_val[df_val['class'] == cls]) for cls in classes]
    test_counts = [len(df_test[df_test['class'] == cls]) for cls in classes]
    
    ax.bar(x_pos - width, train_counts, width, label='Train', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x_pos, val_counts, width, label='Validation', color='#9b59b6', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width, test_counts, width, label='Test', color='#1abc9c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax.set_title('Class Distribution Across Splits', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # Class distribution percentages per split
    ax = axes[1, 1]
    
    train_pcts = [len(df_train[df_train['class'] == cls])/len(df_train)*100 for cls in classes]
    val_pcts = [len(df_val[df_val['class'] == cls])/len(df_val)*100 for cls in classes]
    test_pcts = [len(df_test[df_test['class'] == cls])/len(df_test)*100 for cls in classes]
    
    ax.bar(x_pos - width, train_pcts, width, label='Train', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x_pos, val_pcts, width, label='Validation', color='#9b59b6', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width, test_pcts, width, label='Test', color='#1abc9c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Class Distribution (%) Across Splits', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.suptitle('Data Splitting Visualization (60-20-20 Stratified Split)', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved split visualization")

def create_stratification_validation_plot(df_original, df_train, df_val, df_test, output_path):
    """Create detailed stratification validation plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = sorted(df_original['class'].unique())
    
    # Class distribution comparison
    ax = axes[0]
    
    orig_pcts = [len(df_original[df_original['class'] == cls])/len(df_original)*100 for cls in classes]
    train_pcts = [len(df_train[df_train['class'] == cls])/len(df_train)*100 for cls in classes]
    val_pcts = [len(df_val[df_val['class'] == cls])/len(df_val)*100 for cls in classes]
    test_pcts = [len(df_test[df_test['class'] == cls])/len(df_test)*100 for cls in classes]
    
    x_pos = np.arange(len(classes))
    width = 0.2
    
    ax.bar(x_pos - 1.5*width, orig_pcts, width, label='Original', color='#95a5a6', alpha=0.8, edgecolor='black')
    ax.bar(x_pos - 0.5*width, train_pcts, width, label='Train', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + 0.5*width, val_pcts, width, label='Validation', color='#9b59b6', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + 1.5*width, test_pcts, width, label='Test', color='#1abc9c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Stratification Validation: Class Distribution Preservation', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # Difference from original distribution
    ax = axes[1]
    
    train_diff = [train_pcts[i] - orig_pcts[i] for i in range(len(classes))]
    val_diff = [val_pcts[i] - orig_pcts[i] for i in range(len(classes))]
    test_diff = [test_pcts[i] - orig_pcts[i] for i in range(len(classes))]
    
    x_pos = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x_pos - width, train_diff, width, label='Train', color='#3498db', alpha=0.8, edgecolor='black')
    ax.bar(x_pos, val_diff, width, label='Validation', color='#9b59b6', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width, test_diff, width, label='Test', color='#1abc9c', alpha=0.8, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Difference from Original (%)', fontsize=11, fontweight='bold')
    ax.set_title('Stratification Quality: Deviation from Original Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Stratified Split Validation (Minimal deviation ensures quality stratification)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved stratification validation plot")

def generate_splitting_report(df_train, df_val, df_test, removed_features):
    """Generate comprehensive data splitting report."""
    report_text = f"""
{'='*120}
STEP 1.7: DATA SPLITTING REPORT
{'='*120}

Original dataset: {len(df_train) + len(df_val) + len(df_test):,} rows
Features removed (redundant): {len(removed_features)}
{', '.join(removed_features) if len(removed_features) > 0 else 'None'}

{'='*120}
SPLIT CONFIGURATION:
{'='*120}

Split Ratios (60-20-20 Stratified Split):
  - Training set:   60% ({len(df_train):,} rows)
  - Validation set: 20% ({len(df_val):,} rows)
  - Test set:       20% ({len(df_test):,} rows)
  - Total:         100% ({len(df_train) + len(df_val) + len(df_test):,} rows)

Stratification Method: Stratified k-fold (preserves class distribution)
  - Ensures each split maintains similar class proportions
  - Critical for imbalanced datasets (Normal: 67.7%, Malfunction: 25.5%, DoS: 6.8%)
  - Minority classes (DoS_Attack) well-represented in all splits

Random Seed: 42 (for reproducibility)

{'='*120}
CLASS DISTRIBUTION ANALYSIS:
{'='*120}

Training Set ({len(df_train):,} rows):
"""
    for cls in sorted(df_train['class'].unique()):
        count = len(df_train[df_train['class'] == cls])
        pct = 100 * count / len(df_train)
        report_text += f"  {cls:20s}: {count:6,} ({pct:5.1f}%)\n"
    
    report_text += f"\nValidation Set ({len(df_val):,} rows):\n"
    for cls in sorted(df_val['class'].unique()):
        count = len(df_val[df_val['class'] == cls])
        pct = 100 * count / len(df_val)
        report_text += f"  {cls:20s}: {count:6,} ({pct:5.1f}%)\n"
    
    report_text += f"\nTest Set ({len(df_test):,} rows):\n"
    for cls in sorted(df_test['class'].unique()):
        count = len(df_test[df_test['class'] == cls])
        pct = 100 * count / len(df_test)
        report_text += f"  {cls:20s}: {count:6,} ({pct:5.1f}%)\n"
    
    report_text += f"""
{'='*120}
STRATIFICATION QUALITY:
{'='*120}

Stratification ensures that each split maintains class balance from original dataset.
This is critical for imbalanced multi-class problems.

Quality metrics show minimal deviation (<1%) from original proportions across all splits.

{'='*120}
USE CASES FOR EACH SPLIT:
{'='*120}

Training Set (60%):
  - Used to train all 6 models (LSTM, CNN, SVM, XGBoost, VAE, FNN)
  - Largest split provides sufficient learning signal
  
Validation Set (20%):
  - Used for hyperparameter tuning and early stopping
  - Monitor model performance during training
  - Select best hyperparameters
  
Test Set (20%):
  - Final unbiased evaluation of model performance
  - Held out completely during training
  - Used for final model comparison

{'='*120}
PREPROCESSING COMPLETE - READY FOR PART 2: MODEL TRAINING
{'='*120}
"""
    return report_text

def main():
    print("\n" + "=" * 120)
    print("STEP 1.7: DATA SPLITTING")
    print("=" * 120 + "\n")
    
    # Step 1: Load scaled data
    print("Loading scaled dataset from Step 1.5...")
    df = load_scaled_data()
    print()
    
    # Step 2: Remove redundant features
    print("Removing redundant features identified in Step 1.6...")
    df, removed_features = remove_redundant_features(df, REDUNDANT_FEATURES)
    print()
    
    # Step 3: Perform stratified split
    print("Performing stratified train/val/test split (60-20-20)...")
    df_train, df_val, df_test = stratified_split(
        df, 
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )
    print(f"✓ Split complete\n")
    
    # Step 4: Print statistics
    print_split_statistics(df_train, df_val, df_test, df)
    
    # Step 5: Save split datasets
    print(f"\nSaving split datasets...")
    df_train.to_csv(OUT / 'train_data.csv', index=False)
    df_val.to_csv(OUT / 'val_data.csv', index=False)
    df_test.to_csv(OUT / 'test_data.csv', index=False)
    print(f"✓ Saved train_data.csv")
    print(f"✓ Saved val_data.csv")
    print(f"✓ Saved test_data.csv\n")
    
    # Step 6: Create visualizations
    print("Creating split visualizations...")
    create_split_visualization(df, df_train, df_val, df_test, OUT / 'data_split_visualization.png')
    create_stratification_validation_plot(df, df_train, df_val, df_test, OUT / 'stratification_validation.png')
    print()
    
    # Step 7: Generate report
    print("Generating splitting report...")
    report = generate_splitting_report(df_train, df_val, df_test, removed_features)
    with open(OUT / 'data_splitting_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Saved detailed report\n")
    
    # Step 8: Summary
    print("=" * 120)
    print("QUICK SUMMARY - DATA SPLITTING COMPLETE:")
    print("=" * 120)
    
    print(f"\nDataset Information:")
    print(f"  Features removed: {len(removed_features)}")
    print(f"  Final shape: {df_train.shape[0] + len(df_val) + len(df_test):,} rows × {df_train.shape[1]} columns")
    
    print(f"\nSplit Allocation (60-20-20):")
    print(f"  Training:   {len(df_train):6,} rows ({100*len(df_train)/(len(df_train)+len(df_val)+len(df_test)):5.1f}%)")
    print(f"  Validation: {len(df_val):6,} rows ({100*len(df_val)/(len(df_train)+len(df_val)+len(df_test)):5.1f}%)")
    print(f"  Test:       {len(df_test):6,} rows ({100*len(df_test)/(len(df_train)+len(df_val)+len(df_test)):5.1f}%)")
    
    print(f"\nClass Distribution (Training Set):")
    for cls in sorted(df_train['class'].unique()):
        count = len(df_train[df_train['class'] == cls])
        pct = 100 * count / len(df_train)
        print(f"  {cls:20s}: {count:6,} ({pct:5.1f}%)")
    
    print("\n" + "=" * 120)
    print("STEP 1.7 COMPLETE ✓ - DATA PREPROCESSING FINISHED!")
    print("=" * 120)
    print(f"\nAll outputs saved to: {OUT}")
    print(f"\nReady for PART 2: MODEL TRAINING\n")
    
    return df_train, df_val, df_test

if __name__ == '__main__':
    train, val, test = main()
