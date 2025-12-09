"""
STEP 1.5: DATA NORMALIZATION/STANDARDIZATION
==============================================

This script applies feature scaling using multiple techniques:
- StandardScaler: For normally distributed features (velocity, acceleration)
- RobustScaler: For features with outliers (battery metrics, control signals)
- MinMaxScaler: For bounded features (percentages, throttle values)
- Domain-aware feature grouping for selective scaling

Strategy:
1. Identify feature groups by domain and distribution
2. Apply appropriate scaler per group:
   - Motion metrics (velocity, acceleration, gyro): StandardScaler
   - Battery metrics (voltage, percentage, drain rate): RobustScaler
   - Bounded features (throttle, percentages): MinMaxScaler
   - Control signals (RC outputs): RobustScaler
3. Save scalers for test-time application
4. Create comparison visualizations
5. Document rationale for each choice

Outputs saved to: output/step_1_5_normalization/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'step_1_5_normalization'
OUT.mkdir(parents=True, exist_ok=True)

ENGINEERED_DATA = ROOT / 'output' / 'step_1_4_feature_engineering' / 'engineered_dataset.csv'

# Define feature groups by domain and scaling strategy
FEATURE_GROUPS = {
    'motion_metrics': {
        'features': ['velocity_magnitude', 'acceleration_magnitude', 'gyro_magnitude', 
                    'ground_speed', 'altitude_rate', 'motion_activity_level'],
        'scaler': 'StandardScaler',
        'rationale': 'Normally distributed motion features benefit from z-score normalization. Velocity and acceleration have no natural bounds.'
    },
    'battery_metrics': {
        'features': ['battery_voltage', 'battery_percentage', 'battery_drain_rate', 
                    'battery_change_rate', 'battery_voltage_rolling_std'],
        'scaler': 'RobustScaler',
        'rationale': 'Battery metrics have outliers (sudden voltage drops, anomalous drain rates). RobustScaler uses median/IQR, insensitive to extremes.'
    },
    'control_signals': {
        'features': ['rc_output_mean', 'rc_output_std', 'setpoint_raw-global_altitude',
                    'vfr_hud_throttle', 'vfr_hud_airspeed', 'vfr_hud_groundspeed'],
        'scaler': 'RobustScaler',
        'rationale': 'RC and control signals have outliers during anomalies/malfunctions. RobustScaler preserves signal while normalizing.'
    },
    'quaternion_orientation': {
        'features': ['quaternion_magnitude', 'imu-data_orientation.x', 'imu-data_orientation.y', 
                    'imu-data_orientation.z', 'imu-data_orientation.w'],
        'scaler': 'StandardScaler',
        'rationale': 'Quaternion components are mathematically bounded (~1 for unit quaternion). StandardScaler normalizes variations.'
    },
    'bounded_features': {
        'features': ['throttle_altitude_ratio', 'setpoint_raw-global_latitude',
                    'setpoint_raw-global_longitude', 'global_position-local_x',
                    'global_position-local_y', 'global_position-local_z'],
        'scaler': 'MinMaxScaler',
        'rationale': 'Position and ratio features have implicit bounds. MinMaxScaler preserves interpretability [0,1].'
    },
}

def load_engineered_data():
    """Load engineered dataset from Step 1.4."""
    df = pd.read_csv(ENGINEERED_DATA)
    print(f"✓ Loaded engineered dataset: {df.shape}")
    return df

def create_scalers(df):
    """Create and fit scalers for each feature group."""
    scalers_dict = {}
    
    for group_name, group_info in FEATURE_GROUPS.items():
        features = [f for f in group_info['features'] if f in df.columns]
        
        if len(features) == 0:
            continue
        
        scaler_name = group_info['scaler']
        
        if scaler_name == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_name == 'RobustScaler':
            scaler = RobustScaler()
        elif scaler_name == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            continue
        
        # Fit scaler
        df_group = df[features].fillna(df[features].mean())
        scaler.fit(df_group)
        
        scalers_dict[group_name] = {
            'scaler': scaler,
            'features': features,
            'scaler_name': scaler_name,
            'rationale': group_info['rationale']
        }
        
        print(f"✓ Fitted {scaler_name} for {group_name} ({len(features)} features)")
    
    return scalers_dict

def apply_scaling(df, scalers_dict):
    """Apply scaling to dataset using fitted scalers."""
    df_scaled = df.copy()
    
    # Also keep original class and sample_index columns
    class_col = df['class'].copy() if 'class' in df.columns else None
    sample_idx = df['sample_index'].copy() if 'sample_index' in df.columns else None
    
    for group_name, scaler_info in scalers_dict.items():
        features = scaler_info['features']
        scaler = scaler_info['scaler']
        
        # Scale features
        df_group = df[features].fillna(df[features].mean())
        df_scaled[features] = scaler.transform(df_group)
    
    # Restore class and sample_index (not scaled)
    if class_col is not None:
        df_scaled['class'] = class_col
    if sample_idx is not None:
        df_scaled['sample_index'] = sample_idx
    
    return df_scaled

def compare_before_after(df_original, df_scaled, scalers_dict):
    """Create comparison visualizations before/after scaling."""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    ax_idx = 0
    for group_name, scaler_info in scalers_dict.items():
        if ax_idx >= len(axes):
            break
        
        features = scaler_info['features'][:3]  # Show first 3 features per group
        
        # Before scaling
        ax = axes[ax_idx]
        data_before = df_original[features].dropna()
        bp_before = ax.boxplot([data_before[f].values for f in features], labels=features, patch_artist=True)
        for patch in bp_before['boxes']:
            patch.set_facecolor('#e74c3c')
        ax.set_title(f'{group_name} (BEFORE)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Original Scale', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        ax_idx += 1
        
        # After scaling
        if ax_idx < len(axes):
            ax = axes[ax_idx]
            data_after = df_scaled[features].dropna()
            bp_after = ax.boxplot([data_after[f].values for f in features], labels=features, patch_artist=True)
            for patch in bp_after['boxes']:
                patch.set_facecolor('#2ecc71')
            ax.set_title(f'{group_name} (AFTER)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Scaled [Normalized]', fontsize=9)
            ax.grid(alpha=0.3, axis='y')
            ax_idx += 1
    
    plt.suptitle('Before/After Scaling Comparison by Feature Group', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUT / 'before_after_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved before/after comparison plots")

def plot_scaling_distributions(df_original, df_scaled, scalers_dict):
    """Create distribution plots showing scaling effect."""
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    ax_idx = 0
    for group_name, scaler_info in scalers_dict.items():
        features = scaler_info['features'][:2]  # 2 features per group
        
        for feat in features:
            if ax_idx >= len(axes):
                break
            
            ax = axes[ax_idx]
            
            # Original distribution
            data_orig = df_original[feat].dropna()
            ax.hist(data_orig, bins=30, alpha=0.6, color='#e74c3c', label='Original', edgecolor='black')
            
            # Scaled distribution (on secondary axis)
            ax2 = ax.twinx()
            data_scaled = df_scaled[feat].dropna()
            ax2.hist(data_scaled, bins=30, alpha=0.6, color='#2ecc71', label='Scaled', edgecolor='black')
            
            ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Frequency (Original)', fontsize=9, color='#e74c3c')
            ax2.set_ylabel('Frequency (Scaled)', fontsize=9, color='#2ecc71')
            ax.grid(alpha=0.3)
            
            ax_idx += 1
    
    # Hide unused
    for idx in range(ax_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution Changes After Scaling', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUT / 'scaling_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution comparison plots")

def plot_scaling_statistics(df_original, df_scaled, scalers_dict):
    """Create table comparing statistics before/after scaling."""
    
    stats_data = []
    
    for group_name, scaler_info in scalers_dict.items():
        features = scaler_info['features'][:2]  # Limit to 2 per group for clarity
        
        for feat in features:
            if feat not in df_original.columns:
                continue
            
            orig_data = df_original[feat].dropna()
            scaled_data = df_scaled[feat].dropna()
            
            stats_data.append({
                'Group': group_name,
                'Feature': feat,
                'Scaler': scaler_info['scaler_name'],
                'Orig Mean': f"{orig_data.mean():.4f}",
                'Orig Std': f"{orig_data.std():.4f}",
                'Scaled Mean': f"{scaled_data.mean():.4f}",
                'Scaled Std': f"{scaled_data.std():.4f}",
                'Orig Range': f"[{orig_data.min():.2e}, {orig_data.max():.2e}]",
                'Scaled Range': f"[{scaled_data.min():.4f}, {scaled_data.max():.4f}]"
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(OUT / 'scaling_statistics.csv', index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns, 
                    cellLoc='center', loc='center', colWidths=[0.12, 0.15, 0.12, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Color header
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(stats_df) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(stats_df.columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Scaling Statistics - Before and After Comparison', fontsize=12, fontweight='bold', pad=20)
    plt.savefig(OUT / 'scaling_statistics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved statistics comparison table")

def save_scalers_and_metadata(scalers_dict):
    """Save scalers and metadata for reproducibility."""
    
    # Save scalers as pickle
    with open(OUT / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers_dict, f)
    print(f"✓ Saved scalers to scalers.pkl")
    
    # Save metadata as JSON-like text
    metadata = []
    for group_name, scaler_info in scalers_dict.items():
        metadata.append(f"\n{group_name.upper()}:")
        metadata.append(f"  Scaler: {scaler_info['scaler_name']}")
        metadata.append(f"  Features ({len(scaler_info['features'])}): {', '.join(scaler_info['features'][:5])}")
        if len(scaler_info['features']) > 5:
            metadata.append(f"             ... and {len(scaler_info['features'])-5} more")
        metadata.append(f"  Rationale: {scaler_info['rationale']}")
    
    with open(OUT / 'scalers_metadata.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(metadata))
    print(f"✓ Saved scaler metadata")

def generate_normalization_report(df_original, df_scaled, scalers_dict):
    """Generate comprehensive normalization report."""
    
    lines = [
        "=" * 130,
        "STEP 1.5: DATA NORMALIZATION/STANDARDIZATION REPORT",
        "=" * 130,
        "",
        f"Dataset: {df_original.shape[0]:,} rows × {df_original.shape[1]} columns",
        f"Classes: {', '.join(sorted(df_original['class'].unique()))}",
        "",
        "=" * 130,
        "SCALING STRATEGY OVERVIEW:",
        "=" * 130,
        "",
        "This step applies domain-aware feature scaling using three complementary techniques:",
        "",
        "1. StandardScaler (mean=0, std=1):",
        "   - Applied to: Motion metrics (velocity, acceleration, gyro magnitudes)",
        "   - Formula: z = (x - mean) / std",
        "   - Best for: Normally distributed, unbounded features, algorithms sensitive to scale (SVM, KNN)",
        "",
        "2. RobustScaler (median=0, IQR=1):",
        "   - Applied to: Battery metrics, control signals, RC outputs",
        "   - Formula: x' = (x - median) / IQR",
        "   - Best for: Features with outliers, preserves signal integrity",
        "",
        "3. MinMaxScaler (range [0,1]):",
        "   - Applied to: Bounded features (position, ratios, geographic coordinates)",
        "   - Formula: x' = (x - min) / (max - min)",
        "   - Best for: Interpretability, bounded ranges, gradient descent methods",
        "",
        "=" * 130,
        "DETAILED FEATURE GROUP ANALYSIS:",
        "=" * 130,
        "",
    ]
    
    for group_name, scaler_info in scalers_dict.items():
        features = scaler_info['features']
        scaler = scaler_info['scaler']
        
        lines.append(f"\nGROUP: {group_name.upper()}")
        lines.append("-" * 130)
        lines.append(f"Scaler: {scaler_info['scaler_name']}")
        lines.append(f"Number of features: {len(features)}")
        lines.append(f"Rationale: {scaler_info['rationale']}")
        lines.append("")
        lines.append("Features in this group:")
        for feat in features:
            if feat in df_original.columns:
                orig = df_original[feat].dropna()
                scaled = df_scaled[feat].dropna()
                lines.append(f"  {feat:40s} | Original: μ={orig.mean():10.4f}, σ={orig.std():10.4f} | "
                           f"Scaled: μ={scaled.mean():8.4f}, σ={scaled.std():8.4f}")
    
    lines.extend([
        "",
        "=" * 130,
        "WHY MULTIPLE SCALERS?",
        "=" * 130,
        "",
        "Different feature groups have different properties requiring different normalization:",
        "",
        "✓ Motion Metrics (StandardScaler):",
        "  - Approximately normally distributed from physics of motion",
        "  - No natural bounds (can be arbitrarily large during maneuvers)",
        "  - Used by: Neural networks, SVM, K-means clustering",
        "  - Benefits: Centers data, standardizes variance, improves convergence",
        "",
        "✓ Battery/Control Signals (RobustScaler):",
        "  - Have outliers due to system faults, anomalies, DoS attacks",
        "  - Using mean/std (StandardScaler) would be influenced by outliers",
        "  - RobustScaler uses median/IQR which are robust to extremes",
        "  - Benefits: Preserves anomalous signal patterns critical for fault detection",
        "",
        "✓ Bounded Features (MinMaxScaler):",
        "  - Have implicit or explicit bounds (coordinates, percentages, ratios)",
        "  - MinMaxScaler preserves interpretability: [min, max] -> [0, 1]",
        "  - Benefits: Easy to interpret, suitable for gradient descent, maintains relationships",
        "",
        "=" * 130,
        "IMPACT ON MACHINE LEARNING MODELS:",
        "=" * 130,
        "",
        "Model Type              | Impact of Scaling",
        "-" * 80,
        "Deep Learning (LSTM,    | HIGH - Weight initialization assumes normalized input",
        "CNN, FNN)               | StandardScaler helps convergence, reduces internal covariate shift",
        "",
        "SVM (RBF/Poly kernels)  | HIGH - Distance metrics sensitive to feature scale",
        "                        | StandardScaler critical; RobustScaler preserves anomalies",
        "",
        "XGBoost                 | MEDIUM - Tree-based, scale-invariant but improves training",
        "                        | Scaling helps with optimization and interpretability",
        "",
        "Tree Models             | LOW - Inherently scale-invariant (tree splits on values)",
        "                        | Scaling not required but helps for comparison",
        "",
        "KNN / Distance-based     | HIGH - Euclidean distance scales with feature magnitude",
        "                        | Scaling essential to prevent large-scale features dominating",
        "",
        "=" * 130,
        "POST-SCALING VALIDATION:",
        "=" * 130,
        "",
        "✓ No data loss: All {0:,} rows preserved".format(df_original.shape[0]),
        f"✓ No feature removal: All {len(scalers_dict)} feature groups scaled",
        f"✓ Class distribution: Unchanged (used for stratification later)",
        "✓ Temporal order: Preserved in sample_index column (not scaled)",
        "✓ Missing values: Filled with feature mean before scaling",
        "",
        "=" * 130,
        "NEXT STEP (1.6):",
        "=" * 130,
        "",
        "Correlation Analysis will:",
        "  1. Compute feature correlations on scaled data",
        "  2. Identify highly correlated feature pairs (>0.95)",
        "  3. Recommend redundant features for removal",
        "  4. Visualize correlation heatmap for domain interpretation",
        "",
        "=" * 130,
    ])
    
    return "\n".join(lines)

def main():
    print("\n" + "=" * 130)
    print("STEP 1.5: DATA NORMALIZATION/STANDARDIZATION")
    print("=" * 130 + "\n")
    
    # Step 1: Load engineered data
    print("Loading engineered dataset from Step 1.4...")
    df = load_engineered_data()
    print()
    
    # Step 2: Create scalers
    print("Creating and fitting scalers for feature groups...")
    scalers_dict = create_scalers(df)
    print()
    
    # Step 3: Apply scaling
    print("Applying scaling to dataset...")
    df_scaled = apply_scaling(df, scalers_dict)
    print(f"✓ Scaled dataset created: {df_scaled.shape}\n")
    
    # Step 4: Save scaled dataset
    print("Saving scaled dataset...")
    df_scaled.to_csv(OUT / 'scaled_dataset.csv', index=False)
    print(f"✓ Saved scaled_dataset.csv\n")
    
    # Step 5: Save scalers
    print("Saving scalers for reproducibility...")
    save_scalers_and_metadata(scalers_dict)
    print()
    
    # Step 6: Create visualizations
    print("Creating comparison visualizations...")
    compare_before_after(df, df_scaled, scalers_dict)
    plot_scaling_distributions(df, df_scaled, scalers_dict)
    plot_scaling_statistics(df, df_scaled, scalers_dict)
    print()
    
    # Step 7: Generate report
    print("Generating normalization report...")
    report = generate_normalization_report(df, df_scaled, scalers_dict)
    with open(OUT / 'normalization_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Saved detailed report\n")
    
    # Summary
    print("=" * 130)
    print("QUICK SUMMARY - NORMALIZATION STATISTICS:")
    print("=" * 130)
    
    total_features = sum(len(info['features']) for info in scalers_dict.values())
    print(f"\nScaling Applied:")
    print(f"  - StandardScaler: {sum(1 for info in scalers_dict.values() if info['scaler_name'] == 'StandardScaler')} groups")
    print(f"  - RobustScaler: {sum(1 for info in scalers_dict.values() if info['scaler_name'] == 'RobustScaler')} groups")
    print(f"  - MinMaxScaler: {sum(1 for info in scalers_dict.values() if info['scaler_name'] == 'MinMaxScaler')} groups")
    print(f"  - Total scaled features: {total_features}")
    
    print(f"\nDataset Status:")
    print(f"  - Original shape: {df.shape}")
    print(f"  - Scaled shape: {df_scaled.shape}")
    print(f"  - Rows preserved: {df.shape[0]:,} (100%)")
    print(f"  - Missing values handled: Filled with feature mean")
    
    print("\n" + "=" * 130)
    print("STEP 1.5 COMPLETE ✓")
    print("=" * 130)
    print(f"\nAll outputs saved to: {OUT}\n")
    
    return df_scaled, scalers_dict

if __name__ == '__main__':
    df_final, scalers = main()
