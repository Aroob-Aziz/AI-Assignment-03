"""
STEP 1.4: FEATURE ENGINEERING
==============================

This script performs comprehensive feature engineering:
- Create derived features from domain knowledge:
  * Velocity magnitude from linear velocity components
  * Battery drain rate (change in battery percentage over time)
  * Acceleration magnitude from IMU gyroscope/accelerometer
  * Orientation magnitude (quaternion norm)
  * Motor output intensity (RC throttle/control signals)
  * Position-based features (altitude variations, movement patterns)
  
- Extract temporal features:
  * Time-based patterns (sequential order, elapsed time)
  * Rate of change features for key metrics
  
- Generate all required visualizations:
  * Distribution plots (histograms) for key features
  * Box plots for outlier detection
  * Time series plots for temporal trends
  * Correlation heatmap
  * Pair plots for feature relationships
  * Class distribution

Outputs saved to: output/step_1_4_feature_engineering/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'RawData'
OUT = ROOT / 'output' / 'step_1_4_feature_engineering'
OUT.mkdir(parents=True, exist_ok=True)

# Files to analyze
FILES = {
    'Dos2': (RAW / 'Dos-Drone' / 'Dos2.csv', 'DoS_Attack'),
    'Malfunction2': (RAW / 'Malfunction-Drone' / 'Malfunction2.csv', 'Malfunction'),
    'Normal2': (RAW / 'NormalFlight' / 'Normal2.csv', 'Normal'),
    'Normal4': (RAW / 'NormalFlight' / 'Normal4.csv', 'Normal'),
}

# Columns to drop (sparse)
COLS_TO_DROP_SPARSE = [
    'setpoint_raw-global_Time', 'setpoint_raw-global_header.seq', 
    'setpoint_raw-global_header.stamp.secs',
    'battery_Time', 'battery_header.seq', 'battery_header.stamp.secs',
    'global_position-local_Time', 'global_position-local_header.seq', 
    'global_position-local_header.stamp.secs',
    'imu-data_Time', 'imu-data_header.seq', 'imu-data_header.stamp.secs',
    'rc-out_Time', 'rc-out_header.seq', 'rc-out_header.stamp.secs',
    'vfr_hud_Time', 'vfr_hud_header.seq', 'vfr_hud_header.stamp.secs',
    'global_position-global_header.stamp.secs',
    'setpoint_raw-target_global_Time', 'setpoint_raw-target_global_header.seq',
    'setpoint_raw-target_global_header.stamp.secs',
    'state_Time', 'state_header.seq',
    'RSSI_Time', 'RSSI_Quality', 'RSSI_Signal',
    'CPU_Time', 'CPU_Percent', 'RAM_Time', 'Used_RAM_MB',
    'S.No'
]

def load_clean_data(file_dict):
    """Load data and drop sparse columns."""
    dfs = []
    for name, (path, class_label) in file_dict.items():
        if path.exists():
            df = pd.read_csv(path)
            df['class'] = class_label
            # Drop sparse columns
            df = df.drop(columns=[c for c in COLS_TO_DROP_SPARSE if c in df.columns])
            dfs.append(df)
            print(f"✓ Loaded {name}: {df.shape}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} columns\n")
    return combined

def engineer_features(df):
    """
    Create domain-aware derived features.
    
    Domain Knowledge for Drone Telemetry:
    - Velocity magnitude: Combined effect of linear motion in 3D space
    - Battery drain rate: Indicator of energy consumption patterns
    - IMU magnitude: Acceleration/gyro intensity indicates maneuvers
    - Orientation magnitude: Quaternion norm for orientation complexity
    - Motor output intensity: RC control signal magnitude (throttle/movements)
    - Altitude rate: Vertical movement speed
    - Ground speed intensity: Horizontal movement speed
    """
    
    df_eng = df.copy()
    
    # 1. VELOCITY MAGNITUDE (from global position - linear velocity in 3D)
    # Try different velocity column names
    vel_cols = ['global_position-local_vx', 'global_position-local_vy', 'global_position-local_vz']
    if all(col in df_eng.columns for col in vel_cols):
        df_eng['velocity_magnitude'] = np.sqrt(
            df_eng['global_position-local_vx']**2 + 
            df_eng['global_position-local_vy']**2 + 
            df_eng['global_position-local_vz']**2
        )
        print("✓ Created feature: velocity_magnitude")
    
    # 2. BATTERY DRAIN RATE (change in battery percentage)
    if 'battery_percentage' in df_eng.columns:
        df_eng['battery_drain_rate'] = df_eng.groupby(['class'])['battery_percentage'].diff().fillna(0)
        # Handle negative values (charging) by taking absolute rate of change
        df_eng['battery_change_rate'] = df_eng['battery_drain_rate'].abs()
        print("✓ Created feature: battery_drain_rate, battery_change_rate")
    
    # 3. IMU ACCELERATION MAGNITUDE (from accelerometer data)
    accel_cols = ['imu-data_linear_acceleration.x', 'imu-data_linear_acceleration.y', 'imu-data_linear_acceleration.z']
    if all(col in df_eng.columns for col in accel_cols):
        df_eng['acceleration_magnitude'] = np.sqrt(
            df_eng['imu-data_linear_acceleration.x']**2 + 
            df_eng['imu-data_linear_acceleration.y']**2 + 
            df_eng['imu-data_linear_acceleration.z']**2
        )
        print("✓ Created feature: acceleration_magnitude")
    
    # 4. IMU GYROSCOPE MAGNITUDE (angular velocity intensity)
    gyro_cols = ['imu-data_angular_velocity.x', 'imu-data_angular_velocity.y', 'imu-data_angular_velocity.z']
    if all(col in df_eng.columns for col in gyro_cols):
        df_eng['gyro_magnitude'] = np.sqrt(
            df_eng['imu-data_angular_velocity.x']**2 + 
            df_eng['imu-data_angular_velocity.y']**2 + 
            df_eng['imu-data_angular_velocity.z']**2
        )
        print("✓ Created feature: gyro_magnitude")
    
    # 5. QUATERNION ORIENTATION MAGNITUDE
    quat_cols = ['imu-data_orientation.x', 'imu-data_orientation.y', 'imu-data_orientation.z', 'imu-data_orientation.w']
    if all(col in df_eng.columns for col in quat_cols):
        # Quaternion norm should be ~1, but variations indicate orientation changes
        df_eng['quaternion_magnitude'] = np.sqrt(
            df_eng['imu-data_orientation.x']**2 + 
            df_eng['imu-data_orientation.y']**2 + 
            df_eng['imu-data_orientation.z']**2 + 
            df_eng['imu-data_orientation.w']**2
        )
        print("✓ Created feature: quaternion_magnitude")
    
    # 6. RC OUTPUT MOTOR INTENSITY (combined throttle and control signals)
    rc_cols = [col for col in df_eng.columns if col.startswith('rc-out_')]
    if len(rc_cols) >= 4:
        # Take mean of RC outputs as overall motor command intensity
        df_eng['rc_output_mean'] = df_eng[[col for col in rc_cols if col in df_eng.columns]].mean(axis=1)
        df_eng['rc_output_std'] = df_eng[[col for col in rc_cols if col in df_eng.columns]].std(axis=1)
        print(f"✓ Created feature: rc_output_mean, rc_output_std (from {len(rc_cols)} RC channels)")
    
    # 7. ALTITUDE RATE OF CHANGE (vertical movement speed)
    if 'global_position-local_z' in df_eng.columns:
        df_eng['altitude_rate'] = df_eng.groupby(['class'])['global_position-local_z'].diff().fillna(0).abs()
        print("✓ Created feature: altitude_rate")
    
    # 8. GROUND SPEED (2D horizontal movement)
    if 'global_position-local_vx' in df_eng.columns and 'global_position-local_vy' in df_eng.columns:
        df_eng['ground_speed'] = np.sqrt(
            df_eng['global_position-local_vx']**2 + 
            df_eng['global_position-local_vy']**2
        )
        print("✓ Created feature: ground_speed")
    
    # 9. BATTERY VOLTAGE STABILITY (as variance proxy in moving window)
    if 'battery_voltage' in df_eng.columns:
        df_eng['battery_voltage_rolling_std'] = df_eng.groupby(['class'])['battery_voltage'].rolling(
            window=10, min_periods=1
        ).std().reset_index(drop=True)
        print("✓ Created feature: battery_voltage_rolling_std")
    
    # 10. THROTTLE/ALTITUDE RELATIONSHIP
    if 'setpoint_raw-global_altitude' in df_eng.columns and 'vfr_hud_throttle' in df_eng.columns:
        df_eng['throttle_altitude_ratio'] = (
            (df_eng['setpoint_raw-global_altitude'].fillna(0) + 1) / 
            (df_eng['vfr_hud_throttle'].fillna(1) + 1)
        ).fillna(0)
        print("✓ Created feature: throttle_altitude_ratio")
    
    # 11. TEMPORAL FEATURES
    # Sample index as proxy for time evolution within each file
    df_eng['sample_index'] = df_eng.groupby('class').cumcount()
    print("✓ Created feature: sample_index (temporal order)")
    
    # 12. CLASS-SPECIFIC ACTIVITY LEVEL (combination of motion metrics)
    motion_cols = ['velocity_magnitude', 'acceleration_magnitude', 'gyro_magnitude']
    existing_motion_cols = [col for col in motion_cols if col in df_eng.columns]
    if len(existing_motion_cols) > 0:
        df_eng['motion_activity_level'] = df_eng[existing_motion_cols].mean(axis=1)
        print(f"✓ Created feature: motion_activity_level (from {len(existing_motion_cols)} motion features)")
    
    return df_eng

def create_distribution_plots(df, output_dir):
    """Create distribution histograms for engineered and key original features."""
    # Select features to visualize
    engineered_cols = ['velocity_magnitude', 'battery_drain_rate', 'acceleration_magnitude', 
                      'gyro_magnitude', 'altitude_rate', 'ground_speed', 'motion_activity_level',
                      'rc_output_mean', 'sample_index']
    engineered_cols = [col for col in engineered_cols if col in df.columns]
    
    # Add some original important features
    original_cols = ['battery_voltage', 'battery_percentage', 'setpoint_raw-global_altitude',
                    'vfr_hud_airspeed', 'vfr_hud_groundspeed']
    original_cols = [col for col in original_cols if col in df.columns]
    
    plot_cols = engineered_cols + original_cols[:5]  # Limit to ~14 features
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(plot_cols):
        data = df[col].dropna()
        if len(data) > 1:
            # Check if data is numeric and has variation
            try:
                data_numeric = pd.to_numeric(data, errors='coerce').dropna()
                if len(data_numeric) > 0 and data_numeric.std() > 0:
                    # Adaptive bin count based on unique values
                    n_unique = len(data_numeric.unique())
                    n_bins = min(40, max(5, n_unique // 10))
                    axes[idx].hist(data_numeric, bins=n_bins, alpha=0.7, color='#3498db', edgecolor='black')
                else:
                    # For categorical or constant data, use bar plot
                    value_counts = data.value_counts().head(10)
                    axes[idx].bar(range(len(value_counts)), value_counts.values, alpha=0.7, color='#3498db', edgecolor='black')
            except:
                # Fallback to bar plot for non-numeric
                value_counts = data.value_counts().head(10)
                axes[idx].bar(range(len(value_counts)), value_counts.values, alpha=0.7, color='#3498db', edgecolor='black')
            
            axes[idx].set_title(col, fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontsize=9)
            axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(plot_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution Plots - Engineered & Key Original Features', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_engineered_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution plots")

def create_boxplots(df, output_dir):
    """Create box plots for engineered features by class."""
    engineered_cols = ['velocity_magnitude', 'battery_change_rate', 'acceleration_magnitude', 
                      'gyro_magnitude', 'altitude_rate', 'ground_speed', 'motion_activity_level']
    engineered_cols = [col for col in engineered_cols if col in df.columns]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(engineered_cols):
        data_by_class = [df[df['class'] == cls][col].dropna() for cls in sorted(df['class'].unique())]
        bp = axes[idx].boxplot(data_by_class, labels=sorted(df['class'].unique()), patch_artist=True)
        
        # Color by class
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[idx].set_title(col, fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=9)
        axes[idx].grid(alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(engineered_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Box Plots of Engineered Features by Class', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_engineered_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved box plots by class")

def create_timeseries_plots(df, output_dir):
    """Create time series plots for engineered features."""
    engineered_cols = ['velocity_magnitude', 'battery_percentage', 'acceleration_magnitude', 
                      'gyro_magnitude', 'motion_activity_level']
    engineered_cols = [col for col in engineered_cols if col in df.columns]
    
    fig, axes = plt.subplots(len(engineered_cols), 1, figsize=(14, 12))
    if len(engineered_cols) == 1:
        axes = [axes]
    
    classes = sorted(df['class'].unique())
    colors = {'DoS_Attack': '#e74c3c', 'Malfunction': '#f39c12', 'Normal': '#2ecc71'}
    
    for idx, col in enumerate(engineered_cols):
        for cls in classes:
            df_cls = df[df['class'] == cls]
            axes[idx].plot(df_cls['sample_index'], df_cls[col], label=cls, alpha=0.5, linewidth=1, color=colors.get(cls, '#3498db'))
        
        axes[idx].set_title(f'{col} - Time Series by Class', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=9)
        axes[idx].set_xlabel('Sample Index', fontsize=9)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Time Series Plots of Engineered Features', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'timeseries_engineered_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved time series plots")

def create_correlation_heatmap(df, output_dir):
    """Create correlation matrix heatmap."""
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'class' and df[col].notna().sum() > 100]
    
    # Limit to ~25 features for readability
    if len(numeric_cols) > 25:
        # Prioritize engineered features and high-variance features
        engineered_features = [col for col in numeric_cols if any(
            eng in col for eng in ['magnitude', 'rate', 'drain', 'velocity', 'acceleration', 'gyro']
        )]
        other_features = [col for col in numeric_cols if col not in engineered_features]
        var_sorted = other_features + engineered_features
        numeric_cols = var_sorted[:25]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Heatmap - Engineered & Original Features', 
                fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved correlation heatmap")

def create_pairplot(df, output_dir):
    """Create pair plot for top engineered features."""
    engineered_cols = ['velocity_magnitude', 'acceleration_magnitude', 'gyro_magnitude', 
                      'battery_change_rate', 'motion_activity_level', 'class']
    engineered_cols = [col for col in engineered_cols if col in df.columns]
    
    # Sample data if too large (pair plots can be slow)
    df_sample = df[engineered_cols].sample(n=min(2000, len(df)), random_state=42)
    
    fig = sns.pairplot(df_sample, hue='class', diag_kind='hist', plot_kws={'alpha': 0.5}, 
                       diag_kws={'bins': 30, 'edgecolor': 'black'})
    fig.fig.suptitle('Pair Plot - Top Engineered Features by Class', 
                    fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'pairplot_engineered_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved pair plot")

def create_class_distribution_plot(df, output_dir):
    """Create class distribution visualization."""
    class_counts = df['class'].value_counts()
    class_pcts = 100 * class_counts / len(df)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    axes[0].bar(class_counts.index, class_counts.values, color=colors[:len(class_counts)], alpha=0.8, edgecolor='black')
    axes[0].set_title('Class Distribution (Count)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=10)
    for i, (idx, val) in enumerate(class_counts.items()):
        axes[0].text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Pie chart
    axes[1].pie(class_pcts.values, labels=class_pcts.index, autopct='%1.1f%%', 
               colors=colors[:len(class_pcts)], startangle=90)
    axes[1].set_title('Class Distribution (Percentage)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Class Distribution in Engineered Dataset', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_engineered.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class distribution plot")

def generate_feature_engineering_report(df, original_df):
    """Generate comprehensive feature engineering report."""
    
    # Get engineered features
    engineered_features = [col for col in df.columns if col not in original_df.columns and col != 'class']
    
    lines = [
        "=" * 120,
        "STEP 1.4: FEATURE ENGINEERING REPORT",
        "=" * 120,
        "",
        f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Original features: {original_df.shape[1]}",
        f"Engineered features: {len(engineered_features)}",
        f"Total features after engineering: {df.shape[1] - 1}",  # Excluding 'class'
        f"Classes: {', '.join(sorted(df['class'].unique()))}",
        "",
        "=" * 120,
        "ENGINEERED FEATURES:",
        "=" * 120,
        "",
    ]
    
    feature_descriptions = {
        'velocity_magnitude': 'Combined linear velocity magnitude in 3D space (sqrt(vx^2 + vy^2 + vz^2))',
        'battery_drain_rate': 'Rate of battery percentage change per sample (temporal derivative)',
        'battery_change_rate': 'Absolute rate of battery voltage/percentage change',
        'acceleration_magnitude': 'Combined linear acceleration magnitude from IMU (sqrt(ax^2 + ay^2 + az^2))',
        'gyro_magnitude': 'Combined angular velocity (rotation rate) magnitude (sqrt(gx^2 + gy^2 + gz^2))',
        'quaternion_magnitude': 'Magnitude of orientation quaternion (should be ~1, variation indicates drift)',
        'rc_output_mean': 'Mean RC motor output signal (indicator of overall motor demand)',
        'rc_output_std': 'Standard deviation of RC outputs (indicator of control asymmetry)',
        'altitude_rate': 'Rate of change of altitude (vertical movement speed)',
        'ground_speed': 'Horizontal 2D velocity magnitude (sqrt(vx^2 + vy^2))',
        'battery_voltage_rolling_std': 'Rolling standard deviation of battery voltage (stability indicator)',
        'throttle_altitude_ratio': 'Relationship between throttle command and altitude setpoint',
        'sample_index': 'Temporal order of samples (for sequential pattern detection)',
        'motion_activity_level': 'Combined motion intensity (mean of velocity, acceleration, gyro magnitudes)',
    }
    
    for feat in sorted(engineered_features):
        if feat in feature_descriptions:
            lines.append(f"{feat}:")
            lines.append(f"  Description: {feature_descriptions[feat]}")
            
            # Statistics
            if feat in df.columns:
                data = df[feat].dropna()
                if len(data) > 0:
                    lines.append(f"  Statistics: count={len(data)}, mean={data.mean():.4f}, "
                               f"std={data.std():.4f}, min={data.min():.4f}, max={data.max():.4f}")
        lines.append("")
    
    lines.extend([
        "=" * 120,
        "DOMAIN KNOWLEDGE RATIONALE:",
        "=" * 120,
        "",
        "1. MOTION METRICS (velocity, acceleration, gyro magnitudes):",
        "   - Drone telemetry fundamentally tracks motion in 3D space",
        "   - Individual components may be noisy; magnitude captures overall motion intensity",
        "   - DoS attacks and malfunctions manifest as unusual motion patterns",
        "",
        "2. BATTERY METRICS (drain rate, voltage stability):",
        "   - Battery state is critical for fault detection",
        "   - Sudden voltage drops indicate electrical faults",
        "   - Abnormal drain rates suggest motor/system issues",
        "",
        "3. CONTROL METRICS (RC output mean/std, throttle-altitude ratio):",
        "   - RC signals represent pilot/autopilot commands",
        "   - Asymmetric motor outputs (high std) indicate flight instability",
        "   - Throttle-altitude relationship tracks control compliance",
        "",
        "4. TEMPORAL FEATURES (sample index, altitude rate):",
        "   - Sequential patterns may differentiate normal vs anomalous behavior",
        "   - Rate of altitude change is more informative than absolute altitude",
        "",
        "5. INTEGRATED METRICS (motion_activity_level):",
        "   - Combines multiple motion sources into single activity indicator",
        "   - Useful for rapid anomaly detection (sudden activity spikes)",
        "",
        "=" * 120,
        "FEATURE ENGINEERING IMPACT:",
        "=" * 120,
        "",
        f"Features added: {len(engineered_features)}",
        f"Original sparse features removed: {original_df.shape[1] - df.shape[1] + len(engineered_features) - 1}",
        f"Net feature count change: +{len(engineered_features)} (after sparse column removal)",
        "",
        "Key Benefits:",
        "  ✓ Reduces dimensionality by removing high-sparsity features",
        "  ✓ Increases interpretability with domain-meaningful features",
        "  ✓ Captures non-linear relationships through magnitude features",
        "  ✓ Temporal patterns encoded through rate-of-change and sequence features",
        "  ✓ Better suited for anomaly detection (DoS attacks, malfunctions)",
        "",
        "=" * 120,
    ])
    
    return "\n".join(lines)

def main():
    print("\n" + "=" * 120)
    print("STEP 1.4: FEATURE ENGINEERING")
    print("=" * 120 + "\n")
    
    # Step 1: Load clean data
    print("Loading and cleaning data...")
    df_original = load_clean_data(FILES)
    
    # Step 2: Engineer features
    print("Creating engineered features based on domain knowledge...")
    df_engineered = engineer_features(df_original)
    
    # Handle NaN values in new engineered features
    df_engineered = df_engineered.fillna(df_engineered.mean(numeric_only=True))
    
    new_features = [col for col in df_engineered.columns if col not in df_original.columns]
    print(f"\n✓ Created {len(new_features)} engineered features\n")
    
    # Step 3: Create visualizations
    print("Creating visualizations...")
    create_distribution_plots(df_engineered, OUT)
    create_boxplots(df_engineered, OUT)
    create_timeseries_plots(df_engineered, OUT)
    create_correlation_heatmap(df_engineered, OUT)
    create_pairplot(df_engineered, OUT)
    create_class_distribution_plot(df_engineered, OUT)
    print()
    
    # Step 4: Save engineered dataset
    print("Saving engineered dataset...")
    df_engineered.to_csv(OUT / 'engineered_dataset.csv', index=False)
    print(f"✓ Saved engineered dataset: {df_engineered.shape}")
    
    # Step 5: Generate report
    print("Generating feature engineering report...")
    report = generate_feature_engineering_report(df_engineered, df_original)
    with open(OUT / 'feature_engineering_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Saved detailed report\n")
    
    # Step 6: Summary statistics
    print("=" * 120)
    print("QUICK SUMMARY - FEATURE ENGINEERING STATISTICS:")
    print("=" * 120)
    print(f"\nOriginal dataset: {df_original.shape[0]} rows × {df_original.shape[1]} columns")
    print(f"Engineered dataset: {df_engineered.shape[0]} rows × {df_engineered.shape[1]} columns")
    print(f"\nNew features created: {len(new_features)}")
    for feat in sorted(new_features):
        if feat in df_engineered.columns:
            print(f"  - {feat}")
    
    print(f"\n" + "=" * 120)
    print("STEP 1.4 COMPLETE ✓")
    print("=" * 120)
    print(f"\nAll outputs saved to: {OUT}\n")
    
    return df_engineered

if __name__ == '__main__':
    df_final = main()
