"""
STEP 1.3: OUTLIER DETECTION AND TREATMENT
===========================================

This script performs comprehensive outlier analysis:
- Identify outliers using IQR and Z-score methods
- Visualize outliers with box plots and scatter plots
- Create distribution plots (histograms/KDE) for key features
- Decide on treatment strategy (remove, cap, or transform)
- Document approach and generate comprehensive report

Outputs saved to: output/step_1_3_outlier_analysis/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'RawData'
OUT = ROOT / 'output' / 'step_1_3_outlier_analysis'
OUT.mkdir(parents=True, exist_ok=True)

# Files to analyze
FILES = {
    'Dos2': (RAW / 'Dos-Drone' / 'Dos2.csv', 'DoS_Attack'),
    'Malfunction2': (RAW / 'Malfunction-Drone' / 'Malfunction2.csv', 'Malfunction'),
    'Normal2': (RAW / 'NormalFlight' / 'Normal2.csv', 'Normal'),
    'Normal4': (RAW / 'NormalFlight' / 'Normal4.csv', 'Normal'),
}

# Columns to keep (from Step 1.2 analysis - columns with <80% missing)
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
    print(f"\n✓ Combined dataset (sparse cols removed): {combined.shape[0]} rows × {combined.shape[1]} columns\n")
    return combined

def detect_outliers_iqr(df, column, iqr_multiplier=1.5):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3.0):
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = z_scores > threshold
    return outliers, z_scores

def analyze_outliers_per_column(df):
    """Analyze outliers for all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'class']
    
    outlier_stats = []
    
    for col in numeric_cols:
        # Skip if all NaN
        if df[col].isna().all():
            continue
        
        # IQR method
        outliers_iqr, lower, upper = detect_outliers_iqr(df, col)
        n_outliers_iqr = outliers_iqr.sum()
        pct_outliers_iqr = 100 * n_outliers_iqr / len(df)
        
        # Z-score method
        data_clean = df[col].dropna()
        if len(data_clean) > 0:
            z_scores = np.abs(stats.zscore(data_clean))
            n_outliers_z = (z_scores > 3).sum()
            pct_outliers_z = 100 * n_outliers_z / len(data_clean)
        else:
            n_outliers_z = 0
            pct_outliers_z = 0
        
        outlier_stats.append({
            'column': col,
            'n_outliers_iqr': int(n_outliers_iqr),
            'pct_outliers_iqr': pct_outliers_iqr,
            'n_outliers_zscore': int(n_outliers_z),
            'pct_outliers_zscore': pct_outliers_z,
            'lower_bound_iqr': lower,
            'upper_bound_iqr': upper,
            'mean': data_clean.mean(),
            'median': data_clean.median(),
            'std': data_clean.std(),
        })
    
    outlier_df = pd.DataFrame(outlier_stats)
    outlier_df = outlier_df.sort_values('pct_outliers_iqr', ascending=False)
    return outlier_df

def create_distribution_plots(df, output_dir, n_cols=20):
    """Create distribution plots (histograms/KDE) for top features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'class']
    
    # Select top columns with most variance (interesting features)
    top_cols = df[numeric_cols].var().nlargest(n_cols).index.tolist()
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_cols):
        axes[idx].hist(df[col].dropna(), bins=40, alpha=0.7, color='#3498db', edgecolor='black')
        axes[idx].set_title(col, fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontsize=9)
        axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution Plots (Histograms) - Top 20 Features by Variance', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distribution histograms")

def create_boxplots_outliers(df, output_dir, n_cols=20):
    """Create box plots for outlier detection."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'class']
    
    # Select top columns with most outliers
    top_cols = df[numeric_cols].apply(
        lambda x: detect_outliers_iqr(df, x.name)[0].sum()
    ).nlargest(n_cols).index.tolist()
    
    fig, axes = plt.subplots(5, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_cols):
        bp = axes[idx].boxplot([df[col].dropna()], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        axes[idx].set_title(col, fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=9)
        axes[idx].grid(alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(top_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Box Plots for Outlier Detection - Top 20 Features', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved box plots for outlier detection")

def create_scatter_outliers(df, output_dir):
    """Create scatter plots for top outlier features."""
    outlier_stats = analyze_outliers_per_column(df)
    top_outlier_cols = outlier_stats.head(6)['column'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_outlier_cols):
        outliers, _, _ = detect_outliers_iqr(df, col)
        
        # Plot normal points
        axes[idx].scatter(range(len(df)), df[col], c='#3498db', alpha=0.3, s=10, label='Normal')
        # Highlight outliers
        axes[idx].scatter(np.where(outliers)[0], df.loc[outliers, col], 
                         c='#e74c3c', alpha=0.8, s=20, label='Outliers (IQR)')
        
        axes[idx].set_title(col, fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=9)
        axes[idx].set_xlabel('Sample Index', fontsize=9)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Scatter Plots of Outliers (Top 6 Features by Outlier Count)', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved scatter plots of outliers")

def create_outlier_summary_by_class(df, output_dir):
    """Create bar chart comparing outlier counts by class."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'class']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    classes = sorted(df['class'].unique())
    for class_idx, cls in enumerate(classes):
        df_cls = df[df['class'] == cls]
        
        outlier_counts = []
        col_names = []
        
        for col in numeric_cols:
            outliers, _, _ = detect_outliers_iqr(df_cls, col)
            outlier_counts.append(outliers.sum())
            col_names.append(col)
        
        # Top 15 by outlier count
        sorted_idx = np.argsort(outlier_counts)[::-1][:15]
        top_cols = [col_names[i] for i in sorted_idx]
        top_counts = [outlier_counts[i] for i in sorted_idx]
        
        axes[class_idx].barh(range(len(top_cols)), top_counts, color='#e74c3c', alpha=0.8)
        axes[class_idx].set_yticks(range(len(top_cols)))
        axes[class_idx].set_yticklabels(top_cols, fontsize=8)
        axes[class_idx].set_xlabel('Outlier Count (IQR)', fontsize=10)
        axes[class_idx].set_title(f'{cls}\n(n={len(df_cls):,})', fontsize=11, fontweight='bold')
        axes[class_idx].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Outlier Counts by Class (Top 15 Features)', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'outlier_counts_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved outlier counts by class")

def generate_outlier_report(df, outlier_stats):
    """Generate comprehensive outlier analysis report."""
    lines = [
        "=" * 100,
        "STEP 1.3: OUTLIER DETECTION AND TREATMENT REPORT",
        "=" * 100,
        "",
        f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Classes: {', '.join(sorted(df['class'].unique()))}",
        "",
        "=" * 100,
        "OUTLIER DETECTION SUMMARY:",
        "=" * 100,
        "",
        outlier_stats.to_string(index=False),
        "",
        "=" * 100,
        "OUTLIER TREATMENT STRATEGY:",
        "=" * 100,
        "",
        "METHOD 1: IQR-Based Capping (Recommended)",
        "-" * 100,
        "",
        "Approach:",
        "  - For each numeric feature, calculate Q1, Q3, and IQR = Q3 - Q1",
        "  - Define outlier bounds: Lower = Q1 - 1.5*IQR, Upper = Q3 + 1.5*IQR",
        "  - Cap values: Replace values < Lower with Lower, values > Upper with Upper",
        "  - This preserves all samples while reducing extreme values",
        "",
        "Advantages:",
        "  + Preserves sample count (no data loss)",
        "  + Robust to outlier magnitude",
        "  + Maintains temporal/sequential order if present",
        "  + Retains domain-relevant information in boundary regions",
        "",
        "Implementation:",
        "  - Apply per-feature capping based on IQR bounds",
        "  - Optional: Apply per-class to preserve class-specific characteristics",
        "",
        "Decision: USE IQR-BASED CAPPING",
        "Rationale: Telemetry data outliers may represent transient system states or anomalies",
        "           (e.g., sudden movements, DoS attacks). Capping preserves these signals",
        "           while preventing extreme values from dominating scaled features.",
        "",
        "=" * 100,
        "FEATURES WITH SIGNIFICANT OUTLIERS (>5% by IQR):",
        "=" * 100,
        "",
    ]
    
    sig_outliers = outlier_stats[outlier_stats['pct_outliers_iqr'] > 5]
    for _, row in sig_outliers.head(15).iterrows():
        lines.append(
            f"{row['column']:45s} | IQR: {row['pct_outliers_iqr']:5.2f}% "
            f"({row['n_outliers_iqr']:5d}) | Z-score: {row['pct_outliers_zscore']:5.2f}%"
        )
    
    lines.extend([
        "",
        "=" * 100,
        "TREATMENT PROCEDURE:",
        "=" * 100,
        "",
        "1. For each numeric column:",
        "   - Calculate Q1, Q3, IQR",
        "   - Set lower_bound = Q1 - 1.5 * IQR",
        "   - Set upper_bound = Q3 + 1.5 * IQR",
        "   - Cap: df[col] = df[col].clip(lower_bound, upper_bound)",
        "",
        "2. Expected outcome:",
        f"   - Features affected: ~{len(sig_outliers)} columns",
        f"   - Samples affected: Varying (typically 1-10% per feature)",
        "   - Total data loss: 0 rows (capping preserves all samples)",
        "",
        "3. Post-treatment validation:",
        "   - Check for remaining extreme values",
        "   - Verify class-specific distributions",
        "   - Confirm numeric ranges are reasonable",
        "",
        "=" * 100,
        "OUTLIER CHARACTERISTICS BY CLASS:",
        "=" * 100,
        "",
    ])
    
    for cls in sorted(df['class'].unique()):
        df_cls = df[df['class'] == cls]
        lines.append(f"\nCLASS: {cls.upper()}")
        lines.append("-" * 100)
        
        numeric_cols = df_cls.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'class']
        
        outlier_counts = []
        for col in numeric_cols:
            outliers, _, _ = detect_outliers_iqr(df_cls, col)
            outlier_counts.append(outliers.sum())
        
        total_outliers = sum(outlier_counts)
        total_cells = len(df_cls) * len(numeric_cols)
        lines.append(
            f"  Total outlier instances (IQR): {total_outliers:,} / {total_cells:,} "
            f"({100*total_outliers/total_cells:.2f}%)"
        )
        
        lines.append("")
    
    lines.extend([
        "=" * 100,
        "CONCLUSION:",
        "=" * 100,
        "",
        "Outlier Treatment: IQR-based capping (no removal)",
        "",
        "Reasoning:",
        "  1. Preserves all samples and temporal sequences",
        "  2. Outliers in telemetry data are meaningful (system events/anomalies)",
        "  3. Capping reduces extreme values without losing information",
        "  4. Prepares data for scaling without unbounded values",
        "",
        "=" * 100,
    ])
    
    return "\n".join(lines)

def main():
    print("\n" + "=" * 100)
    print("STEP 1.3: OUTLIER DETECTION AND TREATMENT")
    print("=" * 100 + "\n")
    
    # Step 1: Load clean data
    print("Loading and cleaning data (removing sparse columns)...")
    df = load_clean_data(FILES)
    
    # Step 2: Detect and analyze outliers
    print("Detecting outliers using IQR and Z-score methods...")
    outlier_stats = analyze_outliers_per_column(df)
    outlier_stats.to_csv(OUT / 'outlier_statistics.csv', index=False)
    print(f"✓ Saved outlier statistics to outlier_statistics.csv\n")
    
    # Step 3: Create visualizations
    print("Creating visualizations...")
    create_distribution_plots(df, OUT)
    create_boxplots_outliers(df, OUT)
    create_scatter_outliers(df, OUT)
    create_outlier_summary_by_class(df, OUT)
    print()
    
    # Step 4: Generate report
    print("Generating outlier analysis report...")
    report = generate_outlier_report(df, outlier_stats)
    with open(OUT / 'outlier_analysis_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved detailed report to outlier_analysis_report.txt\n")
    
    # Step 5: Summary statistics
    print("=" * 100)
    print("QUICK SUMMARY - OUTLIER STATISTICS:")
    print("=" * 100)
    print(f"\nTotal numeric features analyzed: {len(outlier_stats)}")
    print(f"\nOutlier prevalence (IQR method):")
    print(f"  - Features with >5% outliers:   {len(outlier_stats[outlier_stats['pct_outliers_iqr'] > 5])}")
    print(f"  - Features with >2% outliers:   {len(outlier_stats[outlier_stats['pct_outliers_iqr'] > 2])}")
    print(f"  - Features with 0.1-2% outliers: {len(outlier_stats[(outlier_stats['pct_outliers_iqr'] >= 0.1) & (outlier_stats['pct_outliers_iqr'] <= 2)])}")
    
    top_outlier = outlier_stats.iloc[0]
    print(f"\nMost outlier-heavy feature: {top_outlier['column']}")
    print(f"  - IQR method: {top_outlier['pct_outliers_iqr']:.2f}% ({top_outlier['n_outliers_iqr']} outliers)")
    print(f"  - Z-score (z>3): {top_outlier['pct_outliers_zscore']:.2f}% ({top_outlier['n_outliers_zscore']} outliers)")
    
    print("\n" + "=" * 100)
    print("STEP 1.3 COMPLETE ✓")
    print("=" * 100)
    print(f"\nAll outputs saved to: {OUT}\n")

if __name__ == '__main__':
    main()
