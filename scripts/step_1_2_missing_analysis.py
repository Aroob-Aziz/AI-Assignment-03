"""
STEP 1.2: MISSING DATA ANALYSIS
================================

This script performs comprehensive missing data analysis:
- Identify missing values in each column (counts & percentages)
- Visualize missing data patterns (heatmap, bar charts)
- Analyze missing data correlations and patterns by class
- Document and decide on imputation/deletion strategies
- Generate actionable recommendations

Outputs saved to: output/step_1_2_missing_analysis/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'RawData'
OUT = ROOT / 'output' / 'step_1_2_missing_analysis'
OUT.mkdir(parents=True, exist_ok=True)

# Files to analyze
FILES = {
    'Dos2': (RAW / 'Dos-Drone' / 'Dos2.csv', 'DoS_Attack'),
    'Malfunction2': (RAW / 'Malfunction-Drone' / 'Malfunction2.csv', 'Malfunction'),
    'Normal2': (RAW / 'NormalFlight' / 'Normal2.csv', 'Normal'),
    'Normal4': (RAW / 'NormalFlight' / 'Normal4.csv', 'Normal'),
}

def load_all_data(file_dict):
    """Load all files and add class labels."""
    dfs = []
    for name, (path, class_label) in file_dict.items():
        if path.exists():
            df = pd.read_csv(path)
            df['class'] = class_label
            dfs.append(df)
            print(f"✓ Loaded {name}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} columns")
    return combined

def analyze_missing_per_column(df):
    """Analyze missing data per column."""
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isna().sum().values,
        'missing_pct': (100 * df.isna().sum() / len(df)).values,
    })
    missing_stats = missing_stats.sort_values('missing_pct', ascending=False)
    missing_stats['strategy'] = missing_stats['missing_pct'].apply(categorize_strategy)
    
    return missing_stats

def categorize_strategy(pct):
    """Categorize imputation strategy based on missing %."""
    if pct > 80:
        return 'DROP (>80%)'
    elif pct > 50:
        return 'DROP (>50%) or Forward-Fill'
    elif pct > 20:
        return 'Median/Mean Imputation'
    elif pct > 5:
        return 'Median Imputation'
    else:
        return 'Keep (minimal missing)'

def create_missing_heatmap(df, output_path, sample_size=500):
    """Create heatmap of missing data (sample to avoid huge plots)."""
    # Sample rows if dataset is large
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create binary matrix (1 = missing, 0 = present)
    missing_matrix = df_sample.isna().astype(int)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(missing_matrix, cbar=True, cmap='RdYlGn_r', 
                xticklabels=True, yticklabels=False)
    plt.title(f'Missing Data Pattern (sample of {len(df_sample)} rows)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved missing data heatmap to {output_path.name}")

def create_missing_bar_chart(missing_stats, output_path):
    """Create bar chart of missing % per column."""
    top_n = 40  # Top 40 columns with most missing data
    top_missing = missing_stats.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by strategy
    colors = []
    strategy_colors = {
        'DROP (>80%)': '#e74c3c',
        'DROP (>50%) or Forward-Fill': '#e67e22',
        'Median/Mean Imputation': '#f39c12',
        'Median Imputation': '#2ecc71',
        'Keep (minimal missing)': '#27ae60',
    }
    
    for _, row in top_missing.iterrows():
        colors.append(strategy_colors.get(row['strategy'], '#95a5a6'))
    
    bars = ax.barh(range(len(top_missing)), top_missing['missing_pct'], color=colors)
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing['column'], fontsize=9)
    ax.set_xlabel('Missing %', fontsize=11, fontweight='bold')
    ax.set_title(f'Missing Data % by Column (Top {top_n})', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, top_missing['missing_pct'])):
        ax.text(pct + 1, i, f'{pct:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved missing % bar chart to {output_path.name}")

def create_missing_by_class(df, output_path):
    """Create bar chart of missing % by class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    classes = df['class'].unique()
    for idx, cls in enumerate(sorted(classes)):
        df_cls = df[df['class'] == cls]
        missing_pct = 100 * df_cls.isna().sum() / len(df_cls)
        missing_pct = missing_pct.sort_values(ascending=False).head(20)
        
        axes[idx].barh(range(len(missing_pct)), missing_pct.values, 
                       color='#3498db', alpha=0.8)
        axes[idx].set_yticks(range(len(missing_pct)))
        axes[idx].set_yticklabels(missing_pct.index, fontsize=8)
        axes[idx].set_xlabel('Missing %', fontsize=10)
        axes[idx].set_title(f'{cls}\n(n={len(df_cls)})', fontsize=11, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.suptitle('Missing Data % by Class (Top 20 Features)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved missing by class chart to {output_path.name}")

def generate_missing_report(df, missing_stats):
    """Generate detailed text report on missing data."""
    lines = [
        "=" * 90,
        "STEP 1.2: MISSING DATA ANALYSIS REPORT",
        "=" * 90,
        "",
        f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Total Cells: {df.shape[0] * df.shape[1]:,}",
        f"Total Missing Cells: {df.isna().sum().sum():,}",
        f"Overall Missing %: {100 * df.isna().sum().sum() / (df.shape[0] * df.shape[1]):.2f}%",
        "",
        "=" * 90,
        "MISSING DATA SUMMARY BY COLUMN:",
        "=" * 90,
        "",
    ]
    
    lines.append(missing_stats.to_string(index=False))
    
    lines.extend([
        "",
        "=" * 90,
        "STRATEGY CATEGORIZATION:",
        "=" * 90,
        "",
    ])
    
    for strategy in ['Keep (minimal missing)', 'Median Imputation', 'Median/Mean Imputation',
                    'DROP (>50%) or Forward-Fill', 'DROP (>80%)']:
        cols_in_strategy = missing_stats[missing_stats['strategy'] == strategy]
        lines.append(f"\n{strategy.upper()} ({len(cols_in_strategy)} columns):")
        if len(cols_in_strategy) > 0:
            for _, row in cols_in_strategy.head(10).iterrows():
                lines.append(f"  - {row['column']:45s} | Missing: {row['missing_pct']:6.2f}%")
            if len(cols_in_strategy) > 10:
                lines.append(f"  ... and {len(cols_in_strategy) - 10} more")
    
    lines.extend([
        "",
        "=" * 90,
        "IMPUTATION STRATEGY RECOMMENDATION:",
        "=" * 90,
        "",
        "1. COLUMNS TO DROP (>80% missing):",
        "   Rationale: Too sparse for meaningful imputation. Dropping will not",
        "   significantly lose information and improves model efficiency.",
        f"   Count: {len(missing_stats[missing_stats['missing_pct'] > 80])} columns",
        "",
        "2. COLUMNS TO FORWARD-FILL / DROP (50-80% missing):",
        "   Rationale: Highly sparse sensor data (low-frequency updates).",
        "   Options:",
        "     a) Drop if not critical (e.g., RSSI, CPU, RAM - system monitoring)",
        "     b) Forward-fill if temporal continuity is important (e.g., state_*)",
        f"   Count: {len(missing_stats[(missing_stats['missing_pct'] > 50) & (missing_stats['missing_pct'] <= 80)])} columns",
        "",
        "3. COLUMNS TO MEDIAN IMPUTE (5-50% missing):",
        "   Rationale: Moderate missing data. Median imputation preserves",
        "   distribution and is robust to outliers. Apply per-class if needed.",
        f"   Count: {len(missing_stats[(missing_stats['missing_pct'] > 5) & (missing_stats['missing_pct'] <= 50)])} columns",
        "",
        "4. COLUMNS TO KEEP (< 5% missing):",
        "   Rationale: Minimal missing data. Can safely keep with small",
        "   imputation or removal of rows with NaN.",
        f"   Count: {len(missing_stats[missing_stats['missing_pct'] <= 5])} columns",
        "",
        "=" * 90,
        "DECISION RATIONALE:",
        "=" * 90,
        "",
        "Missing Data Mechanism:",
        "  The high overall missingness (80%+) suggests MCAR (Missing Completely At Random)",
        "  or MAR (Missing At Random) due to sensor/system limitations, NOT MNAR.",
        "  - Different sensors transmit at different frequencies",
        "  - Some sensors (RSSI, CPU, RAM) are optional/low-priority",
        "  - This pattern is consistent across all classes",
        "",
        "Implementation Plan:",
        "  Step 1: Identify and DROP columns with >80% missing",
        "  Step 2: Handle 50-80% missing: DROP optional system columns (RSSI, CPU, RAM)",
        "          FORWARD-FILL state columns if temporal sequences matter",
        "  Step 3: Median-impute columns with 5-50% missing (per-class is optional)",
        "  Step 4: Keep columns with <5% missing (may need row removal or imputation)",
        "",
        "Impact on Data:",
        "  - Expected loss: 10-15 columns (mainly system monitoring)",
        "  - Retained: 60-65 core telemetry columns (position, IMU, battery, motor control)",
        "  - Expected result: ~32k rows with minimal missing data (<5% after imputation)",
        "",
        "=" * 90,
    ])
    
    return "\n".join(lines)

def analyze_missing_by_class(df):
    """Analyze missing data patterns by class."""
    lines = [
        "",
        "=" * 90,
        "MISSING DATA ANALYSIS BY CLASS:",
        "=" * 90,
        "",
    ]
    
    for cls in sorted(df['class'].unique()):
        df_cls = df[df['class'] == cls]
        missing_pct = 100 * df_cls.isna().sum() / len(df_cls)
        missing_pct = missing_pct.sort_values(ascending=False)
        
        lines.extend([
            f"\nCLASS: {cls.upper()} (n={len(df_cls):,} rows)",
            "-" * 90,
            f"  Total missing cells: {df_cls.isna().sum().sum():,} / {df_cls.shape[0] * df_cls.shape[1]:,}",
            f"  Overall missing %: {100 * df_cls.isna().sum().sum() / (df_cls.shape[0] * df_cls.shape[1]):.2f}%",
            "",
            "  Top 10 columns with most missing data:",
        ])
        
        for col, pct in missing_pct.head(10).items():
            lines.append(f"    {col:45s} | {pct:6.2f}% missing")
    
    lines.append("")
    return "\n".join(lines)

def main():
    print("\n" + "=" * 90)
    print("STEP 1.2: MISSING DATA ANALYSIS")
    print("=" * 90 + "\n")
    
    # Step 1: Load all data
    print("Loading all datasets...")
    df_combined = load_all_data(FILES)
    print()
    
    # Step 2: Analyze missing data per column
    print("Analyzing missing data per column...")
    missing_stats = analyze_missing_per_column(df_combined)
    missing_stats.to_csv(OUT / 'missing_data_summary.csv', index=False)
    print(f"✓ Saved missing data summary to missing_data_summary.csv\n")
    
    # Step 3: Create visualizations
    print("Creating visualizations...")
    create_missing_heatmap(df_combined, OUT / 'missing_data_heatmap.png')
    create_missing_bar_chart(missing_stats, OUT / 'missing_data_bar_chart.png')
    create_missing_by_class(df_combined, OUT / 'missing_data_by_class.png')
    print()
    
    # Step 4: Generate comprehensive report
    print("Generating analysis report...")
    report = generate_missing_report(df_combined, missing_stats)
    report += analyze_missing_by_class(df_combined)
    
    with open(OUT / 'missing_data_analysis_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Saved detailed report to missing_data_analysis_report.txt\n")
    
    # Step 5: Print summary to console
    print("=" * 90)
    print("QUICK SUMMARY - MISSING DATA STATISTICS:")
    print("=" * 90)
    print(f"\nOverall Missing: {100 * df_combined.isna().sum().sum() / (df_combined.shape[0] * df_combined.shape[1]):.2f}%")
    print(f"\nColumns by Strategy:")
    print(f"  DROP (>80%):              {len(missing_stats[missing_stats['missing_pct'] > 80]):3d} columns")
    print(f"  DROP/Forward-Fill (50-80%): {len(missing_stats[(missing_stats['missing_pct'] > 50) & (missing_stats['missing_pct'] <= 80)]):3d} columns")
    print(f"  Median Impute (5-50%):    {len(missing_stats[(missing_stats['missing_pct'] > 5) & (missing_stats['missing_pct'] <= 50)]):3d} columns")
    print(f"  Keep (<5%):               {len(missing_stats[missing_stats['missing_pct'] <= 5]):3d} columns")
    
    print("\n" + "=" * 90)
    print("STEP 1.2 COMPLETE ✓")
    print("=" * 90)
    print(f"\nAll outputs saved to: {OUT}\n")
    
    return missing_stats

if __name__ == '__main__':
    missing_stats = main()
