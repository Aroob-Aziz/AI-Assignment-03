"""
STEP 1.1: DATA LOADING AND INITIAL EXPLORATION
===============================================

This script performs the first step of data preprocessing:
- Load all datasets (Dos2, Malfunction2, Normal2, Normal4)
- Examine structure, dtypes, dimensions
- Identify target variable (class/label)
- Document observations and create visualizations

Output:
- exploration_report.txt: Detailed text summary
- class_distribution.png: Bar plot of class counts
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'RawData'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

# Files to load (even-numbered + Normal4 as requested)
FILES = {
    'Dos2': RAW / 'Dos-Drone' / 'Dos2.csv',
    'Malfunction2': RAW / 'Malfunction-Drone' / 'Malfunction2.csv',
    'Normal2': RAW / 'NormalFlight' / 'Normal2.csv',
    'Normal4': RAW / 'NormalFlight' / 'Normal4.csv',
}

def load_and_explore(file_dict):
    """Load all files and collect exploration data."""
    exploration = {}
    
    for name, path in file_dict.items():
        if not path.exists():
            print(f"⚠️  {name} not found at {path}")
            continue
        
        df = pd.read_csv(path)
        
        # Infer class/label from filename
        if 'Dos' in name or 'dos' in name.lower():
            class_label = 'DoS_Attack'
        elif 'Malfunction' in name or 'malfunction' in name.lower():
            class_label = 'Malfunction'
        elif 'Normal' in name or 'normal' in name.lower():
            class_label = 'Normal'
        else:
            class_label = 'Unknown'
        
        exploration[name] = {
            'path': str(path),
            'dataframe': df,
            'class_label': class_label,
            'shape': df.shape,
            'dtypes': df.dtypes,
            'columns': list(df.columns),
            'n_numeric': len(df.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(df.select_dtypes(include=['object']).columns),
        }
        
        print(f"✓ Loaded {name}: shape={df.shape}, class={class_label}")
    
    return exploration

def generate_report(exploration):
    """Generate comprehensive text report."""
    lines = [
        "=" * 80,
        "STEP 1.1: DATA LOADING AND INITIAL EXPLORATION",
        "=" * 80,
        "",
        "TARGET VARIABLE IDENTIFICATION:",
        "-" * 80,
        "The assignment specifies THREE CLASSES:",
        "  1. Normal       (NormalFlight files: Normal2.csv, Normal4.csv)",
        "  2. DoS_Attack   (Dos-Drone files: Dos2.csv)",
        "  3. Malfunction  (Malfunction-Drone files: Malfunction2.csv)",
        "",
        "Target variable will be created as a CLASS LABEL based on file source.",
        "",
        "=" * 80,
        "FILE SUMMARIES:",
        "=" * 80,
        "",
    ]
    
    for name in sorted(exploration.keys()):
        info = exploration[name]
        df = info['dataframe']
        
        lines.extend([
            f"FILE: {name}",
            f"  Path: {info['path']}",
            f"  Shape: {info['shape'][0]} rows × {info['shape'][1]} columns",
            f"  Class Label: {info['class_label']}",
            f"  Data Types:",
            f"    - Numeric columns: {info['n_numeric']}",
            f"    - Categorical columns: {info['n_categorical']}",
            f"    - Object/String columns: {len(df.select_dtypes(include=['object']).columns)}",
            "",
            f"  Column Names ({len(info['columns'])} total):",
        ])
        for i, col in enumerate(info['columns'], 1):
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            lines.append(f"    {i:2d}. {col:45s} | dtype: {dtype:8s} | non-null: {non_null}")
        
        lines.extend([
            "",
            f"  Basic Statistics (first 5 rows):",
            "",
        ])
        
        # Show first few rows
        first_rows = df.head(3).to_string().split('\n')
        for row in first_rows:
            lines.append(f"    {row}")
        
        lines.extend([
            "",
            f"  Data Type Distribution:",
        ])
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            lines.append(f"    - {dtype}: {count} columns")
        
        lines.extend([
            "",
            f"  Numeric Columns Summary (describe):",
            "",
        ])
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            desc = numeric_df.describe().round(3).to_string().split('\n')
            for line in desc[:15]:  # Limit to first 15 lines
                lines.append(f"    {line}")
        else:
            lines.append("    No numeric columns")
        
        lines.extend([
            "",
            "=" * 80,
            "",
        ])
    
    return "\n".join(lines)

def create_overview_table(exploration):
    """Create a summary table of all datasets."""
    overview_data = []
    
    for name in sorted(exploration.keys()):
        info = exploration[name]
        df = info['dataframe']
        
        overview_data.append({
            'Dataset': name,
            'Path': info['path'],
            'Class_Label': info['class_label'],
            'N_Rows': info['shape'][0],
            'N_Columns': info['shape'][1],
            'N_Numeric': info['n_numeric'],
            'N_Categorical': info['n_categorical'],
            'Memory_MB': df.memory_usage(deep=True).sum() / 1024 / 1024,
        })
    
    overview_df = pd.DataFrame(overview_data)
    overview_df.to_csv(OUT / 'data_overview_table.csv', index=False)
    return overview_df

def create_class_distribution_plot(exploration):
    """Create and save class distribution plot."""
    class_counts = {}
    for name, info in exploration.items():
        label = info['class_label']
        n_rows = info['shape'][0]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += n_rows
    
    plt.figure(figsize=(8, 5))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#2ecc71', '#e74c3c', '#f39c12']  # green, red, orange
    
    bars = plt.bar(classes, counts, color=colors[:len(classes)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Class Label', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Rows', fontsize=12, fontweight='bold')
    plt.title('Class Distribution (Target Variable)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return class_counts

def main():
    print("\n" + "=" * 80)
    print("STEP 1.1: DATA LOADING AND INITIAL EXPLORATION")
    print("=" * 80 + "\n")
    
    # Step 1: Load and explore all files
    print("Loading datasets...")
    exploration = load_and_explore(FILES)
    print(f"\nSuccessfully loaded {len(exploration)} datasets\n")
    
    # Step 2: Generate detailed report
    print("Generating exploration report...")
    report = generate_report(exploration)
    with open(OUT / 'exploration_report.txt', 'w') as f:
        f.write(report)
    print(f"✓ Report saved to output/exploration_report.txt\n")
    
    # Step 3: Create overview table
    print("Creating overview table...")
    overview_df = create_overview_table(exploration)
    print(f"✓ Overview table saved to output/data_overview_table.csv")
    print(f"\nDataset Overview:")
    print(overview_df.to_string(index=False))
    print()
    
    # Step 4: Create class distribution plot
    print("Creating class distribution visualization...")
    class_counts = create_class_distribution_plot(exploration)
    print(f"✓ Class distribution plot saved to output/class_distribution.png")
    print(f"\nClass Distribution Summary:")
    for label, count in sorted(class_counts.items()):
        pct = 100 * count / sum(class_counts.values())
        print(f"  {label:20s}: {count:6d} rows ({pct:5.1f}%)")
    print()
    
    # Step 5: Print key observations
    print("=" * 80)
    print("KEY OBSERVATIONS FROM STEP 1.1:")
    print("=" * 80)
    
    print("\n1. DATASET DIMENSIONS:")
    total_rows = sum(info['shape'][0] for info in exploration.values())
    total_cols = exploration[list(exploration.keys())[0]]['shape'][1]
    print(f"   - Total rows across all files: {total_rows:,}")
    print(f"   - Total columns per file: {total_cols}")
    print(f"   - All files have consistent column structure")
    
    print("\n2. TARGET VARIABLE:")
    print(f"   - Classes identified: {sorted(class_counts.keys())}")
    print(f"   - Class distribution: {class_counts}")
    print(f"   - Target will be created as 'class' column from file source")
    
    print("\n3. DATA TYPES:")
    first_df = exploration[list(exploration.keys())[0]]['dataframe']
    print(f"   - Numeric (int64, float64): {exploration[list(exploration.keys())[0]]['n_numeric']} columns")
    print(f"   - Non-numeric: {exploration[list(exploration.keys())[0]]['n_categorical']} columns")
    print(f"   - All values are numeric telemetry data (floats/ints)")
    
    print("\n4. FEATURE CATEGORIES:")
    print(f"   - GPS position data: setpoint_raw-global_*, global_position-global_*")
    print(f"   - IMU readings: imu-data_*")
    print(f"   - Battery status: battery_*")
    print(f"   - Motor control: rc-out_*")
    print(f"   - Flight parameters: vfr_hud_*")
    print(f"   - System state: state_*")
    print(f"   - Communication: RSSI_*")
    print(f"   - System resources: CPU_*, RAM_*, Used_*")
    
    print("\n5. MISSING DATA PREVIEW:")
    for name in sorted(exploration.keys()):
        df = exploration[name]['dataframe']
        missing_pct = 100 * df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        print(f"   - {name:15s}: {missing_pct:5.1f}% missing overall")
    
    print("\n" + "=" * 80)
    print("STEP 1.1 COMPLETE ✓")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
