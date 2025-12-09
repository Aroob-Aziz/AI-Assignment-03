"""
STEP 1.6: FEATURE CORRELATION ANALYSIS
========================================

This script performs comprehensive correlation analysis:
- Compute correlation matrix (Pearson, Spearman)
- Create detailed correlation heatmaps
- Identify highly correlated feature pairs (>0.95)
- Analyze correlation with target class
- Detect multicollinearity issues
- Recommend redundant features for removal
- Visualize feature relationships

Strategy:
1. Load scaled dataset from Step 1.5
2. Compute correlation matrices (Pearson for linear, Spearman for monotonic)
3. Identify and visualize highly correlated pairs
4. Analyze class-wise correlations
5. Recommend features to drop based on redundancy
6. Document rationale for retention/removal

Outputs saved to: output/step_1_6_correlation_analysis/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings

warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'step_1_6_correlation_analysis'
OUT.mkdir(parents=True, exist_ok=True)

SCALED_DATA = ROOT / 'output' / 'step_1_5_normalization' / 'scaled_dataset.csv'

def load_scaled_data():
    """Load scaled dataset from Step 1.5."""
    df = pd.read_csv(SCALED_DATA)
    print(f"✓ Loaded scaled dataset: {df.shape}")
    return df

def compute_correlation_matrices(df):
    """Compute Pearson and Spearman correlation matrices."""
    # Select only numeric columns (exclude 'class')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'sample_index' in numeric_cols:
        numeric_cols.remove('sample_index')  # Don't correlate with sequential index
    
    df_numeric = df[numeric_cols]
    
    # Pearson correlation (linear relationships)
    corr_pearson = df_numeric.corr(method='pearson')
    
    # Spearman correlation (monotonic relationships)
    corr_spearman = df_numeric.corr(method='spearman')
    
    print(f"✓ Computed Pearson correlation: {corr_pearson.shape}")
    print(f"✓ Computed Spearman correlation: {corr_spearman.shape}")
    
    return corr_pearson, corr_spearman, numeric_cols

def find_highly_correlated_pairs(corr_matrix, threshold=0.95):
    """Find feature pairs with correlation above threshold."""
    corr_pairs = []
    
    # Get upper triangle of correlation matrix (avoid duplicates)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) >= threshold:
                corr_pairs.append({
                    'feature_1': col1,
                    'feature_2': col2,
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val)
                })
    
    # Sort by absolute correlation
    corr_pairs = sorted(corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)
    return corr_pairs

def create_full_heatmap(corr_matrix, output_path, title):
    """Create full correlation heatmap."""
    fig, ax = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                xticklabels=True, yticklabels=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {title}")

def create_top_correlations_heatmap(corr_matrix, output_path, n_top=20):
    """Create heatmap of top correlated features."""
    # Get average correlation for each feature
    avg_corr = corr_matrix.abs().mean(axis=1).sort_values(ascending=False)
    top_features = avg_corr.head(n_top).index.tolist()
    
    corr_subset = corr_matrix.loc[top_features, top_features]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'},
                annot_kws={'size': 8})
    
    ax.set_title(f'Top {n_top} Most Correlated Features (by average absolute correlation)', 
                fontsize=12, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved top correlations heatmap")

def analyze_class_correlations(df, numeric_cols):
    """Analyze correlation of features with target class."""
    # Convert class to numeric
    class_mapping = {cls: idx for idx, cls in enumerate(sorted(df['class'].unique()))}
    class_numeric = df['class'].map(class_mapping)
    
    class_corr = []
    for col in numeric_cols:
        # Pearson correlation with class
        corr_val, p_val = pearsonr(df[col].fillna(df[col].mean()), class_numeric)
        class_corr.append({
            'feature': col,
            'correlation_with_class': corr_val,
            'abs_correlation': abs(corr_val),
            'p_value': p_val
        })
    
    class_corr_df = pd.DataFrame(class_corr).sort_values('abs_correlation', ascending=False)
    return class_corr_df

def create_class_correlation_plot(class_corr_df, output_path):
    """Visualize feature correlation with target class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = 20
    top_features = class_corr_df.head(top_n)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['correlation_with_class']]
    ax.barh(range(len(top_features)), top_features['correlation_with_class'], color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=9)
    ax.set_xlabel('Correlation with Target Class', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Features by Correlation with Target Class', fontsize=12, fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class correlation plot")

def detect_multicollinearity(corr_matrix, numeric_cols, threshold=0.95):
    """
    Detect multicollinearity and recommend features to drop.
    
    Strategy:
    - For each pair of highly correlated features (>0.95)
    - Keep the one with higher average absolute correlation with other features
    - Or keep the one with higher correlation with target class
    """
    corr_pairs = find_highly_correlated_pairs(corr_matrix, threshold)
    
    # Track features to drop
    features_to_drop = set()
    redundancy_info = []
    
    for pair in corr_pairs:
        feat1 = pair['feature_1']
        feat2 = pair['feature_2']
        
        # Skip if already marked for removal
        if feat1 in features_to_drop or feat2 in features_to_drop:
            continue
        
        # Get average correlation with other features
        avg_corr_1 = corr_matrix[feat1].abs().mean()
        avg_corr_2 = corr_matrix[feat2].abs().mean()
        
        # Keep feature with lower average correlation (less redundant)
        if avg_corr_1 < avg_corr_2:
            drop_feat = feat1
            keep_feat = feat2
        else:
            drop_feat = feat2
            keep_feat = feat1
        
        redundancy_info.append({
            'pair': f"{feat1} <-> {feat2}",
            'correlation': pair['correlation'],
            'keep': keep_feat,
            'drop': drop_feat,
            'rationale': f"Dropped {drop_feat} (avg_corr={min(avg_corr_1, avg_corr_2):.3f}) in favor of {keep_feat}"
        })
        
        features_to_drop.add(drop_feat)
    
    return features_to_drop, redundancy_info, corr_pairs

def create_redundancy_report(corr_pairs, redundancy_info, features_to_drop):
    """Create detailed redundancy report."""
    lines = [
        "=" * 120,
        "MULTICOLLINEARITY AND REDUNDANCY ANALYSIS",
        "=" * 120,
        "",
        f"Threshold for high correlation: >0.95",
        f"Total highly correlated pairs found: {len(corr_pairs)}",
        f"Features recommended for removal: {len(features_to_drop)}",
        "",
        "=" * 120,
        "HIGHLY CORRELATED FEATURE PAIRS (correlation >= 0.95):",
        "=" * 120,
        ""
    ]
    
    for pair in corr_pairs:
        lines.append(f"{pair['feature_1']:40s} <--> {pair['feature_2']:40s}")
        lines.append(f"  Correlation: {pair['correlation']:7.4f} (absolute: {pair['abs_correlation']:.4f})")
        lines.append("")
    
    lines.extend([
        "=" * 120,
        "REDUNDANCY RESOLUTION:",
        "=" * 120,
        "",
        "Decision logic: For each correlated pair, keep the feature with lower average correlation",
        "to other features (less redundant with dataset). This preserves diversity.",
        "",
    ])
    
    for info in redundancy_info:
        lines.append(f"Pair: {info['pair']}")
        lines.append(f"  Correlation: {info['correlation']:.4f}")
        lines.append(f"  Keep: {info['keep']}")
        lines.append(f"  Drop: {info['drop']}")
        lines.append(f"  Rationale: {info['rationale']}")
        lines.append("")
    
    lines.extend([
        "=" * 120,
        "FEATURES RECOMMENDED FOR REMOVAL:",
        "=" * 120,
        "",
    ])
    
    if features_to_drop:
        for feat in sorted(features_to_drop):
            lines.append(f"  - {feat}")
    else:
        lines.append("  No features recommended for removal (no pairs exceed 0.95 threshold)")
    
    lines.extend([
        "",
        "=" * 120,
        "RECOMMENDATION:",
        "=" * 120,
        "",
    ])
    
    if features_to_drop:
        lines.append(f"✓ Remove {len(features_to_drop)} redundant features in Step 1.7")
        lines.append("  This improves model efficiency and interpretability without losing information.")
    else:
        lines.append("✓ No features removed - feature set is sufficiently diverse")
    
    lines.append("")
    lines.append("=" * 120)
    
    return "\n".join(lines)

def generate_correlation_report(corr_pearson, class_corr_df, features_to_drop, redundancy_info):
    """Generate comprehensive correlation analysis report."""
    
    lines = [
        "=" * 120,
        "STEP 1.6: FEATURE CORRELATION ANALYSIS REPORT",
        "=" * 120,
        "",
        f"Pearson Correlation Matrix: {corr_pearson.shape}",
        f"Total numeric features: {len(corr_pearson.columns)}",
        "",
        "=" * 120,
        "CORRELATION ANALYSIS SUMMARY:",
        "=" * 120,
        "",
        "Pearson Correlation (Linear Relationships):",
        f"  - Measures linear dependency between features",
        f"  - Range: [-1, 1] where ±1 indicates perfect correlation",
        "",
        "Spearman Correlation (Monotonic Relationships):",
        f"  - Measures monotonic (rank-based) relationships",
        f"  - More robust to non-linear monotonic patterns",
        "",
    ]
    
    # Feature-class correlations
    lines.extend([
        "=" * 120,
        "TOP 15 FEATURES BY CORRELATION WITH TARGET CLASS:",
        "=" * 120,
        "",
    ])
    
    for idx, row in class_corr_df.head(15).iterrows():
        lines.append(f"{row['feature']:40s} | Correlation: {row['correlation_with_class']:7.4f} | "
                    f"P-value: {row['p_value']:.2e}")
    
    lines.extend([
        "",
        "=" * 120,
        "FEATURE REDUNDANCY ANALYSIS:",
        "=" * 120,
        "",
    ])
    
    lines.append(f"High correlation threshold: 0.95")
    lines.append(f"Total feature pairs with correlation >= 0.95: {len(redundancy_info)}")
    
    if len(redundancy_info) > 0:
        lines.append(f"Recommended features to remove: {len(features_to_drop)}")
        lines.append("")
        lines.append("Redundant feature pairs:")
        for info in redundancy_info[:10]:  # Show top 10
            lines.append(f"  {info['pair']}: corr = {info['correlation']:.4f}")
            lines.append(f"    Keep {info['keep']}, drop {info['drop']}")
    else:
        lines.append("No highly redundant pairs found (all pairs have correlation < 0.95)")
    
    lines.extend([
        "",
        "=" * 120,
        "INTERPRETATION GUIDELINES:",
        "=" * 120,
        "",
        "1. FEATURE-CLASS CORRELATION:",
        "   - Features with |correlation| > 0.3: Moderate discriminative power",
        "   - Features with |correlation| > 0.5: Strong relationship with class",
        "   - Low correlation features may still be valuable via interactions",
        "",
        "2. FEATURE-FEATURE CORRELATION:",
        "   - Correlation > 0.95: Highly redundant (candidate for removal)",
        "   - Correlation 0.7-0.95: Strong dependency (consider domain meaning)",
        "   - Correlation < 0.7: Good diversity (retain for coverage)",
        "",
        "3. MULTICOLLINEARITY IMPACT:",
        "   - Affects: Linear models (SVM), logistic regression, neural networks",
        "   - Less impact: Tree-based models (XGBoost), random forests",
        "   - Solution: Remove redundant features or apply dimensionality reduction (PCA)",
        "",
        "=" * 120,
        "DATA QUALITY INSIGHTS:",
        "=" * 120,
        "",
        "✓ Correlation patterns reveal feature relationships",
        "✓ High class correlations identify predictive features",
        "✓ Redundancy detection prevents overfitting from repeated information",
        "✓ Domain knowledge should validate removal recommendations",
        "",
        "=" * 120,
    ])
    
    return "\n".join(lines)

def main():
    print("\n" + "=" * 120)
    print("STEP 1.6: FEATURE CORRELATION ANALYSIS")
    print("=" * 120 + "\n")
    
    # Step 1: Load scaled data
    print("Loading scaled dataset from Step 1.5...")
    df = load_scaled_data()
    print()
    
    # Step 2: Compute correlations
    print("Computing correlation matrices...")
    corr_pearson, corr_spearman, numeric_cols = compute_correlation_matrices(df)
    print()
    
    # Step 3: Save correlation matrices
    print("Saving correlation matrices...")
    corr_pearson.to_csv(OUT / 'pearson_correlation_matrix.csv')
    corr_spearman.to_csv(OUT / 'spearman_correlation_matrix.csv')
    print(f"✓ Saved correlation matrices\n")
    
    # Step 4: Analyze class correlations
    print("Analyzing feature-class correlations...")
    class_corr_df = analyze_class_correlations(df, numeric_cols)
    class_corr_df.to_csv(OUT / 'class_correlations.csv', index=False)
    print(f"✓ Analyzed {len(class_corr_df)} features\n")
    
    # Step 5: Detect redundancy
    print("Detecting multicollinearity and redundancy...")
    features_to_drop, redundancy_info, corr_pairs = detect_multicollinearity(corr_pearson, numeric_cols, threshold=0.95)
    print(f"✓ Found {len(corr_pairs)} highly correlated pairs")
    print(f"✓ Recommended {len(features_to_drop)} features for removal\n")
    
    # Step 6: Save redundancy info
    if redundancy_info:
        redundancy_df = pd.DataFrame(redundancy_info)
        redundancy_df.to_csv(OUT / 'redundancy_analysis.csv', index=False)
    print(f"✓ Saved redundancy analysis\n")
    
    # Step 7: Create visualizations
    print("Creating correlation visualizations...")
    create_full_heatmap(corr_pearson, OUT / 'pearson_correlation_heatmap.png', 
                       'Pearson Correlation Matrix - All Features')
    create_full_heatmap(corr_spearman, OUT / 'spearman_correlation_heatmap.png', 
                       'Spearman Correlation Matrix - All Features')
    create_top_correlations_heatmap(corr_pearson, OUT / 'top_20_correlations_heatmap.png', n_top=20)
    create_class_correlation_plot(class_corr_df, OUT / 'class_correlation_barplot.png')
    print()
    
    # Step 8: Generate reports
    print("Generating correlation reports...")
    redundancy_report = create_redundancy_report(corr_pairs, redundancy_info, features_to_drop)
    with open(OUT / 'redundancy_report.txt', 'w', encoding='utf-8') as f:
        f.write(redundancy_report)
    
    correlation_report = generate_correlation_report(corr_pearson, class_corr_df, features_to_drop, redundancy_info)
    with open(OUT / 'correlation_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(correlation_report)
    print(f"✓ Saved detailed reports\n")
    
    # Step 9: Summary
    print("=" * 120)
    print("QUICK SUMMARY - CORRELATION ANALYSIS:")
    print("=" * 120)
    
    print(f"\nFeature Statistics:")
    print(f"  Total numeric features analyzed: {len(numeric_cols)}")
    print(f"  Target classes: {', '.join(sorted(df['class'].unique()))}")
    
    print(f"\nHighly Correlated Pairs (correlation >= 0.95):")
    print(f"  Total pairs found: {len(corr_pairs)}")
    if len(corr_pairs) > 0:
        print(f"\n  Top 5 most correlated pairs:")
        for pair in corr_pairs[:5]:
            print(f"    {pair['feature_1']} <--> {pair['feature_2']}: {pair['correlation']:.4f}")
    
    print(f"\nRedundancy Recommendation:")
    print(f"  Features to remove: {len(features_to_drop)}")
    if features_to_drop:
        print(f"  Features: {', '.join(sorted(features_to_drop))}")
    
    print(f"\nClass Correlation (Top 5):")
    for idx, row in class_corr_df.head(5).iterrows():
        print(f"  {row['feature']:40s}: {row['correlation_with_class']:7.4f}")
    
    print("\n" + "=" * 120)
    print("STEP 1.6 COMPLETE ✓")
    print("=" * 120)
    print(f"\nAll outputs saved to: {OUT}\n")
    
    return df, corr_pearson, features_to_drop

if __name__ == '__main__':
    df, corr, features_drop = main()
