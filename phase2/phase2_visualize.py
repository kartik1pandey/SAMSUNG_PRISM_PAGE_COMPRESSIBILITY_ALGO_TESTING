"""
Phase 2: Visualize Stage-1 predictor performance
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_stage1_results():
    """Create visualizations for Stage-1 predictor analysis"""
    
    try:
        df = pd.read_csv("phase2_stage1_results.csv")
    except FileNotFoundError:
        print("❌ Error: 'phase2_stage1_results.csv' not found. Run phase2_stage1_predictor.py first.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Phase 2: Stage-1 Predictor Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix Heatmap
    ax1 = axes[0, 0]
    
    # Create confusion matrix with all labels present
    all_labels = ['compressible', 'incompressible']
    conf_matrix = pd.crosstab(
        df["label"], 
        df["stage1_pred"],
        rownames=['Actual'],
        colnames=['Predicted'],
        dropna=False
    )
    
    # Ensure both labels exist in matrix
    for label in all_labels:
        if label not in conf_matrix.columns:
            conf_matrix[label] = 0
        if label not in conf_matrix.index:
            conf_matrix.loc[label] = 0
    
    # Reorder to standard format
    conf_matrix = conf_matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)
    
    im = ax1.imshow(conf_matrix.values, cmap='Blues', aspect='auto')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(all_labels, rotation=45, ha='right')
    ax1.set_yticklabels(all_labels)
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, int(conf_matrix.values[i, j]),
                           ha="center", va="center", color="black", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax1)
    
    # Plot 2: Accuracy by page type
    ax2 = axes[0, 1]
    acc_by_type = df.groupby('page_type').apply(
        lambda x: (x['stage1_pred'] == x['label']).mean()
    ).sort_values()
    
    colors = ['#e74c3c' if v < 0.9 else '#2ecc71' for v in acc_by_type.values]
    ax2.barh(acc_by_type.index, acc_by_type.values, color=colors, edgecolor='black', alpha=0.7)
    ax2.axvline(0.9, color='orange', linestyle='--', linewidth=2, label='90% target')
    ax2.set_xlabel('Accuracy', fontsize=11)
    ax2.set_title('Accuracy by Page Type', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Feature distributions (distinct bytes)
    ax3 = axes[0, 2]
    for label in ['compressible', 'incompressible']:
        subset = df[df['label'] == label]['feat_distinct']
        ax3.hist(subset, bins=30, alpha=0.6, label=label, edgecolor='black')
    ax3.set_xlabel('Distinct Bytes', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Feature: Distinct Bytes', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature distributions (max freq ratio)
    ax4 = axes[1, 0]
    for label in ['compressible', 'incompressible']:
        subset = df[df['label'] == label]['feat_max_freq_ratio']
        ax4.hist(subset, bins=30, alpha=0.6, label=label, edgecolor='black')
    ax4.set_xlabel('Max Frequency Ratio', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Feature: Max Frequency Ratio', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Feature distributions (zero ratio)
    ax5 = axes[1, 1]
    for label in ['compressible', 'incompressible']:
        subset = df[df['label'] == label]['feat_zero_ratio']
        ax5.hist(subset, bins=30, alpha=0.6, label=label, edgecolor='black')
    ax5.set_xlabel('Zero Ratio', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Feature: Zero Ratio', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Feature distributions (run count)
    ax6 = axes[1, 2]
    for label in ['compressible', 'incompressible']:
        subset = df[df['label'] == label]['feat_run_count']
        ax6.hist(subset, bins=30, alpha=0.6, label=label, edgecolor='black')
    ax6.set_xlabel('Run Count', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Feature: Run Count', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase2_stage1_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to 'phase2_stage1_analysis.png'")
    plt.show()
    
    # Error analysis
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    errors = df[df['stage1_pred'] != df['label']]
    print(f"Total errors: {len(errors)} / {len(df)} ({len(errors)/len(df)*100:.2f}%)")
    print()
    
    if len(errors) > 0:
        print("Errors by page type:")
        error_by_type = errors.groupby('page_type').size().sort_values(ascending=False)
        for ptype, count in error_by_type.items():
            total = len(df[df['page_type'] == ptype])
            print(f"  {ptype:10s}: {count:4d} / {total:4d} ({count/total*100:.2f}%)")
    else:
        print("🎉 Perfect predictions! No errors found.")

if __name__ == "__main__":
    visualize_stage1_results()
