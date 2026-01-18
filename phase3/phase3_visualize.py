"""
Phase 3: Visualize two-stage predictor performance
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_two_stage_results():
    """Create visualizations for two-stage predictor analysis"""
    
    try:
        df = pd.read_csv("phase3/phase3_two_stage_results.csv")
    except FileNotFoundError:
        print("❌ Error: 'phase3/phase3_two_stage_results.csv' not found.")
        print("   Run phase3_two_stage_predictor.py first.")
        return
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Phase 3: Two-Stage Predictor Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    all_labels = ['compressible', 'incompressible']
    conf_matrix = pd.crosstab(
        df["label"], 
        df["two_stage_pred"],
        rownames=['Actual'],
        colnames=['Predicted'],
        dropna=False
    )
    
    for label in all_labels:
        if label not in conf_matrix.columns:
            conf_matrix[label] = 0
        if label not in conf_matrix.index:
            conf_matrix.loc[label] = 0
    
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
    
    # Plot 2: Stage usage distribution
    ax2 = fig.add_subplot(gs[0, 1])
    stage_counts = df['stage_used'].value_counts()
    colors = ['#3498db', '#e74c3c']
    ax2.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Stage Usage Distribution', fontsize=12, fontweight='bold')
    
    # Plot 3: Accuracy by page type
    ax3 = fig.add_subplot(gs[0, 2])
    acc_by_type = df.groupby('page_type', group_keys=False).apply(
        lambda x: (x['two_stage_pred'] == x['label']).mean()
    ).sort_values()
    
    colors = ['#e74c3c' if v < 0.9 else '#2ecc71' for v in acc_by_type.values]
    ax3.barh(acc_by_type.index, acc_by_type.values, color=colors, edgecolor='black', alpha=0.7)
    ax3.axvline(0.9, color='orange', linestyle='--', linewidth=2, label='90% target')
    ax3.set_xlabel('Accuracy', fontsize=11)
    ax3.set_title('Accuracy by Page Type', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Stage-1 score distribution by label
    ax4 = fig.add_subplot(gs[1, 0])
    for label in ['compressible', 'incompressible']:
        subset = df[df['label'] == label]['stage1_score']
        ax4.hist(subset, bins=30, alpha=0.6, label=label, edgecolor='black')
    ax4.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Low threshold')
    ax4.axvline(2.5, color='green', linestyle='--', linewidth=2, label='High threshold')
    ax4.set_xlabel('Stage-1 Score', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Stage-1 Score Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Stage-2 compression ratio (for pages that went to Stage-2)
    ax5 = fig.add_subplot(gs[1, 1])
    stage2_data = df[df['stage_used'] == 'stage2']
    if len(stage2_data) > 0:
        for label in ['compressible', 'incompressible']:
            subset = stage2_data[stage2_data['label'] == label]['stage2_ratio']
            if len(subset) > 0:
                ax5.hist(subset, bins=20, alpha=0.6, label=label, edgecolor='black')
        ax5.axvline(0.75, color='red', linestyle='--', linewidth=2, label='β threshold')
        ax5.set_xlabel('Stage-2 Compression Ratio', fontsize=11)
        ax5.set_ylabel('Count', fontsize=11)
        ax5.set_title('Stage-2 Sample Compression Ratio', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No pages sent to Stage-2', ha='center', va='center', fontsize=12)
        ax5.set_title('Stage-2 Sample Compression Ratio', fontsize=12, fontweight='bold')
    
    # Plot 6: Stage-2 usage by page type
    ax6 = fig.add_subplot(gs[1, 2])
    stage2_by_type = df.groupby('page_type')['stage_used'].apply(
        lambda x: (x == 'stage2').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    ax6.bar(range(len(stage2_by_type)), stage2_by_type.values, 
            color='#9b59b6', edgecolor='black', alpha=0.7)
    ax6.set_xticks(range(len(stage2_by_type)))
    ax6.set_xticklabels(stage2_by_type.index, rotation=45, ha='right')
    ax6.set_ylabel('Stage-2 Usage (%)', fontsize=11)
    ax6.set_title('Stage-2 Usage by Page Type', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Comparison - Stage-1 only vs Two-Stage
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Simulate Stage-1 only
    df['stage1_only_pred'] = df['stage1_score'].apply(
        lambda x: "compressible" if x > 0.0 else "incompressible"
    )
    
    comparison_data = {
        'Stage-1 Only': (df['stage1_only_pred'] == df['label']).mean(),
        'Two-Stage': (df['two_stage_pred'] == df['label']).mean()
    }
    
    colors = ['#3498db', '#2ecc71']
    bars = ax7.bar(comparison_data.keys(), comparison_data.values(), 
                   color=colors, edgecolor='black', alpha=0.7)
    ax7.set_ylabel('Accuracy', fontsize=11)
    ax7.set_title('Stage-1 Only vs Two-Stage', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, comparison_data.values()):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 8: Error analysis by stage
    ax8 = fig.add_subplot(gs[2, 1])
    errors = df[df['two_stage_pred'] != df['label']]
    if len(errors) > 0:
        error_by_stage = errors['stage_used'].value_counts()
        ax8.bar(error_by_stage.index, error_by_stage.values, 
                color='#e74c3c', edgecolor='black', alpha=0.7)
        ax8.set_ylabel('Error Count', fontsize=11)
        ax8.set_title('Errors by Stage', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
    else:
        ax8.text(0.5, 0.5, 'No errors!', ha='center', va='center', fontsize=14, fontweight='bold')
        ax8.set_title('Errors by Stage', fontsize=12, fontweight='bold')
    
    # Plot 9: Metrics summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate metrics
    tp = ((df["label"] == "compressible") & (df["two_stage_pred"] == "compressible")).sum()
    tn = ((df["label"] == "incompressible") & (df["two_stage_pred"] == "incompressible")).sum()
    fp = ((df["label"] == "incompressible") & (df["two_stage_pred"] == "compressible")).sum()
    fn = ((df["label"] == "compressible") & (df["two_stage_pred"] == "incompressible")).sum()
    
    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    stage2_rate = (df['stage_used'] == 'stage2').sum() / len(df) * 100
    
    metrics_text = f"""
    PERFORMANCE METRICS
    
    Accuracy:     {accuracy:.4f}
    Precision:    {precision:.4f}
    Recall:       {recall:.4f}
    FPR:          {fpr:.4f}
    
    Stage-2 Rate: {stage2_rate:.1f}%
    
    Total Pages:  {len(df)}
    Errors:       {len(errors)}
    """
    
    ax9.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig('phase3/phase3_two_stage_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to 'phase3/phase3_two_stage_analysis.png'")
    plt.show()
    
    # Error analysis
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    errors = df[df['two_stage_pred'] != df['label']]
    print(f"Total errors: {len(errors)} / {len(df)} ({len(errors)/len(df)*100:.2f}%)")
    print()
    
    if len(errors) > 0:
        print("Errors by page type:")
        error_by_type = errors.groupby('page_type').size().sort_values(ascending=False)
        for ptype, count in error_by_type.items():
            total = len(df[df['page_type'] == ptype])
            print(f"  {ptype:10s}: {count:4d} / {total:4d} ({count/total*100:.2f}%)")
        
        print("\nErrors by stage:")
        error_by_stage = errors.groupby('stage_used').size()
        for stage, count in error_by_stage.items():
            print(f"  {stage:10s}: {count:4d}")
    else:
        print("🎉 Perfect predictions! No errors found.")

if __name__ == "__main__":
    visualize_two_stage_results()
