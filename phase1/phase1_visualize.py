"""
Phase 1: Sanity check visualizations
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LABEL_CSV = "page_labels.csv"
ALPHA = 0.7

def visualize_results():
    """Create sanity check plots"""
    
    try:
        df = pd.read_csv(LABEL_CSV)
    except FileNotFoundError:
        print(f"❌ Error: '{LABEL_CSV}' not found. Run phase1_label_pages.py first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 1: Page Compression Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall compression ratio distribution
    ax1 = axes[0, 0]
    ax1.hist(df['compression_ratio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(ALPHA, color='red', linestyle='--', linewidth=2, label=f'α = {ALPHA}')
    ax1.set_xlabel('Compression Ratio', fontsize=12)
    ax1.set_ylabel('Number of Pages', fontsize=12)
    ax1.set_title('Compression Ratio Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Compression ratio by page type
    ax2 = axes[0, 1]
    page_types = df['page_type'].unique()
    positions = np.arange(len(page_types))
    
    for i, ptype in enumerate(sorted(page_types)):
        data = df[df['page_type'] == ptype]['compression_ratio']
        ax2.violinplot([data], positions=[i], widths=0.7, showmeans=True, showmedians=True)
    
    ax2.axhline(ALPHA, color='red', linestyle='--', linewidth=2, label=f'α = {ALPHA}')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(sorted(page_types), rotation=45)
    ax2.set_ylabel('Compression Ratio', fontsize=12)
    ax2.set_title('Compression Ratio by Page Type', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Label distribution
    ax3 = axes[1, 0]
    label_counts = df['label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax3.bar(label_counts.index, label_counts.values, color=colors, edgecolor='black', alpha=0.7)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Label Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (label, count) in enumerate(label_counts.items()):
        ax3.text(i, count + 50, str(count), ha='center', fontsize=11, fontweight='bold')
    
    # Plot 4: Compressibility by page type (stacked bar)
    ax4 = axes[1, 1]
    comp_by_type = df.groupby(['page_type', 'label']).size().unstack(fill_value=0)
    comp_by_type.plot(kind='bar', stacked=True, ax=ax4, color=['#2ecc71', '#e74c3c'], 
                      edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Page Type', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Compressibility by Page Type', fontsize=13, fontweight='bold')
    ax4.legend(title='Label')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('phase1_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to 'phase1_analysis.png'")
    plt.show()
    
    # Sanity checks
    print("\n" + "="*50)
    print("SANITY CHECKS")
    print("="*50)
    
    zero_ratio = df[df['page_type'] == 'zero']['compression_ratio'].mean()
    rand_ratio = df[df['page_type'] == 'rand']['compression_ratio'].mean()
    
    print(f"✓ Zero pages avg ratio: {zero_ratio:.4f} (should be ~0)")
    print(f"✓ Random pages avg ratio: {rand_ratio:.4f} (should be ~1)")
    
    if zero_ratio < 0.1:
        print("  ✅ Zero pages compress well")
    else:
        print("  ⚠️  Zero pages not compressing as expected")
    
    if rand_ratio > 0.95:
        print("  ✅ Random pages are incompressible")
    else:
        print("  ⚠️  Random pages compressing unexpectedly")

if __name__ == "__main__":
    visualize_results()
