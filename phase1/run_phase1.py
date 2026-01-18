"""
Phase 1: Complete pipeline - Generate, Label, Visualize
"""
import subprocess
import sys

def run_command(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*60)
    print(f"🚀 {description}")
    print("="*60)
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_name}")
        sys.exit(1)
    
    return result.returncode == 0

def main():
    print("="*60)
    print("PHASE 1: DATA COLLECTION & LABELING")
    print("="*60)
    
    # Step 1: Generate pages
    run_command("phase1_generate_pages.py", "Step 1: Generating memory pages")
    
    # Step 2: Label pages
    run_command("phase1_label_pages.py", "Step 2: Compressing and labeling pages")
    
    # Step 3: Visualize
    run_command("phase1_visualize.py", "Step 3: Creating visualizations")
    
    print("\n" + "="*60)
    print("✅ PHASE 1 COMPLETE")
    print("="*60)
    print("\nOutputs:")
    print("  📁 pages/ - Raw page files")
    print("  📄 page_labels.csv - Labeled dataset")
    print("  📊 phase1_analysis.png - Sanity check plots")
    print("\n🎯 Ready for Phase 2: Stage-1 Predictor Development")

if __name__ == "__main__":
    main()
