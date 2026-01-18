"""
Phase 2: Complete pipeline - Stage-1 predictor evaluation and tuning
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
        return False
    
    return True

def main():
    print("="*60)
    print("PHASE 2: STAGE-1 PREDICTOR DEVELOPMENT")
    print("="*60)
    
    # Step 1: Run Stage-1 predictor with default thresholds
    if not run_command("phase2_stage1_predictor.py", "Step 1: Evaluating Stage-1 predictor"):
        return
    
    # Step 2: Visualize results
    if not run_command("phase2_visualize.py", "Step 2: Creating visualizations"):
        return
    
    # Step 3: Optional - Tune thresholds (can be slow)
    print("\n" + "="*60)
    print("⚠️  OPTIONAL: Threshold Tuning")
    print("="*60)
    print("This will test multiple threshold combinations (may take a few minutes)")
    response = input("Run threshold tuning? (y/n): ").strip().lower()
    
    if response == 'y':
        run_command("phase2_tune_thresholds.py", "Step 3: Tuning thresholds (grid search)")
    
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE")
    print("="*60)
    print("\nOutputs:")
    print("  📄 phase2_stage1_results.csv - Detailed predictions and features")
    print("  📊 phase2_stage1_analysis.png - Performance visualizations")
    if response == 'y':
        print("  📄 phase2_threshold_tuning.csv - Grid search results")
    print("\n🎯 Next: Analyze results and adjust thresholds if needed")
    print("   Target: >90% accuracy with low FPR")

if __name__ == "__main__":
    main()
