"""
Phase 3: Complete pipeline - Two-stage predictor evaluation and tuning
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
    print("PHASE 3: TWO-STAGE PREDICTOR DEVELOPMENT")
    print("="*60)
    
    # Step 1: Run two-stage predictor
    if not run_command("phase3/phase3_two_stage_predictor.py", "Step 1: Evaluating two-stage predictor"):
        return
    
    # Step 2: Visualize results
    if not run_command("phase3/phase3_visualize.py", "Step 2: Creating visualizations"):
        return
    
    # Step 3: Benchmark performance
    if not run_command("phase3/phase3_benchmark.py", "Step 3: Benchmarking performance"):
        return
    
    if not run_command("phase3/phase3_benchmark_visualize.py", "Step 4: Benchmark visualizations"):
        return
    
    # Step 4: Optional - Tune thresholds
    print("\n" + "="*60)
    print("⚠️  OPTIONAL: Threshold Tuning")
    print("="*60)
    print("This will test multiple threshold combinations for Stage-1/Stage-2")
    response = input("Run threshold tuning? (y/n): ").strip().lower()
    
    if response == 'y':
        run_command("phase3/phase3_tune_thresholds.py", "Step 5: Tuning thresholds (grid search)")
    
    print("\n" + "="*60)
    print("✅ PHASE 3 COMPLETE")
    print("="*60)
    print("\nOutputs:")
    print("  📄 phase3/phase3_two_stage_results.csv - Detailed predictions")
    print("  📊 phase3/phase3_two_stage_analysis.png - Performance visualizations")
    print("  📄 phase3/phase3_benchmark_results.csv - Benchmark data")
    print("  📊 phase3/phase3_benchmark_analysis.png - Benchmark visualizations")
    print("  📄 phase3/BENCHMARK_SUMMARY.md - Timing analysis")
    if response == 'y':
        print("  📄 phase3/phase3_threshold_tuning.csv - Grid search results")
    print("\n🎯 Two-stage predictor combines:")
    print("   • Fast Stage-1 filtering (32.37 μs per page)")
    print("   • Accurate Stage-2 refinement (46.91 μs when invoked)")
    print("   • Target: >95% accuracy with <15% Stage-2 usage")
    print("\n⚡ Performance:")
    print("   • Stage-1 alone: 1.35x faster than full compression")
    print("   • Two-stage: 94.04% accuracy with 100% recall")

if __name__ == "__main__":
    main()
