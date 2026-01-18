"""
Phase 2B: Evaluate Two-Stage Predictor on Real-World Data
"""
import os
import sys
import numpy as np
import pandas as pd
import lz4.frame
from collections import Counter
from tqdm import tqdm

# Add parent directory to import from phase3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PAGE_SIZE = 4096
SAMPLE_SIZE_STAGE1 = 64
SAMPLE_SIZE_STAGE2 = 256

# Use optimized thresholds from Phase 3
STAGE1_CONFIDENCE_THRESHOLD_HIGH = 2.5
STAGE1_CONFIDENCE_THRESHOLD_LOW = 1.0
BETA_STAGE2 = 0.70

FEATURE_WEIGHTS = {
    "distinct_score": -1.0,
    "max_freq_score": 2.0,
    "zero_score": 1.5,
    "run_score": 1.0
}

# ============== Stage-1 Functions ==============

def sample_page_stage1(page_bytes):
    """Hybrid sampling: first 64 + random 64 bytes"""
    sample1 = page_bytes[:SAMPLE_SIZE_STAGE1]
    offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE_STAGE1)
    sample2 = page_bytes[offset:offset + SAMPLE_SIZE_STAGE1]
    return sample1 + sample2

def extract_features(sample):
    """Extract integer-only features"""
    counts = Counter(sample)
    distinct = len(counts)
    max_freq_ratio = max(counts.values()) / len(sample)
    zero_ratio = counts.get(0, 0) / len(sample)
    run_count = sum(1 for i in range(1, len(sample)) if sample[i] == sample[i-1])
    
    return {
        "distinct": distinct,
        "max_freq_ratio": max_freq_ratio,
        "zero_ratio": zero_ratio,
        "run_count": run_count
    }

def compute_compressibility_score(feats, weights=FEATURE_WEIGHTS):
    """Compute weighted compressibility score"""
    distinct_norm = 1.0 - (feats["distinct"] / 128.0)
    max_freq_norm = feats["max_freq_ratio"]
    zero_norm = feats["zero_ratio"]
    run_norm = min(feats["run_count"] / 50.0, 1.0)
    
    score = (
        weights["distinct_score"] * distinct_norm +
        weights["max_freq_score"] * max_freq_norm +
        weights["zero_score"] * zero_norm +
        weights["run_score"] * run_norm
    )
    
    return score

def stage1_predict(page_bytes):
    """Stage-1 prediction with confidence levels"""
    sample = sample_page_stage1(page_bytes)
    feats = extract_features(sample)
    score = compute_compressibility_score(feats)
    
    if score > STAGE1_CONFIDENCE_THRESHOLD_HIGH:
        return "compressible", score
    elif score < STAGE1_CONFIDENCE_THRESHOLD_LOW:
        return "incompressible", score
    else:
        return "ambiguous", score

# ============== Stage-2 Function ==============

def stage2_predict(page_bytes):
    """Stage-2: Sample compression"""
    ratios = []
    for _ in range(2):
        offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE_STAGE2)
        sample = page_bytes[offset:offset + SAMPLE_SIZE_STAGE2]
        compressed = lz4.frame.compress(sample, compression_level=16)
        ratios.append(len(compressed) / SAMPLE_SIZE_STAGE2)
    
    avg_ratio = np.mean(ratios)
    return "compressible" if avg_ratio <= BETA_STAGE2 else "incompressible"

# ============== Evaluation ==============

def evaluate_two_stage(df, pages_dir):
    """Run two-stage predictor on real-world dataset"""
    
    print("Running Two-Stage Predictor on Real-World Data...")
    print(f"Stage-1 confidence thresholds: [{STAGE1_CONFIDENCE_THRESHOLD_LOW}, {STAGE1_CONFIDENCE_THRESHOLD_HIGH}]")
    print(f"Stage-2 compression threshold: β = {BETA_STAGE2}")
    print()
    
    predictions = []
    stages_used = []
    stage1_scores = []
    
    stage2_invocations = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        page_path = os.path.join(pages_dir, row["page_id"])
        
        with open(page_path, "rb") as f:
            page_bytes = f.read()
        
        # Stage-1
        stage1_decision, stage1_score = stage1_predict(page_bytes)
        
        if stage1_decision == "compressible":
            final_pred = "compressible"
            stage = "stage1"
        elif stage1_decision == "incompressible":
            final_pred = "incompressible"
            stage = "stage1"
        else:  # ambiguous
            stage2_invocations += 1
            final_pred = stage2_predict(page_bytes)
            stage = "stage2"
        
        predictions.append(final_pred)
        stages_used.append(stage)
        stage1_scores.append(stage1_score)
    
    df["two_stage_pred"] = predictions
    df["stage_used"] = stages_used
    df["stage1_score"] = stage1_scores
    
    return df, stage2_invocations

def compute_metrics(df):
    """Compute prediction metrics"""
    accuracy = (df["two_stage_pred"] == df["label"]).mean()
    
    tp = ((df["label"] == "compressible") & (df["two_stage_pred"] == "compressible")).sum()
    tn = ((df["label"] == "incompressible") & (df["two_stage_pred"] == "incompressible")).sum()
    fp = ((df["label"] == "incompressible") & (df["two_stage_pred"] == "compressible")).sum()
    fn = ((df["label"] == "compressible") & (df["two_stage_pred"] == "incompressible")).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def main():
    print("="*70)
    print("PHASE 2B: TWO-STAGE PREDICTOR ON REAL-WORLD DATA")
    print("="*70)
    print()
    
    # Load real-world dataset
    df = pd.read_csv("page_labels_real.csv")
    pages_dir = "pages_real"
    
    print(f"Loaded {len(df)} real-world pages")
    print()
    
    # Run two-stage predictor
    df, stage2_invocations = evaluate_two_stage(df, pages_dir)
    
    # Compute metrics
    metrics = compute_metrics(df)
    
    print("\n" + "="*70)
    print("RESULTS ON REAL-WORLD DATA")
    print("="*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print()
    
    print("Confusion Matrix:")
    print(f"  TP: {metrics['tp']:5d}  |  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  |  TN: {metrics['tn']:5d}")
    print()
    
    # Stage usage
    print("="*70)
    print("STAGE USAGE")
    print("="*70)
    stage_counts = df['stage_used'].value_counts()
    total = len(df)
    for stage, count in stage_counts.items():
        print(f"  {stage:10s}: {count:5d} / {total} ({count/total*100:.2f}%)")
    print()
    
    # Per-source-type accuracy
    print("="*70)
    print("ACCURACY BY SOURCE TYPE")
    print("="*70)
    for source_type in sorted(df['source_type'].unique()):
        subset = df[df['source_type'] == source_type]
        acc = (subset["two_stage_pred"] == subset["label"]).mean()
        stage2_pct = (subset['stage_used'] == 'stage2').sum() / len(subset) * 100
        print(f"  {source_type:10s}: {acc:.4f} (Stage-2: {stage2_pct:.1f}%)")
    print()
    
    # Comparison with synthetic data
    print("="*70)
    print("COMPARISON: SYNTHETIC vs REAL-WORLD")
    print("="*70)
    
    # Load synthetic results if available
    try:
        df_synthetic = pd.read_csv("phase3/phase3_benchmark_results.csv")
        synthetic_acc = (df_synthetic["two_stage_pred"] == df_synthetic["label"]).mean()
        print(f"Synthetic data accuracy:   {synthetic_acc:.4f}")
        print(f"Real-world data accuracy:  {metrics['accuracy']:.4f}")
        print(f"Difference:                {(metrics['accuracy'] - synthetic_acc)*100:+.2f}%")
    except FileNotFoundError:
        print("Synthetic results not found. Run Phase 3 benchmark first.")
    
    print()
    
    # Save results
    df.to_csv("phase2b/phase2b_real_world_results.csv", index=False)
    print("✅ Saved detailed results to 'phase2b/phase2b_real_world_results.csv'")

if __name__ == "__main__":
    main()
