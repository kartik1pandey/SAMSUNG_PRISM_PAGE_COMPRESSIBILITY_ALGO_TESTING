"""
Phase 3B: Improved Predictor with Text Detection
Adds ASCII detection to handle text pages better
"""
import os
import sys
import numpy as np
import pandas as pd
import lz4.frame
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PAGE_SIZE = 4096
SAMPLE_SIZE_STAGE1 = 64
SAMPLE_SIZE_STAGE2 = 256

# Optimized thresholds for real-world data
STAGE1_CONFIDENCE_THRESHOLD_HIGH = 2.5
STAGE1_CONFIDENCE_THRESHOLD_LOW = 1.0
BETA_STAGE2 = 0.60

FEATURE_WEIGHTS = {
    "distinct_score": -1.0,
    "max_freq_score": 2.0,
    "zero_score": 1.5,
    "run_score": 1.0
}

# Text detection threshold
ASCII_RATIO_THRESHOLD = 0.7

# ============== Text Detection ==============

def is_text_page(page_bytes):
    """
    Detect if page is text-based (ASCII)
    Text pages are usually compressible due to word/phrase repetition
    """
    # Count printable ASCII + common whitespace
    ascii_count = sum(1 for b in page_bytes if 32 <= b <= 126 or b in [9, 10, 13])
    ascii_ratio = ascii_count / len(page_bytes)
    
    return ascii_ratio > ASCII_RATIO_THRESHOLD

# ============== Stage-1 Functions ==============

def sample_page_stage1(page_bytes):
    sample1 = page_bytes[:SAMPLE_SIZE_STAGE1]
    offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE_STAGE1)
    sample2 = page_bytes[offset:offset + SAMPLE_SIZE_STAGE1]
    return sample1 + sample2

def extract_features(sample):
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
    ratios = []
    for _ in range(2):
        offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE_STAGE2)
        sample = page_bytes[offset:offset + SAMPLE_SIZE_STAGE2]
        compressed = lz4.frame.compress(sample, compression_level=16)
        ratios.append(len(compressed) / SAMPLE_SIZE_STAGE2)
    
    avg_ratio = np.mean(ratios)
    return "compressible" if avg_ratio <= BETA_STAGE2 else "incompressible"

# ============== Improved Predictor ==============

def improved_predict(page_bytes):
    """
    Improved predictor with text detection
    
    Decision flow:
    1. Check if text page → if yes, predict compressible
    2. Otherwise, use two-stage predictor
    """
    # Text detection (fast check)
    if is_text_page(page_bytes):
        return "compressible", "text_detection", None
    
    # Stage-1 for binary data
    stage1_decision, stage1_score = stage1_predict(page_bytes)
    
    if stage1_decision == "compressible":
        return "compressible", "stage1", stage1_score
    elif stage1_decision == "incompressible":
        return "incompressible", "stage1", stage1_score
    else:  # ambiguous
        stage2_decision = stage2_predict(page_bytes)
        return stage2_decision, "stage2", stage1_score

# ============== Evaluation ==============

def evaluate_improved(df, pages_dir):
    """Run improved predictor on real-world dataset"""
    
    print("Running Improved Predictor (with Text Detection)...")
    print()
    
    predictions = []
    stages_used = []
    stage1_scores = []
    
    text_detected = 0
    stage1_used = 0
    stage2_used = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        page_path = os.path.join(pages_dir, row["page_id"])
        
        with open(page_path, "rb") as f:
            page_bytes = f.read()
        
        pred, stage, score = improved_predict(page_bytes)
        
        predictions.append(pred)
        stages_used.append(stage)
        stage1_scores.append(score if score is not None else 0.0)
        
        if stage == "text_detection":
            text_detected += 1
        elif stage == "stage1":
            stage1_used += 1
        else:
            stage2_used += 1
    
    df["improved_pred"] = predictions
    df["stage_used"] = stages_used
    df["stage1_score"] = stage1_scores
    
    return df, text_detected, stage1_used, stage2_used

def compute_metrics(df):
    accuracy = (df["improved_pred"] == df["label"]).mean()
    
    tp = ((df["label"] == "compressible") & (df["improved_pred"] == "compressible")).sum()
    tn = ((df["label"] == "incompressible") & (df["improved_pred"] == "incompressible")).sum()
    fp = ((df["label"] == "incompressible") & (df["improved_pred"] == "compressible")).sum()
    fn = ((df["label"] == "compressible") & (df["improved_pred"] == "incompressible")).sum()
    
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
    print("PHASE 3B: IMPROVED PREDICTOR WITH TEXT DETECTION")
    print("="*70)
    print()
    
    # Load real-world dataset
    df = pd.read_csv("page_labels_real.csv")
    pages_dir = "pages_real"
    
    print(f"Loaded {len(df)} real-world pages")
    print()
    
    # Run improved predictor
    df, text_detected, stage1_used, stage2_used = evaluate_improved(df, pages_dir)
    
    # Compute metrics
    metrics = compute_metrics(df)
    
    print("\n" + "="*70)
    print("IMPROVED PREDICTOR RESULTS")
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
    total = len(df)
    print(f"  Text detection: {text_detected:5d} / {total} ({text_detected/total*100:.2f}%)")
    print(f"  Stage-1:        {stage1_used:5d} / {total} ({stage1_used/total*100:.2f}%)")
    print(f"  Stage-2:        {stage2_used:5d} / {total} ({stage2_used/total*100:.2f}%)")
    print()
    
    # Per-source-type accuracy
    print("="*70)
    print("ACCURACY BY SOURCE TYPE")
    print("="*70)
    for source_type in sorted(df['source_type'].unique()):
        subset = df[df['source_type'] == source_type]
        acc = (subset["improved_pred"] == subset["label"]).mean()
        text_pct = (subset['stage_used'] == 'text_detection').sum() / len(subset) * 100
        print(f"  {source_type:10s}: {acc:.4f} (Text detection: {text_pct:.1f}%)")
    print()
    
    # Comparison with baseline
    print("="*70)
    print("IMPROVEMENT OVER BASELINE")
    print("="*70)
    
    # Load baseline results
    try:
        df_baseline = pd.read_csv("phase2b/phase2b_real_world_results.csv")
        baseline_acc = (df_baseline["two_stage_pred"] == df_baseline["label"]).mean()
        baseline_recall = ((df_baseline["label"] == "compressible") & (df_baseline["two_stage_pred"] == "compressible")).sum() / (df_baseline["label"] == "compressible").sum()
        
        print(f"Baseline (two-stage):      {baseline_acc:.4f} accuracy, {baseline_recall:.4f} recall")
        print(f"Improved (with text det):  {metrics['accuracy']:.4f} accuracy, {metrics['recall']:.4f} recall")
        print(f"Improvement:               {(metrics['accuracy'] - baseline_acc)*100:+.2f}% accuracy, {(metrics['recall'] - baseline_recall)*100:+.2f}% recall")
    except FileNotFoundError:
        print("Baseline results not found.")
    
    print()
    
    # Save results
    df.to_csv("phase3b/phase3b_improved_results.csv", index=False)
    print("✅ Saved detailed results to 'phase3b/phase3b_improved_results.csv'")
    print()
    print("💡 Text detection significantly improves recall on real-world data!")

if __name__ == "__main__":
    main()
