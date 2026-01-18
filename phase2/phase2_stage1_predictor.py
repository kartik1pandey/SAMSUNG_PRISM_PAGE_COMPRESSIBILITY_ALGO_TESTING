"""
Phase 2: Stage-1 Predictor - Fast, lightweight compressibility prediction
Uses hybrid sampling + integer-only features
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

PAGE_SIZE = 4096
SAMPLE_SIZE = 64

# Initial thresholds (will be tuned)
STAGE1_THRESHOLDS = {
    "distinct_max": 100,      # Low distinct → compressible
    "max_freq_ratio": 0.03,   # High dominance → compressible
    "zero_ratio": 0.1,        # High sparsity → compressible
    "run_count": 5            # High repetition → compressible
}

def sample_page(page_bytes):
    """
    Hybrid sampling strategy:
    - First 64 bytes (captures headers, structure)
    - One random 64-byte window (captures body diversity)
    """
    sample1 = page_bytes[:SAMPLE_SIZE]
    
    # Random 64-byte window
    offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE)
    sample2 = page_bytes[offset:offset + SAMPLE_SIZE]
    
    return sample1 + sample2  # 128 bytes total

def extract_features(sample):
    """
    Extract integer-only features from sample
    All operations are fast: no compression, no floating point
    """
    counts = Counter(sample)
    
    # Feature 1: Distinct byte count
    distinct = len(counts)
    
    # Feature 2: Max frequency ratio (dominant byte)
    max_freq_ratio = max(counts.values()) / len(sample)
    
    # Feature 3: Zero byte ratio
    zero_ratio = counts.get(0, 0) / len(sample)
    
    # Feature 4: Run-length count (consecutive identical bytes)
    run_count = sum(1 for i in range(1, len(sample)) if sample[i] == sample[i-1])
    
    return {
        "distinct": distinct,
        "max_freq_ratio": max_freq_ratio,
        "zero_ratio": zero_ratio,
        "run_count": run_count
    }

def stage1_predict(page_bytes, thresholds=STAGE1_THRESHOLDS):
    """
    Stage-1 decision logic: rule-based classification
    
    Compressible if ANY of these patterns exist:
    - Low distinct bytes (repetitive content)
    - High max frequency (dominant pattern)
    - High zero ratio (sparse)
    - High run count (repetition)
    
    Otherwise: likely incompressible (send to Stage-2 or compress to verify)
    """
    sample = sample_page(page_bytes)
    feats = extract_features(sample)
    
    # Rule: ANY condition suggests compressibility
    if (feats["distinct"] < thresholds["distinct_max"] or
        feats["max_freq_ratio"] > thresholds["max_freq_ratio"] or
        feats["zero_ratio"] > thresholds["zero_ratio"] or
        feats["run_count"] > thresholds["run_count"]):
        return "compressible", feats
    else:
        return "incompressible", feats

def evaluate_stage1(df, pages_dir, thresholds=STAGE1_THRESHOLDS):
    """Run Stage-1 predictor on entire dataset"""
    
    print(f"Running Stage-1 predictor with thresholds:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")
    print()
    
    predictions = []
    features_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        page_path = os.path.join(pages_dir, row["page_id"])
        
        with open(page_path, "rb") as f:
            page_bytes = f.read()
        
        pred, feats = stage1_predict(page_bytes, thresholds)
        predictions.append(pred)
        features_list.append(feats)
    
    df["stage1_pred"] = predictions
    
    # Add features to dataframe for analysis
    for feat_name in ["distinct", "max_freq_ratio", "zero_ratio", "run_count"]:
        df[f"feat_{feat_name}"] = [f[feat_name] for f in features_list]
    
    return df

def compute_metrics(df):
    """Compute prediction metrics"""
    
    # Accuracy
    accuracy = (df["stage1_pred"] == df["label"]).mean()
    
    # Confusion matrix
    tp = ((df["label"] == "compressible") & (df["stage1_pred"] == "compressible")).sum()
    tn = ((df["label"] == "incompressible") & (df["stage1_pred"] == "incompressible")).sum()
    fp = ((df["label"] == "incompressible") & (df["stage1_pred"] == "compressible")).sum()
    fn = ((df["label"] == "compressible") & (df["stage1_pred"] == "incompressible")).sum()
    
    # Metrics
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
    print("="*60)
    print("PHASE 2: STAGE-1 PREDICTOR EVALUATION")
    print("="*60)
    print()
    
    # Load labeled dataset
    df = pd.read_csv("page_labels.csv")
    pages_dir = "pages"
    
    print(f"Loaded {len(df)} labeled pages")
    print()
    
    # Run Stage-1 predictor
    df = evaluate_stage1(df, pages_dir, STAGE1_THRESHOLDS)
    
    # Compute metrics
    metrics = compute_metrics(df)
    
    print("\n" + "="*60)
    print("STAGE-1 PREDICTOR RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print()
    
    print("Confusion Matrix:")
    print(f"  TP: {metrics['tp']:5d}  |  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  |  TN: {metrics['tn']:5d}")
    print()
    
    # Per-type accuracy
    print("="*60)
    print("ACCURACY BY PAGE TYPE")
    print("="*60)
    for ptype in sorted(df['page_type'].unique()):
        subset = df[df['page_type'] == ptype]
        acc = (subset["stage1_pred"] == subset["label"]).mean()
        print(f"  {ptype:10s}: {acc:.4f}")
    print()
    
    # Save results
    df.to_csv("phase2_stage1_results.csv", index=False)
    print("✅ Saved detailed results to 'phase2_stage1_results.csv'")

if __name__ == "__main__":
    main()
