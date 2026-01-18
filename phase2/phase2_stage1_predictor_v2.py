"""
Phase 2: Stage-1 Predictor V2 - Scoring-based approach
Uses weighted feature scoring instead of simple thresholds
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

PAGE_SIZE = 4096
SAMPLE_SIZE = 64

# Scoring weights (higher = more compressible)
FEATURE_WEIGHTS = {
    "distinct_score": -1.0,      # Lower distinct → higher score
    "max_freq_score": 2.0,       # Higher max freq → higher score
    "zero_score": 1.5,           # Higher zero ratio → higher score
    "run_score": 1.0             # Higher run count → higher score
}

COMPRESSIBILITY_THRESHOLD = 0.0  # Score > threshold → compressible

def sample_page(page_bytes):
    """Hybrid sampling: first 64 + random 64 bytes"""
    sample1 = page_bytes[:SAMPLE_SIZE]
    offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE)
    sample2 = page_bytes[offset:offset + SAMPLE_SIZE]
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
    """
    Compute weighted compressibility score
    Higher score = more likely compressible
    """
    # Normalize features to [0, 1] range
    distinct_norm = 1.0 - (feats["distinct"] / 128.0)  # 128 bytes sampled
    max_freq_norm = feats["max_freq_ratio"]
    zero_norm = feats["zero_ratio"]
    run_norm = min(feats["run_count"] / 50.0, 1.0)  # Cap at 50
    
    score = (
        weights["distinct_score"] * distinct_norm +
        weights["max_freq_score"] * max_freq_norm +
        weights["zero_score"] * zero_norm +
        weights["run_score"] * run_norm
    )
    
    return score

def stage1_predict_v2(page_bytes, threshold=COMPRESSIBILITY_THRESHOLD, weights=FEATURE_WEIGHTS):
    """Score-based prediction"""
    sample = sample_page(page_bytes)
    feats = extract_features(sample)
    score = compute_compressibility_score(feats, weights)
    
    pred = "compressible" if score > threshold else "incompressible"
    
    return pred, feats, score

def evaluate_stage1_v2(df, pages_dir, threshold=COMPRESSIBILITY_THRESHOLD, weights=FEATURE_WEIGHTS):
    """Run Stage-1 V2 predictor"""
    
    print(f"Running Stage-1 V2 (scoring-based) predictor")
    print(f"Compressibility threshold: {threshold}")
    print()
    
    predictions = []
    features_list = []
    scores_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        page_path = os.path.join(pages_dir, row["page_id"])
        
        with open(page_path, "rb") as f:
            page_bytes = f.read()
        
        pred, feats, score = stage1_predict_v2(page_bytes, threshold, weights)
        predictions.append(pred)
        features_list.append(feats)
        scores_list.append(score)
    
    df["stage1_pred"] = predictions
    df["compressibility_score"] = scores_list
    
    # Add features
    for feat_name in ["distinct", "max_freq_ratio", "zero_ratio", "run_count"]:
        df[f"feat_{feat_name}"] = [f[feat_name] for f in features_list]
    
    return df

def compute_metrics(df):
    """Compute prediction metrics"""
    accuracy = (df["stage1_pred"] == df["label"]).mean()
    
    tp = ((df["label"] == "compressible") & (df["stage1_pred"] == "compressible")).sum()
    tn = ((df["label"] == "incompressible") & (df["stage1_pred"] == "incompressible")).sum()
    fp = ((df["label"] == "incompressible") & (df["stage1_pred"] == "compressible")).sum()
    fn = ((df["label"] == "compressible") & (df["stage1_pred"] == "incompressible")).sum()
    
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
    print("PHASE 2: STAGE-1 PREDICTOR V2 (SCORING-BASED)")
    print("="*60)
    print()
    
    df = pd.read_csv("page_labels.csv")
    pages_dir = "pages"
    
    print(f"Loaded {len(df)} labeled pages")
    print()
    
    # Run predictor
    df = evaluate_stage1_v2(df, pages_dir)
    
    # Compute metrics
    metrics = compute_metrics(df)
    
    print("\n" + "="*60)
    print("STAGE-1 V2 RESULTS")
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
    
    # Score distribution
    print("="*60)
    print("SCORE DISTRIBUTION BY LABEL")
    print("="*60)
    for label in ['compressible', 'incompressible']:
        scores = df[df['label'] == label]['compressibility_score']
        print(f"{label:15s}: mean={scores.mean():.3f}, std={scores.std():.3f}, min={scores.min():.3f}, max={scores.max():.3f}")
    print()
    
    # Save results
    df.to_csv("phase2_stage1_v2_results.csv", index=False)
    print("✅ Saved detailed results to 'phase2_stage1_v2_results.csv'")

if __name__ == "__main__":
    main()
