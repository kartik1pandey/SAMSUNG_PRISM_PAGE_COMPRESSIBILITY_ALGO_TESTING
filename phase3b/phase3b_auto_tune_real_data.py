"""
Phase 3B: Auto-Tune Stage-1 Thresholds & Stage-2 β for Real-World Data
Grid search to find optimal parameters
"""
import os
import sys
import numpy as np
import pandas as pd
import lz4.frame
from collections import Counter
from itertools import product
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PAGE_SIZE = 4096
SAMPLE_SIZE_STAGE1 = 64
SAMPLE_SIZE_STAGE2 = 256

# Feature weights (keep same)
FEATURE_WEIGHTS = {
    "distinct_score": -1.0,
    "max_freq_score": 2.0,
    "zero_score": 1.5,
    "run_score": 1.0
}

# Search space for real-world data
THRESHOLD_GRID = {
    "confidence_high": [1.5, 2.0, 2.5, 3.0],
    "confidence_low": [0.0, 0.5, 1.0, 1.5],
    "beta_stage2": [0.60, 0.65, 0.70, 0.75, 0.80]
}

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

def stage1_predict_with_thresholds(page_bytes, conf_high, conf_low):
    sample = sample_page_stage1(page_bytes)
    feats = extract_features(sample)
    score = compute_compressibility_score(feats)
    
    if score > conf_high:
        return "compressible", "stage1"
    elif score < conf_low:
        return "incompressible", "stage1"
    else:
        return "ambiguous", "stage2"

def stage2_predict_with_beta(page_bytes, beta):
    ratios = []
    for _ in range(2):
        offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE_STAGE2)
        sample = page_bytes[offset:offset + SAMPLE_SIZE_STAGE2]
        compressed = lz4.frame.compress(sample, compression_level=16)
        ratios.append(len(compressed) / SAMPLE_SIZE_STAGE2)
    
    avg_ratio = np.mean(ratios)
    return "compressible" if avg_ratio <= beta else "incompressible"

def evaluate_thresholds(df, pages_dir, conf_high, conf_low, beta):
    """Evaluate specific threshold configuration"""
    predictions = []
    stages = []
    
    for idx, row in df.iterrows():
        page_path = os.path.join(pages_dir, row["page_id"])
        with open(page_path, "rb") as f:
            page_bytes = f.read()
        
        stage1_result, stage = stage1_predict_with_thresholds(page_bytes, conf_high, conf_low)
        
        if stage1_result == "ambiguous":
            pred = stage2_predict_with_beta(page_bytes, beta)
        else:
            pred = stage1_result
        
        predictions.append(pred)
        stages.append(stage)
    
    df_temp = df.copy()
    df_temp["pred"] = predictions
    df_temp["stage"] = stages
    
    # Metrics
    accuracy = (df_temp["pred"] == df_temp["label"]).mean()
    
    tp = ((df_temp["label"] == "compressible") & (df_temp["pred"] == "compressible")).sum()
    tn = ((df_temp["label"] == "incompressible") & (df_temp["pred"] == "incompressible")).sum()
    fp = ((df_temp["label"] == "incompressible") & (df_temp["pred"] == "compressible")).sum()
    fn = ((df_temp["label"] == "compressible") & (df_temp["pred"] == "incompressible")).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    stage2_rate = (df_temp["stage"] == "stage2").sum() / len(df_temp)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "stage2_rate": stage2_rate
    }

def grid_search():
    """Grid search over threshold space"""
    
    print("="*70)
    print("PHASE 3B: AUTO-TUNE THRESHOLDS FOR REAL-WORLD DATA")
    print("="*70)
    print()
    
    # Load real-world dataset
    df = pd.read_csv("page_labels_real.csv")
    pages_dir = "pages_real"
    
    print(f"Loaded {len(df)} real-world pages")
    print()
    
    # Generate all combinations
    keys = list(THRESHOLD_GRID.keys())
    values = [THRESHOLD_GRID[k] for k in keys]
    combinations = list(product(*values))
    
    print(f"Testing {len(combinations)} threshold combinations...")
    print()
    
    results = []
    
    for combo in tqdm(combinations, desc="Grid search"):
        conf_high, conf_low, beta = combo
        
        # Skip invalid combinations
        if conf_low >= conf_high:
            continue
        
        metrics = evaluate_thresholds(df, pages_dir, conf_high, conf_low, beta)
        
        results.append({
            "confidence_high": conf_high,
            "confidence_low": conf_low,
            "beta_stage2": beta,
            **metrics
        })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("accuracy", ascending=False)
    
    # Save results
    results_df.to_csv("phase3b/phase3b_threshold_tuning_real.csv", index=False)
    
    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS")
    print("="*70)
    print()
    
    top10 = results_df.head(10)
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"Rank {idx}:")
        print(f"  Accuracy: {row['accuracy']:.4f} | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | FPR: {row['fpr']:.4f}")
        print(f"  Stage-2 rate: {row['stage2_rate']*100:.1f}%")
        print(f"  conf_high={row['confidence_high']:.1f}, conf_low={row['confidence_low']:.1f}, beta={row['beta_stage2']:.2f}")
        print()
    
    # Best configuration
    best = results_df.iloc[0]
    print("="*70)
    print("BEST CONFIGURATION FOR REAL-WORLD DATA")
    print("="*70)
    print(f"Accuracy:      {best['accuracy']:.4f}")
    print(f"Precision:     {best['precision']:.4f}")
    print(f"Recall:        {best['recall']:.4f}")
    print(f"FPR:           {best['fpr']:.4f}")
    print(f"Stage-2 rate:  {best['stage2_rate']*100:.1f}%")
    print()
    print("Thresholds:")
    print(f"  confidence_high: {best['confidence_high']:.1f}")
    print(f"  confidence_low:  {best['confidence_low']:.1f}")
    print(f"  beta_stage2:     {best['beta_stage2']:.2f}")
    print()
    
    # Compare with synthetic thresholds
    print("="*70)
    print("COMPARISON: SYNTHETIC vs REAL-WORLD THRESHOLDS")
    print("="*70)
    print("Synthetic (optimized for synthetic data):")
    print("  confidence_high: 2.5")
    print("  confidence_low:  1.0")
    print("  beta_stage2:     0.70")
    print()
    print("Real-World (optimized for real-world data):")
    print(f"  confidence_high: {best['confidence_high']:.1f}")
    print(f"  confidence_low:  {best['confidence_low']:.1f}")
    print(f"  beta_stage2:     {best['beta_stage2']:.2f}")
    print()
    
    print("✅ Saved all results to 'phase3b/phase3b_threshold_tuning_real.csv'")
    print()
    print("💡 Use these optimized thresholds for real-world deployment!")

if __name__ == "__main__":
    grid_search()
