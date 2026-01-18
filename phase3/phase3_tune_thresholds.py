"""
Phase 3: Tune Stage-1 confidence thresholds and Stage-2 beta
Find optimal balance between Stage-1 rejection rate and overall accuracy
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

# Import Stage-1 functions
from phase3.phase3_two_stage_predictor import (
    sample_page_stage1, extract_features, compute_compressibility_score
)

# Threshold search space
THRESHOLD_GRID = {
    "confidence_high": [2.0, 2.5, 3.0],
    "confidence_low": [0.0, 0.5, 1.0],
    "beta_stage2": [0.70, 0.75, 0.80]
}

def stage2_predict_simple(page_bytes, beta, num_samples=2):
    """Simple Stage-2 prediction"""
    ratios = []
    for _ in range(num_samples):
        offset = np.random.randint(0, PAGE_SIZE - SAMPLE_SIZE_STAGE2)
        sample = page_bytes[offset:offset + SAMPLE_SIZE_STAGE2]
        compressed = lz4.frame.compress(sample, compression_level=16)
        ratios.append(len(compressed) / SAMPLE_SIZE_STAGE2)
    
    avg_ratio = np.mean(ratios)
    return "compressible" if avg_ratio <= beta else "incompressible"

def two_stage_predict_with_thresholds(page_bytes, conf_high, conf_low, beta):
    """Two-stage prediction with custom thresholds"""
    sample = sample_page_stage1(page_bytes)
    feats = extract_features(sample)
    score = compute_compressibility_score(feats)
    
    if score > conf_high:
        return "compressible", "stage1"
    elif score < conf_low:
        return "incompressible", "stage1"
    else:
        pred = stage2_predict_simple(page_bytes, beta)
        return pred, "stage2"

def evaluate_thresholds(df, pages_dir, conf_high, conf_low, beta):
    """Evaluate specific threshold configuration"""
    predictions = []
    stages = []
    
    for idx, row in df.iterrows():
        page_path = os.path.join(pages_dir, row["page_id"])
        with open(page_path, "rb") as f:
            page_bytes = f.read()
        
        pred, stage = two_stage_predict_with_thresholds(page_bytes, conf_high, conf_low, beta)
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
    
    print("="*60)
    print("PHASE 3: THRESHOLD TUNING (GRID SEARCH)")
    print("="*60)
    print()
    
    # Load dataset
    df = pd.read_csv("page_labels.csv")
    pages_dir = "pages"
    
    print(f"Loaded {len(df)} labeled pages")
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
    results_df.to_csv("phase3/phase3_threshold_tuning.csv", index=False)
    
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS")
    print("="*60)
    print()
    
    top10 = results_df.head(10)
    for idx, row in top10.iterrows():
        print(f"Rank {idx+1}:")
        print(f"  Accuracy: {row['accuracy']:.4f} | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | FPR: {row['fpr']:.4f}")
        print(f"  Stage-2 rate: {row['stage2_rate']*100:.1f}%")
        print(f"  conf_high={row['confidence_high']:.1f}, conf_low={row['confidence_low']:.1f}, beta={row['beta_stage2']:.2f}")
        print()
    
    # Best configuration
    best = results_df.iloc[0]
    print("="*60)
    print("BEST CONFIGURATION")
    print("="*60)
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
    
    print("✅ Saved all results to 'phase3/phase3_threshold_tuning.csv'")

if __name__ == "__main__":
    grid_search()
