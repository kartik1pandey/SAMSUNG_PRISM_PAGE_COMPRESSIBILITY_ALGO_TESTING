"""
Phase 2: Threshold tuning - Grid search for optimal Stage-1 thresholds
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from phase2_stage1_predictor import sample_page, extract_features, compute_metrics

PAGE_SIZE = 4096

# Threshold search space
THRESHOLD_GRID = {
    "distinct_max": [80, 90, 100, 110, 120],
    "max_freq_ratio": [0.02, 0.03, 0.04, 0.05],
    "zero_ratio": [0.05, 0.10, 0.15, 0.20],
    "run_count": [3, 5, 7, 10]
}

def stage1_predict_with_thresholds(page_bytes, thresholds):
    """Stage-1 prediction with custom thresholds"""
    sample = sample_page(page_bytes)
    feats = extract_features(sample)
    
    # ANY condition suggests compressibility
    if (feats["distinct"] < thresholds["distinct_max"] or
        feats["max_freq_ratio"] > thresholds["max_freq_ratio"] or
        feats["zero_ratio"] > thresholds["zero_ratio"] or
        feats["run_count"] > thresholds["run_count"]):
        return "compressible"
    else:
        return "incompressible"

def evaluate_thresholds(df, pages_dir, thresholds):
    """Evaluate a specific threshold configuration"""
    predictions = []
    
    for idx, row in df.iterrows():
        page_path = os.path.join(pages_dir, row["page_id"])
        with open(page_path, "rb") as f:
            page_bytes = f.read()
        
        pred = stage1_predict_with_thresholds(page_bytes, thresholds)
        predictions.append(pred)
    
    df_temp = df.copy()
    df_temp["stage1_pred"] = predictions
    
    return compute_metrics(df_temp)

def grid_search():
    """Grid search over threshold space"""
    
    print("="*60)
    print("PHASE 2: THRESHOLD TUNING (GRID SEARCH)")
    print("="*60)
    print()
    
    # Load dataset
    df = pd.read_csv("page_labels.csv")
    pages_dir = "pages"
    
    print(f"Loaded {len(df)} labeled pages")
    print()
    
    # Generate all threshold combinations
    keys = list(THRESHOLD_GRID.keys())
    values = [THRESHOLD_GRID[k] for k in keys]
    combinations = list(product(*values))
    
    print(f"Testing {len(combinations)} threshold combinations...")
    print()
    
    results = []
    
    for combo in tqdm(combinations, desc="Grid search"):
        thresholds = dict(zip(keys, combo))
        metrics = evaluate_thresholds(df, pages_dir, thresholds)
        
        results.append({
            **thresholds,
            **metrics
        })
    
    # Convert to DataFrame and sort by accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("accuracy", ascending=False)
    
    # Save all results
    results_df.to_csv("phase2_threshold_tuning.csv", index=False)
    
    print("\n" + "="*60)
    print("TOP 10 THRESHOLD CONFIGURATIONS")
    print("="*60)
    print()
    
    top10 = results_df.head(10)
    for idx, row in top10.iterrows():
        print(f"Rank {idx+1}:")
        print(f"  Accuracy: {row['accuracy']:.4f} | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | FPR: {row['fpr']:.4f}")
        print(f"  distinct_max={row['distinct_max']:.0f}, max_freq_ratio={row['max_freq_ratio']:.2f}, zero_ratio={row['zero_ratio']:.2f}, run_count={row['run_count']:.0f}")
        print()
    
    # Best configuration
    best = results_df.iloc[0]
    print("="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    print(f"Accuracy:  {best['accuracy']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall:    {best['recall']:.4f}")
    print(f"FPR:       {best['fpr']:.4f}")
    print()
    print("Thresholds:")
    print(f"  distinct_max:     {best['distinct_max']:.0f}")
    print(f"  max_freq_ratio:   {best['max_freq_ratio']:.2f}")
    print(f"  zero_ratio:       {best['zero_ratio']:.2f}")
    print(f"  run_count:        {best['run_count']:.0f}")
    print()
    
    print("✅ Saved all results to 'phase2_threshold_tuning.csv'")

if __name__ == "__main__":
    grid_search()
