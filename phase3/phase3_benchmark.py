"""
Phase 3: Two-Stage Predictor Benchmark
Measures actual CPU time and performance metrics
"""
import time
import pandas as pd
import os
import numpy as np
import lz4.frame
from collections import Counter
from tqdm import tqdm

# ---------- Parameters ----------
PAGE_SIZE = 4096
SAMPLE_SIZE_STAGE1 = 64
SAMPLE_SIZE_STAGE2 = 256

# Optimized thresholds from tuning
STAGE1_CONFIDENCE_THRESHOLD_HIGH = 2.5
STAGE1_CONFIDENCE_THRESHOLD_LOW = 1.0
BETA_STAGE2 = 0.70

# Feature weights
FEATURE_WEIGHTS = {
    "distinct_score": -1.0,
    "max_freq_score": 2.0,
    "zero_score": 1.5,
    "run_score": 1.0
}

pages_dir = "pages"
df = pd.read_csv("page_labels.csv")

# ---------- Stage-1 Functions ----------
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
        return "compressible"
    elif score < STAGE1_CONFIDENCE_THRESHOLD_LOW:
        return "incompressible"
    else:
        return "ambiguous"

# ---------- Stage-2 Function ----------
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

# ---------- Full Page Compression (Baseline) ----------
def full_page_compress(page_bytes):
    """Baseline: compress entire page"""
    compressed = lz4.frame.compress(page_bytes, compression_level=16)
    ratio = len(compressed) / PAGE_SIZE
    return "compressible" if ratio <= 0.7 else "incompressible"

# ---------- Benchmark ----------
print("="*70)
print("PHASE 3: TWO-STAGE PREDICTOR BENCHMARK")
print("="*70)
print()
print(f"Configuration:")
print(f"  Stage-1 confidence thresholds: [{STAGE1_CONFIDENCE_THRESHOLD_LOW}, {STAGE1_CONFIDENCE_THRESHOLD_HIGH}]")
print(f"  Stage-2 beta threshold: {BETA_STAGE2}")
print(f"  Dataset size: {len(df)} pages")
print()

# Timing arrays
stage1_times = []
stage2_times = []
full_compress_times = []

# Counters
stage2_invocations = 0
stage1_compressible = 0
stage1_incompressible = 0

# Results
predictions = []

print("Running benchmark...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
    page_path = os.path.join(pages_dir, row["page_id"])
    with open(page_path, "rb") as f:
        page_bytes = f.read()
    
    # Stage-1 timing
    t0 = time.perf_counter()
    stage1_result = stage1_predict(page_bytes)
    t1 = time.perf_counter()
    stage1_times.append(t1 - t0)
    
    # Stage-2 if needed
    if stage1_result == "ambiguous":
        stage2_invocations += 1
        t2 = time.perf_counter()
        final_pred = stage2_predict(page_bytes)
        t3 = time.perf_counter()
        stage2_times.append(t3 - t2)
    elif stage1_result == "incompressible":
        stage1_incompressible += 1
        final_pred = "incompressible"
    else:  # compressible
        stage1_compressible += 1
        final_pred = "compressible"
    
    predictions.append(final_pred)
    
    # Baseline: full page compression timing (sample 100 pages)
    if idx < 100:
        t4 = time.perf_counter()
        _ = full_page_compress(page_bytes)
        t5 = time.perf_counter()
        full_compress_times.append(t5 - t4)

df["two_stage_pred"] = predictions

# ---------- Compute Metrics ----------
accuracy = (df["two_stage_pred"] == df["label"]).mean()

tp = ((df["label"] == "compressible") & (df["two_stage_pred"] == "compressible")).sum()
tn = ((df["label"] == "incompressible") & (df["two_stage_pred"] == "incompressible")).sum()
fp = ((df["label"] == "incompressible") & (df["two_stage_pred"] == "compressible")).sum()
fn = ((df["label"] == "compressible") & (df["two_stage_pred"] == "incompressible")).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# Timing statistics
stage1_avg_time = np.mean(stage1_times) * 1e6  # microseconds
stage1_std_time = np.std(stage1_times) * 1e6
stage1_min_time = np.min(stage1_times) * 1e6
stage1_max_time = np.max(stage1_times) * 1e6

stage2_avg_time = np.mean(stage2_times) * 1e6 if stage2_times else 0
stage2_std_time = np.std(stage2_times) * 1e6 if stage2_times else 0
stage2_min_time = np.min(stage2_times) * 1e6 if stage2_times else 0
stage2_max_time = np.max(stage2_times) * 1e6 if stage2_times else 0

full_avg_time = np.mean(full_compress_times) * 1e6
full_std_time = np.std(full_compress_times) * 1e6

stage2_invocation_rate = stage2_invocations / len(df)
stage1_decision_rate = (stage1_compressible + stage1_incompressible) / len(df)

# Weighted average time per page
avg_time_per_page = (
    stage1_avg_time +  # All pages go through Stage-1
    stage2_avg_time * stage2_invocation_rate  # Only some go through Stage-2
)

# ---------- Results ----------
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)
print()

print("ACCURACY METRICS:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  FPR:       {fpr*100:.2f}%")
print()

print("CONFUSION MATRIX:")
print(f"              Predicted")
print(f"              Comp    Incomp")
print(f"  Actual Comp {tp:5d}   {fn:5d}")
print(f"         Incomp {fp:5d}   {tn:5d}")
print()

print("STAGE USAGE:")
print(f"  Stage-1 only (compressible):   {stage1_compressible:5d} ({stage1_compressible/len(df)*100:.1f}%)")
print(f"  Stage-1 only (incompressible): {stage1_incompressible:5d} ({stage1_incompressible/len(df)*100:.1f}%)")
print(f"  Stage-2 invoked:               {stage2_invocations:5d} ({stage2_invocation_rate*100:.1f}%)")
print(f"  Stage-1 decision rate:         {stage1_decision_rate*100:.1f}%")
print()

print("TIMING PERFORMANCE:")
print(f"  Stage-1 per page:")
print(f"    Mean:   {stage1_avg_time:7.2f} μs")
print(f"    Std:    {stage1_std_time:7.2f} μs")
print(f"    Min:    {stage1_min_time:7.2f} μs")
print(f"    Max:    {stage1_max_time:7.2f} μs")
print()

if stage2_times:
    print(f"  Stage-2 per page (when invoked):")
    print(f"    Mean:   {stage2_avg_time:7.2f} μs")
    print(f"    Std:    {stage2_std_time:7.2f} μs")
    print(f"    Min:    {stage2_min_time:7.2f} μs")
    print(f"    Max:    {stage2_max_time:7.2f} μs")
    print()

print(f"  Full page compression (baseline, n=100):")
print(f"    Mean:   {full_avg_time:7.2f} μs")
print(f"    Std:    {full_std_time:7.2f} μs")
print()

print(f"  Weighted average per page:")
print(f"    Two-stage: {avg_time_per_page:7.2f} μs")
print(f"    Full comp: {full_avg_time:7.2f} μs")
print(f"    Speedup:   {full_avg_time/avg_time_per_page:.2f}x")
print()

print("PERFORMANCE TARGETS:")
target_stage1 = 100  # μs
target_stage2_rate = 15  # %
target_accuracy = 95  # %

stage1_ok = "✅" if stage1_avg_time < target_stage1 else "⚠️"
stage2_ok = "✅" if stage2_invocation_rate*100 < target_stage2_rate else "⚠️"
accuracy_ok = "✅" if accuracy*100 >= target_accuracy else "⚠️"

print(f"  {stage1_ok} Stage-1 time < {target_stage1} μs: {stage1_avg_time:.2f} μs")
print(f"  {stage2_ok} Stage-2 rate < {target_stage2_rate}%: {stage2_invocation_rate*100:.1f}%")
print(f"  {accuracy_ok} Accuracy >= {target_accuracy}%: {accuracy*100:.2f}%")
print()

# Per-type timing analysis
print("="*70)
print("PER-TYPE ANALYSIS")
print("="*70)
print()

for ptype in sorted(df['page_type'].unique()):
    subset = df[df['page_type'] == ptype]
    type_acc = (subset["two_stage_pred"] == subset["label"]).mean()
    print(f"{ptype:10s}: {type_acc*100:5.1f}% accuracy")

print()

# Save results
df.to_csv("phase3/phase3_benchmark_results.csv", index=False)
print("✅ Saved detailed results to 'phase3/phase3_benchmark_results.csv'")

# Save timing summary
timing_summary = {
    "metric": [
        "stage1_mean_us", "stage1_std_us",
        "stage2_mean_us", "stage2_std_us",
        "full_compress_mean_us", "full_compress_std_us",
        "weighted_avg_us", "speedup_vs_full",
        "stage1_decision_rate", "stage2_invocation_rate",
        "accuracy", "precision", "recall", "fpr"
    ],
    "value": [
        stage1_avg_time, stage1_std_time,
        stage2_avg_time, stage2_std_time,
        full_avg_time, full_std_time,
        avg_time_per_page, full_avg_time/avg_time_per_page,
        stage1_decision_rate, stage2_invocation_rate,
        accuracy, precision, recall, fpr
    ]
}

timing_df = pd.DataFrame(timing_summary)
timing_df.to_csv("phase3/phase3_timing_summary.csv", index=False)
print("✅ Saved timing summary to 'phase3/phase3_timing_summary.csv'")
