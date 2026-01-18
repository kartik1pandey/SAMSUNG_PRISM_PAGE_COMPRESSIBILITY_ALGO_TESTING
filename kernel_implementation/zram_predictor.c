// SPDX-License-Identifier: GPL-2.0
/*
 * zram_predictor.c - Memory page compressibility prediction
 *
 * Two-stage predictor for identifying incompressible pages.
 * Optimized for kernel use: no floating point, no heap allocation,
 * bounded execution time, cache-friendly.
 *
 * Copyright (C) 2026
 */

#include <linux/kernel.h>
#include <linux/random.h>
#include <linux/types.h>
#include "zram_predictor.h"

/* Page and sampling parameters */
#define PAGE_BYTES        4096
#define S1_SAMPLE_BYTES   64
#define TOTAL_SAMPLES     (2 * S1_SAMPLE_BYTES)  /* 128 bytes */

/* Stage-1 thresholds (tuned on real-world data) */
#define DISTINCT_MAX      80    /* High diversity → incompressible */
#define MAX_FREQ_PCT      5     /* Low dominance → incompressible */
#define ZERO_PCT          20    /* Low sparsity → incompressible */
#define RUN_COUNT_MIN     5     /* Low repetition → incompressible */

/* Text detection threshold */
#define ASCII_RATIO_PCT   70    /* >70% ASCII → text → compressible */

/* Scoring weights (fixed-point, scaled by 100) */
#define WEIGHT_DISTINCT   -100  /* Lower distinct → higher score */
#define WEIGHT_MAX_FREQ   200   /* Higher max freq → higher score */
#define WEIGHT_ZERO       150   /* Higher zero ratio → higher score */
#define WEIGHT_RUN        100   /* Higher run count → higher score */

/* Confidence thresholds (scaled by 100) */
#define CONFIDENCE_HIGH   250   /* Score > 2.5 → compressible */
#define CONFIDENCE_LOW    100   /* Score < 1.0 → incompressible */

/**
 * zram_page_is_text() - Fast text page detection
 *
 * Text pages (source code, CSV, logs, markdown) are usually compressible
 * due to word/phrase repetition, even though they have high byte diversity.
 * This fast check catches them before Stage-1 feature extraction.
 */
bool zram_page_is_text(void *page)
{
	u8 *p = (u8 *)page;
	u32 ascii_count = 0;
	int i;

	/* Count printable ASCII (32-126) + whitespace (9, 10, 13) */
	for (i = 0; i < PAGE_BYTES; i++) {
		u8 b = p[i];
		if ((b >= 32 && b <= 126) || b == 9 || b == 10 || b == 13)
			ascii_count++;
	}

	/* Text if >70% ASCII */
	return (ascii_count * 100 / PAGE_BYTES) > ASCII_RATIO_PCT;
}

/**
 * stage1_extract_features() - Extract features from sampled bytes
 *
 * Computes:
 *   - distinct: Number of unique byte values
 *   - max_freq: Frequency of most common byte
 *   - zeros: Count of zero bytes
 *   - runs: Count of consecutive identical bytes
 */
static void stage1_extract_features(u8 *sample, int sample_size,
				    u16 *distinct, u16 *max_freq,
				    u16 *zeros, u16 *runs)
{
	u16 freq[256] = {0};
	int i;

	/* Build frequency table */
	for (i = 0; i < sample_size; i++)
		freq[sample[i]]++;

	/* Compute features */
	*distinct = 0;
	*max_freq = 0;
	for (i = 0; i < 256; i++) {
		if (freq[i]) {
			(*distinct)++;
			if (freq[i] > *max_freq)
				*max_freq = freq[i];
		}
	}

	*zeros = freq[0];

	/* Count runs (consecutive identical bytes) */
	*runs = 0;
	for (i = 1; i < sample_size; i++) {
		if (sample[i] == sample[i - 1])
			(*runs)++;
	}
}

/**
 * stage1_compute_score() - Compute compressibility score
 *
 * Uses weighted feature scoring. Higher score = more compressible.
 * All arithmetic is integer-based (scaled by 100).
 *
 * Returns: Score (typically -100 to 400)
 */
static int stage1_compute_score(u16 distinct, u16 max_freq,
				 u16 zeros, u16 runs)
{
	int score = 0;
	int distinct_norm, max_freq_norm, zero_norm, run_norm;

	/* Normalize features to [0, 100] */
	distinct_norm = 100 - (distinct * 100 / TOTAL_SAMPLES);
	max_freq_norm = max_freq * 100 / TOTAL_SAMPLES;
	zero_norm = zeros * 100 / TOTAL_SAMPLES;
	run_norm = (runs > 50) ? 100 : (runs * 100 / 50);

	/* Weighted sum (weights scaled by 100) */
	score += WEIGHT_DISTINCT * distinct_norm / 100;
	score += WEIGHT_MAX_FREQ * max_freq_norm / 100;
	score += WEIGHT_ZERO * zero_norm / 100;
	score += WEIGHT_RUN * run_norm / 100;

	return score;
}

/**
 * zram_page_incompressible() - Main prediction function
 *
 * Decision flow:
 *   1. Text detection (fast) → if text, return false (compressible)
 *   2. Stage-1 (fast features) → if confident, return decision
 *   3. Stage-2 (sample compression) → for ambiguous cases [OPTIONAL]
 *
 * Current implementation: Text detection + Stage-1 only
 * Stage-2 can be added if higher accuracy is needed.
 */
bool zram_page_incompressible(void *page)
{
	u8 *p = (u8 *)page;
	u8 sample[TOTAL_SAMPLES];
	u16 distinct, max_freq, zeros, runs;
	int score;
	u32 offset;
	int i;

	/* Step 1: Text detection (fastest) */
	if (zram_page_is_text(page))
		return false;  /* Text is usually compressible */

	/* Step 2: Stage-1 feature extraction */

	/* Hybrid sampling: first 64 bytes + random 64 bytes */
	for (i = 0; i < S1_SAMPLE_BYTES; i++)
		sample[i] = p[i];

	/* Random offset for second sample */
	offset = prandom_u32() % (PAGE_BYTES - S1_SAMPLE_BYTES);
	for (i = 0; i < S1_SAMPLE_BYTES; i++)
		sample[S1_SAMPLE_BYTES + i] = p[offset + i];

	/* Extract features */
	stage1_extract_features(sample, TOTAL_SAMPLES,
				&distinct, &max_freq, &zeros, &runs);

	/* Compute compressibility score */
	score = stage1_compute_score(distinct, max_freq, zeros, runs);

	/* Decision based on confidence thresholds */
	if (score > CONFIDENCE_HIGH)
		return false;  /* High confidence: compressible */
	else if (score < CONFIDENCE_LOW)
		return true;   /* High confidence: incompressible */

	/*
	 * Ambiguous case: score between thresholds
	 * 
	 * Option 1: Conservative (current) - assume compressible
	 * Option 2: Add Stage-2 sample compression here
	 * Option 3: Use additional heuristics
	 */
	return false;  /* Conservative: try compression */
}

/*
 * Optional: Stage-2 sample compression (not implemented)
 *
 * For higher accuracy, Stage-2 can compress 2 random 256-byte samples
 * and estimate compression ratio. This adds ~10-20 μs per page.
 *
 * Implementation would require:
 *   - Access to LZ4 compression function
 *   - Temporary buffer for compressed output
 *   - Compression ratio threshold check
 */
