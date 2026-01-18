// SPDX-License-Identifier: GPL-2.0
/*
 * zram_predictor_stage2.c - Stage-2 sample compression predictor
 *
 * Complete two-stage predictor implementation for kernel integration.
 * Stage-1: Fast feature-based classification
 * Stage-2: Sample compression for ambiguous cases
 *
 * Copyright (C) 2026
 */

#include <linux/kernel.h>
#include <linux/random.h>
#include <linux/types.h>
#include <linux/string.h>
#include <crypto/compress.h>
#include "zram_predictor.h"

/* Page and sampling parameters */
#define PAGE_BYTES        4096
#define S1_SAMPLE_BYTES   64
#define S2_SAMPLE_BYTES   256
#define S2_SAMPLES        2
#define TOTAL_S1_SAMPLES  (2 * S1_SAMPLE_BYTES)

/* Stage-1 thresholds */
#define DISTINCT_MAX      80
#define MAX_FREQ_PCT      5
#define ZERO_PCT          20
#define RUN_COUNT_MIN     5

/* Stage-2 threshold */
#define BETA_PCT          60    /* Sample ratio > 60% → incompressible */

/* Text detection threshold */
#define ASCII_RATIO_PCT   70

/* Scoring weights (fixed-point, scaled by 100) */
#define WEIGHT_DISTINCT   -100
#define WEIGHT_MAX_FREQ   200
#define WEIGHT_ZERO       150
#define WEIGHT_RUN        100

/* Confidence thresholds (scaled by 100) */
#define CONFIDENCE_HIGH   250   /* Score > 2.5 → compressible */
#define CONFIDENCE_LOW    100   /* Score < 1.0 → incompressible */

/*
 * ============================================================================
 * TEXT DETECTION (Fast Pre-Filter)
 * ============================================================================
 */

/**
 * zram_page_is_text() - Fast text page detection
 *
 * Text pages are usually compressible despite high byte diversity.
 */
bool zram_page_is_text(void *page)
{
	u8 *p = (u8 *)page;
	u32 ascii_count = 0;
	int i;

	/* Count printable ASCII + whitespace */
	for (i = 0; i < PAGE_BYTES; i++) {
		u8 b = p[i];
		if ((b >= 32 && b <= 126) || b == 9 || b == 10 || b == 13)
			ascii_count++;
	}

	return (ascii_count * 100 / PAGE_BYTES) > ASCII_RATIO_PCT;
}

/*
 * ============================================================================
 * STAGE-1: Fast Feature-Based Classification
 * ============================================================================
 */

/**
 * stage1_extract_features() - Extract features from sampled bytes
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

	/* Count runs */
	*runs = 0;
	for (i = 1; i < sample_size; i++) {
		if (sample[i] == sample[i - 1])
			(*runs)++;
	}
}

/**
 * stage1_compute_score() - Compute compressibility score
 *
 * Returns: Score (typically -100 to 400)
 */
static int stage1_compute_score(u16 distinct, u16 max_freq,
				 u16 zeros, u16 runs)
{
	int score = 0;
	int distinct_norm, max_freq_norm, zero_norm, run_norm;

	/* Normalize features to [0, 100] */
	distinct_norm = 100 - (distinct * 100 / TOTAL_S1_SAMPLES);
	max_freq_norm = max_freq * 100 / TOTAL_S1_SAMPLES;
	zero_norm = zeros * 100 / TOTAL_S1_SAMPLES;
	run_norm = (runs > 50) ? 100 : (runs * 100 / 50);

	/* Weighted sum */
	score += WEIGHT_DISTINCT * distinct_norm / 100;
	score += WEIGHT_MAX_FREQ * max_freq_norm / 100;
	score += WEIGHT_ZERO * zero_norm / 100;
	score += WEIGHT_RUN * run_norm / 100;

	return score;
}

/**
 * stage1_predict() - Stage-1 classification
 *
 * Returns:
 *   1 = definitely compressible (high confidence)
 *   0 = ambiguous (send to Stage-2)
 *  -1 = definitely incompressible (high confidence)
 */
static int stage1_predict(void *page)
{
	u8 *p = (u8 *)page;
	u8 sample[TOTAL_S1_SAMPLES];
	u16 distinct, max_freq, zeros, runs;
	int score;
	u32 offset;
	int i;

	/* Hybrid sampling: first 64 bytes + random 64 bytes */
	for (i = 0; i < S1_SAMPLE_BYTES; i++)
		sample[i] = p[i];

	offset = prandom_u32() % (PAGE_BYTES - S1_SAMPLE_BYTES);
	for (i = 0; i < S1_SAMPLE_BYTES; i++)
		sample[S1_SAMPLE_BYTES + i] = p[offset + i];

	/* Extract features */
	stage1_extract_features(sample, TOTAL_S1_SAMPLES,
				&distinct, &max_freq, &zeros, &runs);

	/* Compute score */
	score = stage1_compute_score(distinct, max_freq, zeros, runs);

	/* Decision based on confidence */
	if (score > CONFIDENCE_HIGH)
		return 1;   /* Compressible */
	else if (score < CONFIDENCE_LOW)
		return -1;  /* Incompressible */
	else
		return 0;   /* Ambiguous */
}

/*
 * ============================================================================
 * STAGE-2: Sample Compression
 * ============================================================================
 */

/**
 * stage2_predict() - Stage-2 sample compression
 * @comp: Crypto compressor instance (reused from zram)
 * @page: Page data
 * @comp_buf: Temporary buffer for compressed output
 * @comp_buf_size: Size of compression buffer
 *
 * Compresses 2 random 256-byte samples and estimates compression ratio.
 *
 * Returns: true if incompressible, false if compressible
 *
 * Note: Reuses zram's compressor instance - no allocation needed.
 */
static bool stage2_predict(struct crypto_comp *comp, void *page,
			   void *comp_buf, unsigned int comp_buf_size)
{
	u8 *p = (u8 *)page;
	u8 sample[S2_SAMPLE_BYTES];
	u32 offset;
	unsigned int clen;
	unsigned int total_ratio = 0;
	int i, ret;

	/* Compress multiple samples */
	for (i = 0; i < S2_SAMPLES; i++) {
		/* Random sample */
		offset = prandom_u32() % (PAGE_BYTES - S2_SAMPLE_BYTES);
		memcpy(sample, &p[offset], S2_SAMPLE_BYTES);

		/* Compress sample */
		clen = comp_buf_size;
		ret = crypto_comp_compress(comp, sample, S2_SAMPLE_BYTES,
					   comp_buf, &clen);

		if (ret) {
			/*
			 * Compression failed - fail open (assume compressible)
			 * Better to compress unnecessarily than miss opportunity
			 */
			return false;
		}

		/* Accumulate compression ratio (scaled by 100) */
		total_ratio += (clen * 100) / S2_SAMPLE_BYTES;
	}

	/* Average ratio */
	return (total_ratio / S2_SAMPLES) > BETA_PCT;
}

/*
 * ============================================================================
 * PUBLIC API: Two-Stage Predictor
 * ============================================================================
 */

/**
 * zram_page_incompressible() - Main prediction function (Stage-1 only)
 *
 * Fast version using only Stage-1. Use this for maximum throughput.
 *
 * Returns: true if incompressible, false otherwise
 */
bool zram_page_incompressible(void *page)
{
	int stage1_result;

	/* Text detection (fastest) */
	if (zram_page_is_text(page))
		return false;

	/* Stage-1 classification */
	stage1_result = stage1_predict(page);

	if (stage1_result > 0)
		return false;  /* Compressible */
	else if (stage1_result < 0)
		return true;   /* Incompressible */

	/*
	 * Ambiguous case: Stage-1 not confident
	 * 
	 * Conservative approach: assume compressible
	 * This maintains high recall at cost of some false positives
	 */
	return false;
}

/**
 * zram_page_incompressible_stage2() - Full two-stage prediction
 * @comp: Crypto compressor instance
 * @page: Page data
 * @comp_buf: Temporary buffer for Stage-2 compression
 * @comp_buf_size: Size of compression buffer
 *
 * Complete two-stage predictor. Use this for maximum accuracy.
 *
 * Returns: true if incompressible, false otherwise
 */
bool zram_page_incompressible_stage2(struct crypto_comp *comp, void *page,
				     void *comp_buf, unsigned int comp_buf_size)
{
	int stage1_result;

	/* Text detection (fastest) */
	if (zram_page_is_text(page))
		return false;

	/* Stage-1 classification */
	stage1_result = stage1_predict(page);

	if (stage1_result > 0)
		return false;  /* High confidence: compressible */
	else if (stage1_result < 0)
		return true;   /* High confidence: incompressible */

	/*
	 * Ambiguous case: Stage-1 not confident
	 * Run Stage-2 for refinement
	 */
	return stage2_predict(comp, page, comp_buf, comp_buf_size);
}

/*
 * ============================================================================
 * STATISTICS HELPERS (Optional)
 * ============================================================================
 */

/**
 * zram_predictor_get_stage() - Determine which stage would decide
 * @page: Page data
 *
 * For statistics tracking: determine which stage would make the decision.
 *
 * Returns:
 *   0 = text detection
 *   1 = Stage-1
 *   2 = Stage-2 (ambiguous)
 */
int zram_predictor_get_stage(void *page)
{
	int stage1_result;

	if (zram_page_is_text(page))
		return 0;  /* Text detection */

	stage1_result = stage1_predict(page);

	if (stage1_result != 0)
		return 1;  /* Stage-1 decided */
	else
		return 2;  /* Would go to Stage-2 */
}
