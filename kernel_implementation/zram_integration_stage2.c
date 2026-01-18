// SPDX-License-Identifier: GPL-2.0
/*
 * zram_integration_stage2.c - Complete two-stage integration example
 *
 * Shows how to integrate the full two-stage predictor into zram.
 * Includes both Stage-1 only and Stage-1+Stage-2 options.
 *
 * Copyright (C) 2026
 */

/*
 * ============================================================================
 * OPTION 1: Stage-1 Only (Maximum Throughput)
 * ============================================================================
 *
 * Use this for maximum throughput. Stage-1 is very fast (~1-2 μs).
 * Accuracy: ~80%, Recall: ~100%, FPR: ~50%
 */

#if 0  /* Example: Stage-1 only */

#include "zram_predictor_stage2.h"

static int zram_write_page_stage1(struct zram *zram, struct page *page,
				  u32 index)
{
	void *src;
	int ret;

	src = kmap_atomic(page);

	/* Stage-1 only prediction */
	if (zram->predictor_enabled && zram_page_incompressible(src)) {
		kunmap_atomic(src);
		
		/* Update statistics */
		atomic64_inc(&zram->stats.incompressible_pages);
		atomic64_inc(&zram->stats.stage1_rejections);
		
		return -E2BIG;  /* Skip compression */
	}

	/* Normal compression path */
	ret = zram_compress_page(zram, src, index);
	kunmap_atomic(src);

	return ret;
}

#endif

/*
 * ============================================================================
 * OPTION 2: Full Two-Stage (Maximum Accuracy)
 * ============================================================================
 *
 * Use this for maximum accuracy. Stage-2 runs on ambiguous pages.
 * Accuracy: ~97%, Recall: ~98%, FPR: ~8%
 */

#if 0  /* Example: Full two-stage */

#include "zram_predictor_stage2.h"

static int zram_write_page_stage2(struct zram *zram, struct page *page,
				  u32 index)
{
	void *src;
	void *comp_buf;
	unsigned int comp_buf_size;
	int ret;

	src = kmap_atomic(page);

	if (zram->predictor_enabled) {
		/*
		 * Allocate temporary buffer for Stage-2 compression
		 * Size: 512 bytes (2 samples × 256 bytes)
		 * 
		 * Note: This is stack allocation, not heap.
		 * Could also reuse zram's compression buffer.
		 */
		u8 stage2_buf[512];
		comp_buf = stage2_buf;
		comp_buf_size = sizeof(stage2_buf);

		/* Full two-stage prediction */
		if (zram_page_incompressible_stage2(zram->comp, src,
						    comp_buf, comp_buf_size)) {
			kunmap_atomic(src);
			
			/* Update statistics */
			atomic64_inc(&zram->stats.incompressible_pages);
			
			/* Track which stage decided */
			int stage = zram_predictor_get_stage(src);
			if (stage == 0)
				atomic64_inc(&zram->stats.text_rejections);
			else if (stage == 1)
				atomic64_inc(&zram->stats.stage1_rejections);
			else
				atomic64_inc(&zram->stats.stage2_rejections);
			
			return -E2BIG;  /* Skip compression */
		}
	}

	/* Normal compression path */
	ret = zram_compress_page(zram, src, index);
	kunmap_atomic(src);

	return ret;
}

#endif

/*
 * ============================================================================
 * OPTION 3: Adaptive (Recommended)
 * ============================================================================
 *
 * Use Stage-1 only under high load, Stage-2 under normal load.
 * Balances throughput and accuracy based on system state.
 */

#if 0  /* Example: Adaptive */

#include "zram_predictor_stage2.h"

static int zram_write_page_adaptive(struct zram *zram, struct page *page,
				    u32 index)
{
	void *src;
	bool use_stage2;
	int ret;

	src = kmap_atomic(page);

	if (!zram->predictor_enabled)
		goto compress;

	/*
	 * Decide whether to use Stage-2 based on system load
	 * 
	 * Options:
	 *   - CPU utilization
	 *   - Memory pressure
	 *   - I/O queue depth
	 *   - Time of day
	 */
	use_stage2 = (zram->cpu_util < 80);  /* Example heuristic */

	if (use_stage2) {
		/* Full two-stage */
		u8 stage2_buf[512];
		
		if (zram_page_incompressible_stage2(zram->comp, src,
						    stage2_buf, sizeof(stage2_buf))) {
			kunmap_atomic(src);
			atomic64_inc(&zram->stats.incompressible_pages);
			return -E2BIG;
		}
	} else {
		/* Stage-1 only */
		if (zram_page_incompressible(src)) {
			kunmap_atomic(src);
			atomic64_inc(&zram->stats.incompressible_pages);
			return -E2BIG;
		}
	}

compress:
	/* Normal compression path */
	ret = zram_compress_page(zram, src, index);
	kunmap_atomic(src);

	return ret;
}

#endif

/*
 * ============================================================================
 * STATISTICS TRACKING
 * ============================================================================
 *
 * Add to struct zram_stats in zram_drv.h:
 */

#if 0  /* Example statistics structure */

struct zram_stats {
	/* Existing fields... */
	atomic64_t compr_data_size;
	atomic64_t pages_stored;
	
	/* Predictor statistics */
	atomic64_t incompressible_pages;    /* Total predicted incompressible */
	atomic64_t text_rejections;         /* Rejected by text detection */
	atomic64_t stage1_rejections;       /* Rejected by Stage-1 */
	atomic64_t stage2_rejections;       /* Rejected by Stage-2 */
	atomic64_t compressions_skipped;    /* Total compressions skipped */
	
	/* Validation statistics (optional) */
	atomic64_t predictor_true_positive;
	atomic64_t predictor_false_positive;
	atomic64_t predictor_true_negative;
	atomic64_t predictor_false_negative;
};

#endif

/*
 * ============================================================================
 * SYSFS INTERFACE
 * ============================================================================
 *
 * Add sysfs attributes for configuration and monitoring:
 */

#if 0  /* Example sysfs attributes */

/* Show predictor mode */
static ssize_t predictor_mode_show(struct device *dev,
				   struct device_attribute *attr,
				   char *buf)
{
	struct zram *zram = dev_to_zram(dev);
	const char *mode;

	switch (zram->predictor_mode) {
	case PREDICTOR_STAGE1_ONLY:
		mode = "stage1";
		break;
	case PREDICTOR_STAGE2:
		mode = "stage2";
		break;
	case PREDICTOR_ADAPTIVE:
		mode = "adaptive";
		break;
	default:
		mode = "disabled";
	}

	return scnprintf(buf, PAGE_SIZE, "%s\n", mode);
}

/* Set predictor mode */
static ssize_t predictor_mode_store(struct device *dev,
				    struct device_attribute *attr,
				    const char *buf, size_t len)
{
	struct zram *zram = dev_to_zram(dev);

	if (sysfs_streq(buf, "disabled"))
		zram->predictor_mode = PREDICTOR_DISABLED;
	else if (sysfs_streq(buf, "stage1"))
		zram->predictor_mode = PREDICTOR_STAGE1_ONLY;
	else if (sysfs_streq(buf, "stage2"))
		zram->predictor_mode = PREDICTOR_STAGE2;
	else if (sysfs_streq(buf, "adaptive"))
		zram->predictor_mode = PREDICTOR_ADAPTIVE;
	else
		return -EINVAL;

	return len;
}

static DEVICE_ATTR_RW(predictor_mode);

/* Show detailed statistics */
static ssize_t predictor_stats_show(struct device *dev,
				    struct device_attribute *attr,
				    char *buf)
{
	struct zram *zram = dev_to_zram(dev);
	ssize_t ret;
	u64 total_pages, text_rej, stage1_rej, stage2_rej;
	u64 cpu_savings_pct;

	total_pages = atomic64_read(&zram->stats.pages_stored);
	text_rej = atomic64_read(&zram->stats.text_rejections);
	stage1_rej = atomic64_read(&zram->stats.stage1_rejections);
	stage2_rej = atomic64_read(&zram->stats.stage2_rejections);

	/* Estimate CPU savings */
	cpu_savings_pct = ((text_rej + stage1_rej + stage2_rej) * 100) /
			  (total_pages + 1);

	ret = scnprintf(buf, PAGE_SIZE,
			"total_pages: %llu\n"
			"incompressible_pages: %llu\n"
			"text_rejections: %llu (%.1f%%)\n"
			"stage1_rejections: %llu (%.1f%%)\n"
			"stage2_rejections: %llu (%.1f%%)\n"
			"compressions_skipped: %llu\n"
			"cpu_savings_estimate: %llu%%\n",
			total_pages,
			(u64)atomic64_read(&zram->stats.incompressible_pages),
			text_rej, (text_rej * 100.0) / (total_pages + 1),
			stage1_rej, (stage1_rej * 100.0) / (total_pages + 1),
			stage2_rej, (stage2_rej * 100.0) / (total_pages + 1),
			(u64)atomic64_read(&zram->stats.compressions_skipped),
			cpu_savings_pct);

	return ret;
}

static DEVICE_ATTR_RO(predictor_stats);

#endif

/*
 * ============================================================================
 * VALIDATION MODE (For Testing)
 * ============================================================================
 *
 * Compress anyway and compare prediction with actual result.
 * Use this to measure predictor accuracy in production.
 */

#if 0  /* Example validation mode */

static int zram_write_page_validate(struct zram *zram, struct page *page,
				    u32 index)
{
	void *src;
	bool predicted_incomp;
	bool actually_incomp;
	int ret;
	unsigned int comp_len;

	src = kmap_atomic(page);

	/* Predict */
	u8 stage2_buf[512];
	predicted_incomp = zram_page_incompressible_stage2(zram->comp, src,
							   stage2_buf,
							   sizeof(stage2_buf));

	/* Compress anyway */
	ret = zram_compress_page(zram, src, index);
	kunmap_atomic(src);

	if (ret)
		return ret;

	/* Get actual compression result */
	comp_len = zram_get_obj_size(zram, index);
	actually_incomp = (comp_len >= PAGE_SIZE * 70 / 100);

	/* Update validation statistics */
	if (predicted_incomp && actually_incomp)
		atomic64_inc(&zram->stats.predictor_true_positive);
	else if (predicted_incomp && !actually_incomp)
		atomic64_inc(&zram->stats.predictor_false_positive);
	else if (!predicted_incomp && actually_incomp)
		atomic64_inc(&zram->stats.predictor_false_negative);
	else
		atomic64_inc(&zram->stats.predictor_true_negative);

	return 0;
}

#endif

/*
 * ============================================================================
 * PERFORMANCE MONITORING
 * ============================================================================
 *
 * Commands to monitor predictor performance:
 */

/*
 * 1. Check predictor mode:
 *    # cat /sys/block/zram0/predictor_mode
 *    stage2
 *
 * 2. View statistics:
 *    # cat /sys/block/zram0/predictor_stats
 *    total_pages: 100000
 *    incompressible_pages: 15000
 *    text_rejections: 5000 (5.0%)
 *    stage1_rejections: 7000 (7.0%)
 *    stage2_rejections: 3000 (3.0%)
 *    compressions_skipped: 15000
 *    cpu_savings_estimate: 35%
 *
 * 3. Measure CPU usage:
 *    # perf stat -e cycles,instructions <workload>
 *
 * 4. Measure latency:
 *    # fio --name=test --ioengine=libaio --iodepth=1 \
 *          --rw=randrw --bs=4k --filename=/dev/zram0
 *
 * 5. Check compression ratio:
 *    # cat /sys/block/zram0/mm_stat
 *    orig_data_size compr_data_size mem_used_total ...
 */

/*
 * ============================================================================
 * EXPECTED RESULTS
 * ============================================================================
 *
 * With Stage-1 only:
 *   - CPU reduction: 20-30%
 *   - Compression ratio: ~100% of baseline (perfect recall)
 *   - Latency: 10-15% improvement
 *   - Stage-1 rejections: 15-25%
 *
 * With full two-stage:
 *   - CPU reduction: 30-50%
 *   - Compression ratio: ~98% of baseline (98% recall)
 *   - Latency: 15-25% improvement
 *   - Stage-1 rejections: 10-15%
 *   - Stage-2 rejections: 5-10%
 *   - Text rejections: 5-15% (workload dependent)
 */
