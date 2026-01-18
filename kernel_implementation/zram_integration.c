// SPDX-License-Identifier: GPL-2.0
/*
 * zram_integration.c - Integration example for zram
 *
 * Shows how to integrate the predictor into zram compression path.
 * This is example code showing the integration points.
 *
 * Copyright (C) 2026
 */

/*
 * ============================================================================
 * INTEGRATION POINT 1: zram compression path
 * ============================================================================
 *
 * File: drivers/block/zram/zram_drv.c
 * Function: zram_write_page() or zram_bvec_write()
 *
 * Add predictor check before compression:
 */

#if 0  /* Example integration code */

#include "zram_predictor.h"

static int zram_write_page(struct zram *zram, struct page *page,
			   u32 index)
{
	void *src;
	int ret;

	src = kmap_atomic(page);

	/* NEW: Check if page is incompressible */
	if (zram_page_incompressible(src)) {
		kunmap_atomic(src);
		
		/* Update statistics */
		atomic64_inc(&zram->stats.incompressible_pages);
		atomic64_inc(&zram->stats.pages_stored);
		
		/*
		 * Option 1: Store uncompressed (if supported)
		 * Option 2: Skip storage, return error
		 * Option 3: Compress anyway (for validation)
		 */
		
		/* For now, skip compression */
		return -E2BIG;  /* Signal incompressible */
	}

	/* Normal compression path */
	ret = zram_compress_page(zram, src, index);
	kunmap_atomic(src);

	return ret;
}

#endif

/*
 * ============================================================================
 * INTEGRATION POINT 2: Statistics tracking
 * ============================================================================
 *
 * File: drivers/block/zram/zram_drv.h
 *
 * Add to struct zram_stats:
 */

#if 0  /* Example statistics */

struct zram_stats {
	/* Existing fields... */
	atomic64_t compr_data_size;
	atomic64_t pages_stored;
	
	/* NEW: Predictor statistics */
	atomic64_t incompressible_pages;    /* Predicted incompressible */
	atomic64_t predictor_true_positive; /* Correct predictions */
	atomic64_t predictor_false_positive;/* Incorrect predictions */
	atomic64_t text_pages_detected;     /* Text pages caught */
};

#endif

/*
 * ============================================================================
 * INTEGRATION POINT 3: sysfs configuration
 * ============================================================================
 *
 * File: drivers/block/zram/zram_drv.c
 *
 * Add sysfs attributes for runtime configuration:
 */

#if 0  /* Example sysfs attributes */

/* Enable/disable predictor */
static ssize_t predictor_enabled_show(struct device *dev,
				      struct device_attribute *attr,
				      char *buf)
{
	struct zram *zram = dev_to_zram(dev);
	return scnprintf(buf, PAGE_SIZE, "%d\n", zram->predictor_enabled);
}

static ssize_t predictor_enabled_store(struct device *dev,
				       struct device_attribute *attr,
				       const char *buf, size_t len)
{
	struct zram *zram = dev_to_zram(dev);
	int val;

	if (kstrtoint(buf, 10, &val))
		return -EINVAL;

	zram->predictor_enabled = !!val;
	return len;
}

static DEVICE_ATTR_RW(predictor_enabled);

/* Show predictor statistics */
static ssize_t predictor_stats_show(struct device *dev,
				    struct device_attribute *attr,
				    char *buf)
{
	struct zram *zram = dev_to_zram(dev);
	ssize_t ret;

	ret = scnprintf(buf, PAGE_SIZE,
			"incompressible_pages: %llu\n"
			"text_pages_detected: %llu\n"
			"true_positives: %llu\n"
			"false_positives: %llu\n",
			(u64)atomic64_read(&zram->stats.incompressible_pages),
			(u64)atomic64_read(&zram->stats.text_pages_detected),
			(u64)atomic64_read(&zram->stats.predictor_true_positive),
			(u64)atomic64_read(&zram->stats.predictor_false_positive));

	return ret;
}

static DEVICE_ATTR_RO(predictor_stats);

#endif

/*
 * ============================================================================
 * INTEGRATION POINT 4: Validation mode (optional)
 * ============================================================================
 *
 * For testing, compress anyway and compare prediction with actual result:
 */

#if 0  /* Example validation code */

static int zram_write_page_with_validation(struct zram *zram,
					   struct page *page, u32 index)
{
	void *src;
	bool predicted_incompressible;
	int ret;
	size_t comp_len;

	src = kmap_atomic(page);

	/* Predict */
	predicted_incompressible = zram_page_incompressible(src);

	/* Compress anyway (for validation) */
	ret = zram_compress_page(zram, src, index);
	comp_len = zram_get_obj_size(zram, index);

	/* Check prediction accuracy */
	bool actually_incompressible = (comp_len >= PAGE_SIZE * 0.7);

	if (predicted_incompressible == actually_incompressible)
		atomic64_inc(&zram->stats.predictor_true_positive);
	else
		atomic64_inc(&zram->stats.predictor_false_positive);

	kunmap_atomic(src);
	return ret;
}

#endif

/*
 * ============================================================================
 * INTEGRATION POINT 5: Kconfig option
 * ============================================================================
 *
 * File: drivers/block/zram/Kconfig
 *
 * Add configuration option:
 */

#if 0  /* Example Kconfig */

config ZRAM_PREDICTOR
	bool "Enable compressibility prediction"
	depends on ZRAM
	default y
	help
	  Enable fast prediction of page compressibility before compression.
	  This can reduce CPU usage by skipping compression of incompressible
	  pages (random data, encrypted data, already compressed data).

	  The predictor uses lightweight sampling and feature extraction to
	  identify incompressible pages with ~97% accuracy.

	  If unsure, say Y.

#endif

/*
 * ============================================================================
 * USAGE EXAMPLE
 * ============================================================================
 *
 * After integration, the predictor runs automatically:
 *
 * 1. Enable zram:
 *    # echo 1G > /sys/block/zram0/disksize
 *    # mkswap /dev/zram0
 *    # swapon /dev/zram0
 *
 * 2. Check predictor is enabled:
 *    # cat /sys/block/zram0/predictor_enabled
 *    1
 *
 * 3. Monitor statistics:
 *    # cat /sys/block/zram0/predictor_stats
 *    incompressible_pages: 12345
 *    text_pages_detected: 5678
 *    true_positives: 11000
 *    false_positives: 1345
 *
 * 4. Disable predictor (if needed):
 *    # echo 0 > /sys/block/zram0/predictor_enabled
 */
