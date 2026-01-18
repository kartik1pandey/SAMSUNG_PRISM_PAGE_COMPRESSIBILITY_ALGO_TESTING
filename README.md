# Memory Page Compressibility Predictor

A fast, accurate predictor for memory page compressibility to optimize zram/swap compression decisions.

## Overview

This project implements a two-stage predictor that determines whether a memory page is worth compressing before actually performing compression, reducing CPU overhead in memory management systems.

## Key Features

- **High Accuracy**: 97% on real-world data, 95-98% on synthetic data
- **Fast Performance**: Stage-1 is 1.35x faster than full compression
- **Two-Stage Architecture**: Fast heuristics + sampled compression for borderline cases
- **Text Detection**: Optimized for real-world workloads with mixed data types
- **Kernel Ready**: C implementation for Linux kernel integration

## Performance

- **Accuracy**: 97% on real-world data
- **Recall**: 98% (critical for not missing compressible pages)
- **Speed**: 32-50 μs per page prediction
- **Stage-2 Usage**: <15% (most decisions made quickly)

## Project Structure

```
├── phase1/          # Synthetic data generation and labeling
├── phase1b/         # Real-world data collection
├── phase2/          # Stage-1 predictor development
├── phase2b/         # Real-world evaluation
├── phase3/          # Two-stage predictor
├── phase3b/         # Real-world optimization
├── kernel_implementation/  # C code for kernel integration
└── run_all_phases.py      # Complete pipeline execution
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run_all_phases.py

# Or run individual phases
python phase1/run_phase1.py
python phase2/run_phase2.py
python phase3/run_phase3.py
```

## Results Summary

| Metric | Stage-1 Only | Two-Stage | Improved (Text Detection) |
|--------|--------------|-----------|---------------------------|
| Accuracy | 80% | 94% | **97%** |
| Recall | 100% | 100% | **98%** |
| Speed | 32 μs | 50 μs | 40 μs |

## Applications

- **zram optimization**: Reduce CPU overhead in compressed swap
- **Memory management**: Intelligent compression decisions
- **Storage systems**: Predictive compression for better performance

## License

MIT License - see LICENSE file for details.
