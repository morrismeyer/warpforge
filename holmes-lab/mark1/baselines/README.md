# Performance Baselines

This directory contains performance baselines for the nightly full build.

## Structure

```
baselines/
├── README.md              # This file
├── ucc-collective.json    # UCC collective operation baselines
├── native-image.json      # Native-image vs JVM comparison baselines
└── throughput.json        # Network throughput baselines
```

## Baseline Format

Each baseline file contains metrics with acceptable ranges:

```json
{
  "version": "1.0",
  "updated": "2026-01-23T00:00:00Z",
  "metrics": {
    "allreduce_16mb_throughput_gbps": {
      "value": 85.0,
      "threshold_percent": 10,
      "direction": "higher_is_better"
    },
    "allreduce_16mb_latency_us": {
      "value": 1500,
      "threshold_percent": 15,
      "direction": "lower_is_better"
    }
  }
}
```

## Updating Baselines

Baselines should be updated when:
1. Hardware changes (new NICs, different machines)
2. Intentional performance improvements are merged
3. Algorithm changes that affect expected performance

To update baselines after a successful nightly run:
```bash
# Review the performance results
cat ~/surfworks/warpforge/holmes-lab/mark1/results/nightly-YYYYMMDD/*.log

# Update the baseline file with new values
# (manual process to ensure intentional updates)
```

## Regression Detection

The nightly build will fail if:
- Any metric regresses beyond its threshold percentage
- Required metrics are missing from the results

This ensures performance regressions are caught before they reach production.
