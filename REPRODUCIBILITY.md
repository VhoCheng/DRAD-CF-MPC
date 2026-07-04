# Reproducibility Notes

This package is a lightweight public release. It keeps the core simulation code and the key tables used by the manuscript, while excluding historical intermediate outputs and large raw trace files that are not needed for a first GitHub release.

## Reported Main Table

The main strategy-level table in `outputs/tables/experiment_strategy_summary_parallel.csv` should round to:

| Strategy | EPmax | PAS | RT | RI | RE |
|---|---:|---:|---:|---:|---:|
| DRAD-CF-MPC | 0.289 | 0.306 | 6.638 | 0.827 | 0.444 |
| CDP | 0.298 | 0.315 | 6.765 | 0.822 | 0.439 |
| DRAD | 0.348 | 0.369 | 7.657 | 0.796 | 0.402 |
| CP | 0.465 | 0.499 | 9.697 | 0.716 | 0.319 |
| PDP | 0.465 | 0.503 | 9.770 | 0.714 | 0.316 |
| RP | 0.567 | 0.611 | 11.529 | 0.647 | 0.244 |
| NP | 0.786 | 0.848 | 17.968 | 0.449 | 0.152 |

These values are checked by `quick_check.py`.

## Fixed PAS Weights

The released code uses:

```text
load demand       0.35
capacity proxy    0.25
power degree      0.20
power betweenness 0.20
```

The weight vector is fixed across all strategies and scenarios. The precomputed CSV files are preserved as the manuscript result tables.

## Full Rerun

Run:

```bash
python src/05_run_experiments_parallel.py
```

The reported configuration uses 10 Monte Carlo repetitions and 5 same-state counterfactual rollouts, matching Table II of the manuscript draft. A full rerun can take several hours.
