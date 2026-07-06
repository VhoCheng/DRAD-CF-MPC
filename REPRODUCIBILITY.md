# Reproducibility Notes

This repository is a lightweight public release for the DRAD-CF-MPC manuscript. It includes the core simulation code, main result tables, selected figure data, manuscript source, and a quick integrity check.

## Main Result Table

The strategy-level table in `outputs/tables/experiment_strategy_summary_parallel.csv` should round to:

| Strategy | EPmax | PAS | RT | RI | RE |
|---|---:|---:|---:|---:|---:|
| DRAD-CF-MPC | 0.289 | 0.306 | 6.638 | 0.827 | 0.444 |
| CDP | 0.298 | 0.315 | 6.765 | 0.822 | 0.439 |
| DRAD | 0.348 | 0.369 | 7.657 | 0.796 | 0.402 |
| CP | 0.465 | 0.499 | 9.697 | 0.716 | 0.319 |
| PDP | 0.465 | 0.503 | 9.770 | 0.714 | 0.316 |
| RP | 0.567 | 0.611 | 11.529 | 0.647 | 0.244 |
| NP | 0.786 | 0.848 | 17.968 | 0.449 | 0.152 |

These values are checked by:

```bash
python quick_check.py
```

## PAS Proxy Weights

The released code uses the fixed bus-impact weight vector:

```text
load demand       0.35
capacity proxy    0.25
power degree      0.20
power betweenness 0.20
```

The same vector is used across all scenarios and strategies.

## Main Experiment

Run:

```bash
python src/05_run_experiments_parallel.py
```

The reported configuration uses 50 Monte Carlo repetitions and 20 same-state counterfactual rollouts. A full rerun can take several hours.

## Scope of This Release

The release keeps the files needed for paper-level reproducibility and public inspection. Historical intermediate outputs, cache files, and large raw evolution traces are omitted to keep the GitHub repository compact.
