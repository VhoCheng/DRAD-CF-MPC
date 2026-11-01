# Reproducibility Notes

This repository contains the baseline and supplementary experiment code, input tables, recorded results, figures, execution logs, and integrity checks for the DRAD-CF-MPC manuscript. The complete revised manuscript source is supplied separately in the Overleaf submission package.

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
python src/05_run_experiments_parallel.py --risk-profile distributed-feature --out-dir outputs/rerun_baseline
```

The reported configuration uses 50 Monte Carlo repetitions and 20 same-state counterfactual rollouts. The comprehensive baseline uses the distributed-feature DRAD profile `(0.20, 0.15, 0.15, 0.15, 0.20, 0.15)`, ordered as communication degree, communication betweenness, power degree, power capacity, infection risk, and uncertainty. The author confirmed this configuration on 2026-10-28. The record is in `configuration/baseline_profile_provenance.json`.

The main runner explicitly passes these weights to standalone DRAD and the DRAD_SCORE candidate. The rank-ensemble candidate has its own ranking weights. New results include the selected profile and the six weights, and each run writes `run_configuration.json`. New runs default to `outputs/rerun_baseline/`, leaving the supplied result tables untouched. A full rerun can take several hours.

The `degree-prioritized` option retains the alternative `(0.36, 0.18, 0.08, 0.08, 0.20, 0.10)` profile for controlled comparisons. The supplementary drivers load their own frozen source and configuration files under `supplementary/`. These source hashes and result files were preserved. The author confirmation identifies the baseline parameter values; it does not establish byte identity with a historical executable. This release checks parameter transmission and runs small smoke tasks, but does not repeat the full baseline or supplementary Monte Carlo campaigns.

```bash
python verify_baseline_profile.py
python supplementary/verify_release.py
```

## Figure Scope

The figure index under `paper/figures/revision/` identifies the revised manuscript and supplementary figures. The four author-supplied PDF replacements and the information-flow PDF are included. Plotting code and its recorded input data are under `supplementary/plotting/`.

## Final figure and table exports

`paper/figures/revision/README.md` maps every main and supplementary figure to its PDF master. `paper/tables/revision/README.md` maps final table numbers to exact LaTeX table fragments and their reading copies. These exports use the final revised manuscript text, rather than intermediate response-letter table captures.

To check this repository snapshot without rerunning simulations:

```bash
python verify_repository_release.py
python quick_check.py
python supplementary/verify_release.py
```

The 2026-11-01 update assembles final visual resources and navigation on top of the validated experiment package. It does not generate new simulation outcomes or alter recorded numerical data. The separate baseline-profile check runs small configuration-transmission tasks and is not a full experimental reproduction.
