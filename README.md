# RAMD-CPPS

This repository provides a lightweight reproducibility package for the paper:

**Safe-Guarded Counterfactual Defense for Malware-Resilient Cyber-Physical Power Systems**

The package contains the core simulation code, the draft manuscript PDF, and the key CSV tables used by the paper. The physical degradation metric is reported as the physics-informed percentage of affected service (PAS).

## Contents

- `src/`: core CPPS construction, malware propagation, baseline defenses, dynamic risk-aware defense, and counterfactual MPC simulation code.
- `outputs/tables/`: benchmark node features, coupling maps, main result tables, and selected paper figure/table data.
- `paper/`: draft manuscript PDF; replace with the final checked preprint before public release.
- `logs/`: execution log for the reported main run.
- `quick_check.py`: fast consistency check for the released package.

The release intentionally omits historical intermediate files, cache files, and very large raw evolution traces. The included CSV files are enough to check the main reported table values and support the paper-level results.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Check

```bash
python quick_check.py
```

The check verifies that:

- the main strategy-level result table matches the manuscript values,
- PAS terminology is used consistently in the released code and tables,
- all released Python files pass syntax compilation.

## Main Experiment

The reported main experiment evaluates seven defense strategies over five cyber-physical coupling patterns, three attacker-information levels, six defense budgets, and eleven attack onset times. The released run uses:

- Monte Carlo repetitions per task: `10`
- same-state counterfactual rollouts per candidate: `5`
- total strategy-scenario-budget-timing tasks: `6930`

To rerun the main experiment:

```bash
python src/05_run_experiments_parallel.py
```

The full run can take several hours depending on the machine. The precomputed result tables in `outputs/tables/` preserve the manuscript values.

## PAS Proxy

The PAS proxy uses the fixed bus-impact weight vector:

```text
load demand:       0.35
capacity proxy:    0.25
power degree:      0.20
power betweenness: 0.20
```

The same weights are used across scenarios and defense strategies.

## Key Files

- `outputs/tables/experiment_strategy_summary_parallel.csv`: Table III main strategy-level results.
- `outputs/tables/experiment_results_optimal_parallel.csv`: scenario-level optimal strategy results.
- `outputs/tables/experiment_results_raw_parallel.csv`: raw main-experiment strategy results.
- `outputs/tables/figure*_used.csv`: selected data used for paper figures.
- `manuscript/main.tex`: manuscript source.

## Notes

If you upload this project to GitHub, add a license file according to how you want others to reuse the code. Common choices are MIT for permissive reuse or a more restrictive academic-use license if reuse should be limited.
