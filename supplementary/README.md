# Supplementary experimental resources

This directory contains the actual experiment sources supplied with the manuscript revision, the model input tables, frozen protocol records, and selected numerical outputs. Original relative source paths are preserved beneath `supplementary/`, so this snapshot can run without replacing the repository's baseline implementation. Numbered script names are retained for source traceability.

## Set up and inspect

From the repository root:

```bash
cd supplementary
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_release.py
python prepare_inputs.py
```

`requirements-core.txt` is sufficient for the cyber simulations and most analysis. `requirements-physical.txt` additionally installs pandapower and PYPOWER for electrical studies. `environment-observed.txt` records only versions present in the supplied environment export; it is not a complete lockfile. Run the scripts from this directory, because some use the working directory to resolve inputs.

Large original CSVs are stored as lossless gzip files. `prepare_inputs.py` restores their original filenames after checking the original SHA-256 and refuses to replace different data. `SOURCE_MANIFEST.csv` maps every imported file to its source archive/member and original hash. Source code and numerical data have not been rewritten to improve outcomes. Full raw checkpoint archives and document/response files are outside this code subset.

## Study map

| Evidence or component | Original entry point | Protocol / retained outputs |
|---|---|---|
| Safe-Guard replay grid | `revision_experiments/02_safeguard_full.py` | Timestamped full-grid config and raw/summary CSVs in `revision_experiments/results/` |
| Objective and risk weights | `revision_experiments/18_objective_weight_sensitivity.py`, `19_dynamic_risk_weights.py` | Coefficients remain in each source; these drivers regenerate the profile studies |
| Topology and transmission variants | `revision_experiments/06_topology_robustness.py`, `15_transmission_mechanisms.py`, `17_transmission_statistics.py` | Distinct propagation/graph protocols; do not pool with the local-information study |
| Multiscale networks / computation | `revision_experiments/25_build_multiscale_cpps.py`, `26_multiscale_defense.py`, `16_decision_latency.py` | Retained network inputs in `results/25_multiscale_networks/` |
| Matched policy-library rolling ablation | `revision_experiments/47_final_closed_loop_validation.py` | `results/47_independent_closed_loop/`, paired cases and frozen source hashes |
| Development and observation controls | `tnse_exp49.py`, `tnse_exp50_v2.py`, `tnse_exp51.py`, `tnse_exp52.py`, `tnse_exp53_audit.py` | Actual development drivers, including the retained confirmation sweep |
| Three-stage local-information study | `tnse_exp54_final.py` | `results/54_graph_observation_delay/full_g5_s10_start17000/` |
| Electrical source case / stress / N−1 | `tnse_exp56_pglib_acopf.py`, `tnse_exp57.py` | PGLib v23.07 cases and recorded solver/coverage results |
| Assumed interrupted-demand sensitivity | `tnse_exp58.py`, `tnse_exp60.py` | `results/58_cyber_power_sensitivity/` and `60_physical_replication_audit/` |
| Physical provenance / rating coverage | `tnse_exp61.py` | `results/61_physical_provenance_audit/` |
| Complete harm and tail profiles | `analysis/recompute_safeguard.py` | Read-only analysis of the retained paired data and development/replay cohorts |

## Bounded command checks

```bash
python revision_experiments/47_final_closed_loop_validation.py --help
python tnse_exp54_final.py --help
python tnse_exp58.py --help
python tnse_exp60.py --help
```

These commands show the actual CLI. They are not full reproductions. Some imports create their designated results directory.

## Full reproduction commands

The following commands execute the genuine simulators or data analyses and may take substantial time. Use a separate work copy for reruns. Protocol/hash mismatch checks should be investigated, not bypassed; outputs from different study designs must remain separate.

```bash
# Policy-library rolling comparison: default 50 seeds for the full design.
python revision_experiments/47_final_closed_loop_validation.py --mode full --seeds 50
# Final local-information design: 15 graphs and 9600 paired evaluation contexts.
python tnse_exp54_final.py --phase full --graphs 5 --seeds 10 --seed-first 17000
# Risk summaries from the retained original data (no simulator rerun).
python analysis/recompute_safeguard.py --source-root . --output analysis/recomputed
# Physical sensitivity and audit, after its required cyber/source inputs exist.
python tnse_exp57.py --phase full
python tnse_exp58.py --phase full --graphs 5 --seeds 2
python tnse_exp60.py --source full_g5_s2
python tnse_exp61.py
```

The local-information study compares a static and an infection-aware alternative under a common CDP reference, with independent confirmation and state continuation. Its outcomes are not interchangeable with the comprehensive policy-library previews or the fixed-graph development sweep. Harm rates compare complete paired episodes; they do not isolate causal harm from an individual switch. PAS is a service proxy. The electrical sensitivity assumes a specified mapping from terminal infection to interrupted demand; it does not report measured utility outages or establish a coupled physical-safety guarantee. Solver failures and incomplete thermal-rating coverage remain in the result files.

## Third-party input attribution

The included four PGLib-OPF case files are copied unchanged from the supplied v23.07 source snapshot. Each retains its source and Creative Commons Attribution 4.0 notice; upstream project: https://github.com/power-grid-lib/pglib-opf/tree/v23.07 . The original IEEE data attribution, conversion history and source URLs remain in the file headers. No missing branch rating was filled or inferred. The repository's code license applies as specified at repository root; third-party data retains its own notices.

## Figure exports and redraw tools

`plotting/` contains the recorded CSV inputs and portable scripts for the principal new figures. Run `python run_all.py` there after installing its requirements. It regenerates four figures without simulation. Additional exported experimental views are in `revision_experiments/figures/TNSE_advanced_gallery/`, with the original generator `tnse_advanced_figures_v1.py`. The root `paper/figures/revision/README.md` maps the main and supplementary figure PDFs.
