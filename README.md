<div align="center">

# DRAD-CF-MPC

**Safe-Guarded Counterfactual Defense for Malware-Resilient Cyber-Physical Power Systems**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#-installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-quick_check-brightgreen)](#-quick-check)
[![Metric](https://img.shields.io/badge/Metric-PAS-orange)](#-main-result-table)

</div>

**DRAD-CF-MPC** is a reproducibility package for a dynamic risk-aware counterfactual model predictive control framework for malware-resilient cyber-physical power systems (CPPSs). It provides the baseline and supplementary experiment code, input tables, recorded results, and final figure and table resources for inspecting and reproducing the reported studies.

<p align="center">
  <img src="assets/framework.png" width="860" alt="Overall DRAD-CF-MPC framework">
</p>

## ✨ Highlights

- **State-to-consequence modeling**: malware diffusion in the cyber layer is linked to affected physical service and restoration burden.
- **Physics-aware affected service metric**: cyber infection is evaluated through a CPPS service-impact proxy rather than as a purely cyber count.
- **Interpretable candidate defenses**: fixed baselines, dynamic risk-aware defense, and hybrid policies are evaluated under the same observed state.
- **Counterfactual MPC selection**: candidate policies are compared with common-random-number rollouts from the same state.
- **Safe-Guarded switching**: adaptive switching is accepted only when the predicted improvement over the baseline passes a conservative safeguard test.
- **Compact reproducibility package**: the released tables preserve the manuscript results and `quick_check.py` verifies the main table, PAS terminology, and Python syntax.

## 🗂 Repository Structure

```text
DRAD-CF-MPC
├── assets/                 # README figures converted from paper-used figures only
├── logs/                   # main-run execution log
├── outputs/tables/         # released CSV tables used by the manuscript
├── paper/figures/          # final figure PDFs, including revision/ exports
├── paper/tables/revision/  # final main and supplementary table exports
├── supplementary/         # supplementary simulations, paired results and plotting tools
├── src/                    # simulation and defense code
├── CITATION.cff
├── LICENSE
├── README.md
├── REPRODUCIBILITY.md
├── quick_check.py
└── requirements.txt
```

## 🧠 Method Overview

The malware process follows a protected SEID-style cyber-state model. Protected nodes are unavailable for infection and transmission in the current defense horizon.

<p align="center">
  <img src="assets/malware_protection.png" width="760" alt="Malware propagation and protection model">
</p>

The proposed controller evaluates candidate defense policies from the same observed cyber-physical state, scores their counterfactual rollouts by degradation and recovery indicators, and implements adaptive switching only after the Safe-Guard test.

## 🔨 Installation

```bash
git clone https://github.com/VhoCheng/DRAD-CF-MPC.git
cd DRAD-CF-MPC

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Quick Check

Run:

```bash
python quick_check.py
```

The check verifies that:

- the main strategy-level result table matches the manuscript values,
- PAS terminology is used consistently in the released code and tables,
- all released Python files pass syntax compilation.

## 🔁 Reproducing the Main Experiment

The main experiment evaluates seven defense strategies over five cyber-physical coupling patterns, three attacker-information levels, six defense budgets, and eleven attack onset times.

```bash
python src/05_run_experiments_parallel.py --risk-profile distributed-feature --out-dir outputs/rerun_baseline
```

The reported configuration uses:

| Setting | Value |
|---|---:|
| Monte Carlo repetitions per task | 50 |
| Same-state counterfactual rollouts per candidate | 20 |
| Strategy-scenario-budget-timing tasks | 6930 |

The main runner explicitly uses the distributed-feature DRAD weights `(0.20, 0.15, 0.15, 0.15, 0.20, 0.15)`, in the order communication degree, communication betweenness, power degree, power capacity, infection risk, and uncertainty. The profile is applied to standalone DRAD and the DRAD-score candidate. The rank-ensemble candidate retains its separately defined ranking weights.

The full run can take several hours depending on the machine. The released CSV files in `outputs/tables/` preserve the manuscript result values. New runs go to a separate output directory and write the profile and weights in `run_configuration.json` and the result rows. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the configuration record and validation scope.

For a small execution check or a comparison using the degree-prioritized profile:

```bash
python src/05_run_experiments_parallel.py --smoke --out-dir outputs/smoke_baseline
python src/05_run_experiments_parallel.py --risk-profile degree-prioritized --out-dir outputs/rerun_degree_profile
```

The smoke command runs two tasks with one repetition and one preview. It is a wiring check, not a reproduction of the reported table. Supplementary drivers use the frozen core under `supplementary/src/` and are independent of the main runner's profile selection.

## 📊 Main Result Table

The main strategy-level comparison is stored in `outputs/tables/experiment_strategy_summary_parallel.csv`.

| Strategy | Mean EPmax | Mean PAS | Mean RT | Mean RI | Mean RE |
|---|---:|---:|---:|---:|---:|
| DRAD-CF-MPC | 0.289 | 0.306 | 6.638 | 0.827 | 0.444 |
| CDP | 0.298 | 0.315 | 6.765 | 0.822 | 0.439 |
| DRAD-score | 0.348 | 0.369 | 7.657 | 0.796 | 0.402 |
| CP | 0.465 | 0.499 | 9.697 | 0.716 | 0.319 |
| PDP | 0.465 | 0.503 | 9.770 | 0.714 | 0.316 |
| RP | 0.567 | 0.611 | 11.529 | 0.647 | 0.244 |
| NP | 0.786 | 0.848 | 17.968 | 0.449 | 0.152 |

## 🖼 Paper-used Figures

The overall framework and malware propagation/protection figures are shown above. This section presents the remaining evidence figures. The [final figure index](paper/figures/revision/README.md) identifies the current manuscript numbering while preserving existing filenames for compatibility.

### Information flow and closed-loop execution

<p align="center">
  <a href="paper/figures/revision/fig2b_information_flow.pdf"><img src="assets/revision/information_flow.png" width="880" alt="Observed inputs, candidate evaluation, Safe-Guard decision and closed-loop execution"></a>
</p>

### Key-node Protection and Safe-Guarded Switching

| Key-node evidence | Safe-Guarded switching |
|---|---|
| <img src="assets/key_node_evidence.png" width="420" alt="Key-node protection evidence"> | <img src="assets/safeguarded_switching.png" width="420" alt="Safe-Guarded policy switching"> |

### Degradation-recovery Process Evidence

<p align="center">
  <img src="./assets/process_evidence.png" width="760" alt="Degradation-recovery process evidence">
</p>

### Sensitivity Analysis

<p align="center">
  <img src="./assets/sensitivity_analysis.png" width="760" alt="Sensitivity analysis">
</p>

## 📁 Released Tables

Important CSV files:

- `outputs/tables/experiment_strategy_summary_parallel.csv`: main strategy-level comparison.
- `outputs/tables/experiment_results_optimal_parallel.csv`: scenario-level optimal results.
- `outputs/tables/experiment_results_raw_parallel.csv`: raw main-experiment strategy results.
- `outputs/tables/figure*_used.csv`: released figure-level data used in the manuscript.
- `outputs/tables/comm_features.csv`, `power_features.csv`, and `coupling_*.csv`: CPPS benchmark features and coupling maps.

## 📦 Package Scope

This package is designed for GitHub release and reproducibility checking. It includes runnable code, result CSV files, paper-used figures, and a run log.


## Supplementary experiments and revised figures

The complete experimental resources include the baseline policy-library study and the supplementary validation studies. See [supplementary/README.md](supplementary/README.md) for executable entry points, input preparation and the study map.

- **Code and frozen protocols:** rolling execution, local infection information, observation and delay controls, Safe-Guard thresholds, component comparisons and electrical operating-point checks.
- **Recorded results:** paired outcomes, scenario summaries and physical coverage checks are retained under `supplementary/revision_experiments/results/`. Lossless gzip files can be restored with `python supplementary/prepare_inputs.py`.
- **Figures:** [the final figure index](paper/figures/revision/README.md) maps all 18 PDF assets to the main and supplementary figure numbers.
- **Tables:** [final table exports](paper/tables/revision/README.md) contain main Tables I–IX and supplementary Tables S1–S14, with exact source fragments and an index.
- **Integrity:** `python verify_repository_release.py` checks the packaged file manifest without changing results.
- **Plotting:** [supplementary/plotting](supplementary/plotting/README.md) contains runnable scripts and CSV inputs for the principal new experimental plots. The supplementary gallery generator and its exported PDFs/PNGs/SVGs are also included.

```bash
python supplementary/verify_release.py
python supplementary/prepare_inputs.py
cd supplementary/plotting
python -m pip install -r requirements.txt
python run_all.py
```

The policy-library previews and state-continuing validation use different candidate spaces and sampling designs. Their result files retain those study definitions. Reproducing plots is distinct from rerunning the simulators.

### Local-information and reliability results

<p align="center">
  <a href="paper/figures/revision/local_effects.pdf"><img src="supplementary/plotting/figures/new/local_effects.png" width="860" alt="Matched local-versus-static PAS contrasts by topology, observation recall and execution delay"></a>
</p>
<p align="center">
  <a href="paper/figures/revision/reliability_profile.pdf"><img src="supplementary/plotting/figures/new/reliability_profile.png" width="860" alt="Baseline-relative deterioration, absolute PAS tails and positive-regret tails"></a>
</p>

The matched local-information study separates average loss, baseline-relative deterioration and tail outcomes. The figure captions and supplementary protocols identify the comparison and uncertainty calculation. The original outcome values are retained in the supplied CSV files.


## 👥 Authors

| Author | Affiliation | Email |
|---|---|---|
| Weihao Cheng | School of Communication Engineering, Hangzhou Dianzi University, Hangzhou 310018, China | weihao_cheng@hdu.edu.cn |
| Haicheng Tu | School of Communication Engineering, Hangzhou Dianzi University, Hangzhou 310018, China | tuhc@hdu.edu.cn |
| Youjia Ling | Hangzhou Dianzi University, Hangzhou 310018, China | lingyoujia@hdu.edu.cn |
| Yongxiang Xia | School of Communication Engineering, Hangzhou Dianzi University, Hangzhou 310018, China | xiayx@hdu.edu.cn |
| Xi Chen | Department of Artificial Intelligence, China Electric Power Research Institute, Beijing 100192, China | xc@ieee.org |

## 📝 Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@misc{cheng2026dradcfmpc,
  title  = {Safe-Guarded Counterfactual Defense for Malware-Resilient Cyber-Physical Power Systems},
  author = {Cheng, Weihao and Tu, Haicheng and Ling, Youjia and Xia, Yongxiang and Chen, Xi},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 📜 License

This repository is released under the MIT License.
