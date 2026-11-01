# Plotting recorded experimental results

From this directory, install `requirements.txt` and run `python run_all.py`. The two scripts regenerate `local_effects`, `reliability_profile`, `guard_cost` and `parameter_sensitivity` from the bundled, hash-recorded CSV inputs. They do not rerun simulations or alter outcome data. Numerical labels may be rounded for presentation; plotted points retain their original precision.

`figures/` also contains the final supplementary evidence figures. Additional gallery figures and their source generator are under `../revision_experiments/figures/` and `../tnse_advanced_figures_v1.py`. The Chinese guide documents the data definitions and method-name mapping.

Final figure PDFs are also collected under `paper/figures/revision/` at the repository root. Hand-edited figures are provided as the authors supplied them; the experimental result CSVs remain unchanged.

For the final paper numbering and table exports, see [the figure index](../../paper/figures/revision/README.md) and [the table index](../../paper/tables/revision/README.md).
