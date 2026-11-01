from pathlib import Path
from datetime import datetime, timezone
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "CDP",
    "Full",
    "NoSafeGuard",
    "NoDynamicRisk",
    "RiskHeuristicNoCF",
]

COMPARISONS = [
    ("Full", "CDP"),
    ("NoSafeGuard", "Full"),
    ("NoDynamicRisk", "Full"),
    ("RiskHeuristicNoCF", "Full"),
]

BOOTSTRAPS = 3000
RNG_SEED = 20261018


def load_data():
    files = sorted(
        OUT.glob("03_ablation_full_*_summary.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            "No full ablation summary CSV found."
        )

    summary_path = files[0]

    prefix = summary_path.name.removesuffix(
        "_summary.csv"
    )

    run_path = OUT / f"{prefix}_per_run.csv"

    if not run_path.exists():
        raise FileNotFoundError(
            f"Missing per-run file: {run_path}"
        )

    summary = pd.read_csv(summary_path)
    runs = pd.read_csv(run_path)

    return summary, runs, summary_path, run_path


def verify_data(summary, runs):
    required_summary = {
        "scenario_id", "pattern", "variant",
        "EP", "PAS", "PS",
    }

    required_runs = {
        "scenario_id", "seed", "variant",
        "PAS", "failed",
    }

    if not required_summary.issubset(summary.columns):
        raise RuntimeError(
            "Missing summary columns."
        )

    if not required_runs.issubset(runs.columns):
        raise RuntimeError(
            "Missing per-run columns."
        )

    if len(summary) != 300:
        raise RuntimeError(
            f"Expected 300 summary rows; got {len(summary)}"
        )

    if len(runs) != 3000:
        raise RuntimeError(
            f"Expected 3000 per-run rows; got {len(runs)}"
        )

    if summary["scenario_id"].nunique() != 60:
        raise RuntimeError(
            "Expected 60 scenarios."
        )

    if set(summary["variant"]) != set(VARIANTS):
        raise RuntimeError(
            "Unexpected ablation variants."
        )

    counts = runs.groupby(
        ["scenario_id", "variant"]
    )["seed"].nunique()

    if not np.all(counts.to_numpy() == 10):
        raise RuntimeError(
            "Every scenario/variant must have 10 seeds."
        )

    if summary.duplicated(
        ["scenario_id", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate summary entries."
        )

    if runs.duplicated(
        ["scenario_id", "seed", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate per-run entries."
        )

    print("DATA VALIDATION PASSED")
    print("Scenarios: 60")
    print("Variants: 5")
    print("Runs per scenario: 10")
    print("Per-run records: 3000")


def build_matrices(summary, runs):
    scenario_ids = sorted(
        summary["scenario_id"].unique()
    )

    ep_wide = summary.pivot(
        index="scenario_id",
        columns="variant",
        values="EP",
    ).loc[scenario_ids, VARIANTS]

    pas_wide = runs.pivot(
        index=["scenario_id", "seed"],
        columns="variant",
        values="PAS",
    )

    failed_wide = runs.pivot(
        index=["scenario_id", "seed"],
        columns="variant",
        values="failed",
    )

    if pas_wide.isna().any().any():
        raise RuntimeError("Missing PAS data.")

    if failed_wide.isna().any().any():
        raise RuntimeError("Missing failure data.")

    pas_matrix = np.stack([
        pas_wide.loc[sid, VARIANTS].to_numpy(
            dtype=float
        )
        for sid in scenario_ids
    ])

    failed_matrix = np.stack([
        failed_wide.loc[sid, VARIANTS].to_numpy(
            dtype=float
        )
        for sid in scenario_ids
    ])

    # Expected shape: scenarios x MC runs x variants.
    assert pas_matrix.shape == (60, 10, 5)
    assert failed_matrix.shape == (60, 10, 5)

    reconstructed_ep = (
        pas_matrix.mean(axis=1)
        * (1.0 - failed_matrix.mean(axis=1))
    )

    original_ep = ep_wide.to_numpy(
        dtype=float
    )

    max_error = float(np.max(np.abs(
        reconstructed_ep - original_ep
    )))

    if max_error > 1e-10:
        raise RuntimeError(
            "EP RECONSTRUCTION FAILED: "
            f"max_error={max_error}"
        )

    print(
        "EP RECONSTRUCTION PASSED; "
        f"max_error={max_error:.3e}"
    )

    return (
        scenario_ids,
        original_ep,
        pas_matrix,
        failed_matrix,
    )


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    n = len(values)

    order = np.argsort(values)
    ranked = values[order]

    adjusted = (
        ranked * (n - np.arange(n))
    )

    adjusted = np.maximum.accumulate(
        adjusted
    )

    adjusted = np.minimum(adjusted, 1.0)

    result = np.empty(n, dtype=float)
    result[order] = adjusted

    return result


def main():
    print("=" * 75)
    print("TNSE MAJOR REVISION")
    print("EXPERIMENT 04: PAIRED STATISTICAL ANALYSIS")
    print("=" * 75)

    summary, runs, source_summary, source_runs = (
        load_data()
    )

    print("Source summary:", source_summary)
    print("Source per-run:", source_runs)

    verify_data(summary, runs)

    scenario_ids, ep, pas, failed = (
        build_matrices(summary, runs)
    )

    index = {
        name: i
        for i, name in enumerate(VARIANTS)
    }

    # Stratify scenario resampling by coupling pattern.
    pattern_map = (
        summary.drop_duplicates("scenario_id")
        .set_index("scenario_id")["pattern"]
    )

    coupling_labels = [
        pattern_map.loc[sid]
        for sid in scenario_ids
    ]

    rng = np.random.default_rng(RNG_SEED)

    groups = [
        np.array([
            i for i, name in enumerate(coupling_labels)
            if name == coupling
        ], dtype=int)
        for coupling in sorted(
            set(coupling_labels)
        )
    ]

    if any(len(group) != 12 for group in groups):
        raise RuntimeError(
            "Unexpected coupling group sizes."
        )

    # Stage 1: sample scenarios within each coupling.
    sampled_scenarios = np.concatenate([
        rng.choice(
            group,
            size=(BOOTSTRAPS, len(group)),
            replace=True,
        )
        for group in groups
    ], axis=1)

    # Stage 2: sample Monte Carlo runs within each
    # selected scenario, preserving pairing of variants.
    sampled_runs = rng.integers(
        0, 10,
        size=(BOOTSTRAPS, 60, 10),
    )

    bootstrap_pas = pas[
        sampled_scenarios[:, :, None],
        sampled_runs,
    ].mean(axis=2)

    bootstrap_ps = (
        1.0
        - failed[
            sampled_scenarios[:, :, None],
            sampled_runs,
        ].mean(axis=2)
    )

    bootstrap_ep = (
        bootstrap_pas * bootstrap_ps
    )

    rows = []

    for left, right in COMPARISONS:
        a = index[left]
        b = index[right]

        differences = ep[:, a] - ep[:, b]

        boot_diff = (
            bootstrap_ep[:, :, a]
            - bootstrap_ep[:, :, b]
        ).mean(axis=1)

        ci_low, ci_high = np.quantile(
            boot_diff,
            [0.025, 0.975],
        )

        if np.allclose(
            differences, 0.0, atol=1e-15
        ):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(
                    differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )

        rows.append({
            "method_A": left,
            "method_B": right,
            "mean_EP_A": float(ep[:, a].mean()),
            "mean_EP_B": float(ep[:, b].mean()),
            "mean_delta_EP_A_minus_B": (
                float(differences.mean())
            ),
            "bootstrap_CI_low": float(ci_low),
            "bootstrap_CI_high": float(ci_high),
            "wilcoxon_p": p_value,
            "A_better_scenarios": int(
                np.sum(differences < -1e-12)
            ),
            "A_worse_scenarios": int(
                np.sum(differences > 1e-12)
            ),
            "tied_scenarios": int(
                np.sum(
                    np.abs(differences) <= 1e-12
                )
            ),
            "mean_delta_PAS": float(
                pas[:, :, a].mean()
                - pas[:, :, b].mean()
            ),
        })

    result = pd.DataFrame(rows)

    result["holm_adjusted_p"] = holm_adjust(
        result["wilcoxon_p"].to_numpy()
    )

    # Metric-definition diagnostic:
    # Failed-attack runs already have PAS = 0.
    # Therefore unconditional mean PAS and
    # mean PAS multiplied by PS are different metrics.
    metric_rows = []

    for name in VARIANTS:
        i = index[name]

        ps_by_scenario = (
            1.0 - failed[:, :, i].mean(axis=1)
        )

        pas_by_scenario = (
            pas[:, :, i].mean(axis=1)
        )

        original_ep_by_scenario = (
            pas_by_scenario * ps_by_scenario
        )

        metric_rows.append({
            "variant": name,
            "mean_original_EP": float(
                original_ep_by_scenario.mean()
            ),
            "mean_unconditional_PAS": float(
                pas_by_scenario.mean()
            ),
            "mean_PS": float(
                ps_by_scenario.mean()
            ),
            "mean_extra_weighting_effect": float(
                (
                    pas_by_scenario
                    - original_ep_by_scenario
                ).mean()
            ),
        })

    metric_result = pd.DataFrame(
        metric_rows
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    stats_path = (
        OUT / f"04_paired_stats_{timestamp}.csv"
    )

    metric_path = (
        OUT / f"04_metric_audit_{timestamp}.csv"
    )

    metadata_path = (
        OUT / f"04_stats_config_{timestamp}.json"
    )

    result.to_csv(
        stats_path, index=False
    )

    metric_result.to_csv(
        metric_path, index=False
    )

    metadata_path.write_text(
        json.dumps({
            "experiment": "Paired statistical analysis",
            "bootstrap_iterations": BOOTSTRAPS,
            "bootstrap_seed": RNG_SEED,
            "bootstrap_method": (
                "Stratified scenario and "
                "paired within-scenario run resampling"
            ),
            "source_summary": str(source_summary),
            "source_per_run": str(source_runs),
            "comparison_count": len(COMPARISONS),
            "significance_test": (
                "Two-sided paired Wilcoxon"
            ),
            "multiple_testing": "Holm",
        }, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("PAIRED STATISTICAL RESULTS")
    print("=" * 75)

    print(result.to_string(index=False))

    print("\n" + "=" * 75)
    print("METRIC-DEFINITION AUDIT")
    print("=" * 75)

    print(
        metric_result.to_string(index=False)
    )

    print("\nInterpretation:")
    print(
        "Positive Delta EP (A-B) = "
        "method A has higher EP (worse)."
    )
    print(
        "Negative Delta EP (A-B) = "
        "method A has lower EP (better)."
    )
    print(
        "Confidence intervals reflect bootstrap "
        "resampling of the tested scenarios and runs."
    )

    print("\nSaved statistics:", stats_path)
    print("Saved metric audit:", metric_path)
    print("Saved configuration:", metadata_path)

    print("EXPERIMENT 04 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
