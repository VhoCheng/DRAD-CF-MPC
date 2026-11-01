from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"
SOURCE = OUT / "21_balanced_holdout_full"

METHODS = ["CDP", "Original", "Balanced"]

# Exploratory service-impact thresholds.
# These are NOT physical power-flow failure limits.
THRESHOLDS = [0.10, 0.20, 0.30]

B = 3000
RNG = np.random.default_rng(20261019)


def main():
    print("=" * 76)
    print("TNSE EXPERIMENT 24")
    print("EP DEFINITION AND SERVICE-IMPACT AUDIT")
    print("=" * 76)

    df = pd.read_csv(SOURCE / "per_run.csv")
    summary = pd.read_csv(SOURCE / "summary.csv")

    if len(df) != 1440:
        raise RuntimeError("Expected 1440 run records")

    if len(summary) != 120:
        raise RuntimeError("Expected 120 summary records")

    keys = ["scenario_id", "run_index"]

    if df.duplicated(keys + ["variant"]).any():
        raise RuntimeError("Duplicate run records")

    if set(df["variant"]) != set(METHODS):
        raise RuntimeError("Unexpected methods")

    if not df["failed"].isin([0, 1]).all():
        raise RuntimeError("Invalid failure flag")

    failed_rows = df[df["failed"] == 1]

    if (
        not failed_rows.empty
        and not np.allclose(
            failed_rows["PAS"].to_numpy(),
            0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError(
            "PAS was not zero on attack failure"
        )

    agg = (
        df.groupby(
            ["scenario_id", "variant"],
            as_index=False
        )
        .agg(
            mean_PAS=("PAS", "mean"),
            failure_probability=("failed", "mean"),
        )
    )

    agg["PS_rebuilt"] = (
        1.0 - agg["failure_probability"]
    )

    agg["EP_original_rebuilt"] = (
        agg["mean_PAS"] * agg["PS_rebuilt"]
    )

    # Single-run loss already contains zeros on failure.
    agg["expected_single_run_loss"] = (
        agg["mean_PAS"]
    )

    agg["conditional_success_PAS"] = np.where(
        agg["PS_rebuilt"] > 1e-12,
        agg["mean_PAS"] / agg["PS_rebuilt"],
        np.nan,
    )

    agg["double_weight_gap"] = (
        agg["expected_single_run_loss"]
        - agg["EP_original_rebuilt"]
    )

    check = summary.merge(
        agg,
        on=["scenario_id", "variant"],
        validate="one_to_one",
    )

    errors = {
        "EP": float(np.max(np.abs(
            check["EP"]
            - check["EP_original_rebuilt"]
        ))),
        "PAS": float(np.max(np.abs(
            check["PAS"]
            - check["mean_PAS"]
        ))),
        "PS": float(np.max(np.abs(
            check["PS"]
            - check["PS_rebuilt"]
        ))),
    }

    if max(errors.values()) > 1e-10:
        raise RuntimeError(
            f"HISTORICAL METRIC CHECK FAILED: {errors}"
        )

    print(
        "HISTORICAL EP / PAS / PS "
        "RECONSTRUCTION PASSED"
    )

    print("Maximum errors:", errors)

    overall = (
        agg.groupby("variant", as_index=False)
        .agg(
            original_EP=("EP_original_rebuilt", "mean"),
            expected_single_run_loss=(
                "expected_single_run_loss", "mean"
            ),
            mean_success_probability=(
                "PS_rebuilt", "mean"
            ),
            mean_double_weight_gap=(
                "double_weight_gap", "mean"
            ),
        )
    )

    # Paired scenario-level differences.
    wide_old = (
        agg.pivot(
            index="scenario_id",
            columns="variant",
            values="EP_original_rebuilt",
        )
    )

    wide_new = (
        agg.pivot(
            index="scenario_id",
            columns="variant",
            values="expected_single_run_loss",
        )
    )

    ids = sorted(df["scenario_id"].unique())

    meta = (
        summary.drop_duplicates("scenario_id")
        .set_index("scenario_id")
        .loc[ids]
    )

    patterns = meta["pattern"].to_numpy()

    groups = [
        np.where(patterns == name)[0]
        for name in sorted(set(patterns))
    ]

    if len(groups) != 5:
        raise RuntimeError(
            "Expected five coupling structures"
        )

    if any(len(g) != 8 for g in groups):
        raise RuntimeError(
            "Unexpected number of scenarios per coupling"
        )

    pas_wide = df.pivot(
        index=keys,
        columns="variant",
        values="PAS",
    )

    fail_wide = df.pivot(
        index=keys,
        columns="variant",
        values="failed",
    )

    pas = np.stack([
        pas_wide.loc[
            [(sid, r) for r in range(12)],
            METHODS,
        ].to_numpy(dtype=float)
        for sid in ids
    ])

    failed = np.stack([
        fail_wide.loc[
            [(sid, r) for r in range(12)],
            METHODS,
        ].to_numpy(dtype=float)
        for sid in ids
    ])

    if pas.shape != (40, 12, 3):
        raise RuntimeError("Invalid paired tensor")

    draw_s = np.concatenate([
        RNG.choice(
            group,
            size=(B, len(group)),
            replace=True,
        )
        for group in groups
    ], axis=1)

    draw_r = RNG.integers(
        0, 12,
        size=(B, 40, 12),
    )

    boot_pas = pas[
        draw_s[:, :, None],
        draw_r,
        :
    ].mean(axis=2)

    boot_ps = (
        1.0
        - failed[
            draw_s[:, :, None],
            draw_r,
            :
        ].mean(axis=2)
    )

    boot_old = boot_pas * boot_ps
    boot_new = boot_pas

    comparisons = []

    index = {
        name: i
        for i, name in enumerate(METHODS)
    }

    for metric_name, wide, tensor in [
        ("Original EP", wide_old, boot_old),
        ("Unconditional expected loss", wide_new, boot_new),
    ]:
        for proposed, baseline in [
            ("Original", "CDP"),
            ("Balanced", "CDP"),
            ("Balanced", "Original"),
        ]:
            a = index[proposed]
            b = index[baseline]

            differences = (
                wide.loc[ids, baseline].to_numpy()
                - wide.loc[ids, proposed].to_numpy()
            )

            sampled = (
                tensor[:, :, b]
                - tensor[:, :, a]
            ).mean(axis=1)

            lo, hi = np.quantile(
                sampled,
                [0.025, 0.975],
            )

            if np.all(
                np.abs(differences) < 1e-14
            ):
                p = 1.0
            else:
                p = float(wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                ).pvalue)

            comparisons.append({
                "metric": metric_name,
                "comparison": (
                    f"{proposed} vs {baseline}"
                ),
                "mean_gain": float(
                    differences.mean()
                ),
                "CI_low": float(lo),
                "CI_high": float(hi),
                "exploratory_wilcoxon_p": p,
                "better_scenarios": int(
                    (differences > 1e-12).sum()
                ),
                "worse_scenarios": int(
                    (differences < -1e-12).sum()
                ),
                "tied_scenarios": int(
                    (np.abs(differences) <= 1e-12).sum()
                ),
            })

    comparison_df = pd.DataFrame(comparisons)

    # Sensitivity analysis for service-impact thresholds.
    # Threshold crossings are NOT AC power-flow violations.
    threshold_rows = []

    for threshold in THRESHOLDS:
        for name in METHODS:
            sub = df[df["variant"] == name]

            x = sub["PAS"].to_numpy(dtype=float)

            threshold_rows.append({
                "threshold": threshold,
                "method": name,
                "n_runs": len(x),
                "proxy_exceedance_rate": float(
                    np.mean(x > threshold)
                ),
                "pooled_PAS_P95": float(
                    np.quantile(x, 0.95)
                ),
                "pooled_PAS_P99": float(
                    np.quantile(x, 0.99)
                ),
                "pooled_PAS_max": float(
                    x.max()
                ),
            })

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    # Stratify by budget/onset using the unchanged
    # historical reported EP definition.
    merged_meta = meta[
        ["pattern", "info", "budget", "onset"]
    ].join(wide_old.loc[ids, METHODS])

    grouped = (
        merged_meta.groupby(
            ["budget", "onset"]
        )[METHODS]
        .mean()
        .reset_index()
    )

    grouped["Original_minus_CDP"] = (
        grouped["Original"]
        - grouped["CDP"]
    )

    grouped["Balanced_minus_CDP"] = (
        grouped["Balanced"]
        - grouped["CDP"]
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    prefix = f"24_metric_audit_{stamp}"

    files = {
        "overall": OUT / f"{prefix}_overall.csv",
        "comparisons": OUT / f"{prefix}_comparisons.csv",
        "thresholds": OUT / f"{prefix}_thresholds.csv",
        "conditions": OUT / f"{prefix}_conditions.csv",
        "per_scenario": OUT / f"{prefix}_per_scenario.csv",
    }

    overall.to_csv(files["overall"], index=False)
    comparison_df.to_csv(
        files["comparisons"], index=False
    )
    threshold_df.to_csv(
        files["thresholds"], index=False
    )
    grouped.to_csv(
        files["conditions"], index=False
    )
    check.to_csv(
        files["per_scenario"], index=False
    )

    print("\n" + "=" * 76)
    print("ORIGINAL EP VS UNCONDITIONAL EXPECTED LOSS")
    print("=" * 76)
    print(overall.to_string(index=False))

    print("\n" + "=" * 76)
    print("PAIRED METRIC COMPARISONS")
    print("=" * 76)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 76)
    print("SERVICE-IMPACT THRESHOLD SENSITIVITY")
    print("=" * 76)
    print(threshold_df.to_string(index=False))

    print("\n" + "=" * 76)
    print("RESULTS BY BUDGET AND ATTACK ONSET")
    print("=" * 76)
    print(grouped.to_string(index=False))

    print(
        "\nIMPORTANT: These thresholds are descriptive "
        "service-impact proxies, not physical stability "
        "or AC power-flow constraints."
    )

    for key, path in files.items():
        print(f"Saved {key}: {path}")

    print(
        "\nEXPERIMENT 24 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
