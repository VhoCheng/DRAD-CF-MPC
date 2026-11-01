from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"

VARIANTS = [
    "CDP",
    "Legacy-K5",
    "MoreCF-K20",
    "Confirm15-K05",
    "Confirm15-K10",
    "Confirm15-Robust",
]

TARGET = "Confirm15-Robust"
COMPARISONS = ["Legacy-K5", "CDP"]

B = 3000
RNG = np.random.default_rng(20261018)


def holm_two(p):
    order = np.argsort(p)
    result = np.zeros(len(p))

    previous = 0.0
    for rank, position in enumerate(order):
        adjusted = min(
            1.0, float(p[position]) * (len(p) - rank)
        )
        previous = max(previous, adjusted)
        result[position] = previous

    return result


def main():
    files = sorted(
        OUT.glob("12_holdout_test_*_summary.csv"),
        key=lambda p: p.stat().st_mtime
    )

    if not files:
        raise FileNotFoundError(
            "Experiment 12 holdout summary not found"
        )

    source = files[-1]
    prefix = source.name.removesuffix("_summary.csv")
    detail_source = OUT / f"{prefix}_per_run.csv"

    summary = pd.read_csv(source)
    details = pd.read_csv(detail_source)

    assert len(summary) == 240, (
        f"Expected 240 summary rows, got {len(summary)}"
    )
    assert len(details) == 2400, (
        f"Expected 2400 per-run rows, got {len(details)}"
    )

    assert set(summary["variant"]) == set(VARIANTS)
    assert summary["scenario_id"].nunique() == 40

    assert not summary.duplicated(
        ["scenario_id", "variant"]
    ).any()

    assert not details.duplicated(
        ["scenario_id", "run_index", "variant"]
    ).any()

    ids = sorted(summary["scenario_id"].unique())

    wide_pas = details.pivot(
        index=["scenario_id", "run_index"],
        columns="variant",
        values="PAS"
    )

    wide_failed = details.pivot(
        index=["scenario_id", "run_index"],
        columns="variant",
        values="failed"
    )

    if wide_pas.isna().any().any():
        raise RuntimeError("Missing PAS observations")

    if wide_failed.isna().any().any():
        raise RuntimeError("Missing failed observations")

    pas = np.stack([
        wide_pas.loc[sid, VARIANTS].to_numpy(
            dtype=float
        )
        for sid in ids
    ])

    failed = np.stack([
        wide_failed.loc[sid, VARIANTS].to_numpy(
            dtype=float
        )
        for sid in ids
    ])

    assert pas.shape == (40, 10, 6)
    assert failed.shape == (40, 10, 6)

    reconstructed = (
        pas.mean(axis=1)
        * (1.0 - failed.mean(axis=1))
    )

    ref = (
        summary.pivot(
            index="scenario_id",
            columns="variant",
            values="EP"
        )
        .loc[ids, VARIANTS]
        .to_numpy(dtype=float)
    )

    error = float(
        np.max(np.abs(reconstructed - ref))
    )

    if error > 1e-10:
        raise RuntimeError(
            f"EP reconstruction failed: {error}"
        )

    print("=" * 75)
    print("TNSE EXPERIMENT 13 - FROZEN HOLDOUT STATISTICS")
    print("=" * 75)
    print("DATA VALIDATION PASSED")
    print("Scenarios: 40")
    print("Paired Monte Carlo repetitions: 10")
    print("Method-level records: 2400")
    print("EP reconstruction error:", error)

    idx = {
        name: i for i, name in enumerate(VARIANTS)
    }

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

    assert len(groups) == 5
    assert all(len(group) == 8 for group in groups)

    # Same resampling indices for all methods:
    # preserves the paired experimental design.
    bs_scenarios = np.concatenate([
        RNG.choice(
            group,
            size=(B, len(group)),
            replace=True
        )
        for group in groups
    ], axis=1)

    bs_runs = RNG.integers(
        low=0,
        high=10,
        size=(B, 40, 10)
    )

    bs_pas = pas[
        bs_scenarios[:, :, None],
        bs_runs,
        :
    ].mean(axis=2)

    bs_ps = 1.0 - failed[
        bs_scenarios[:, :, None],
        bs_runs,
        :
    ].mean(axis=2)

    bs_ep = bs_pas * bs_ps

    main_rows = []

    for opponent in COMPARISONS:
        a = idx[opponent]
        b = idx[TARGET]

        # Positive means Robust is better.
        gain = reconstructed[:, a] - reconstructed[:, b]

        boot_base = bs_ep[:, :, a].mean(axis=1)
        boot_robust = bs_ep[:, :, b].mean(axis=1)

        boot_gain = boot_base - boot_robust

        ci = np.quantile(
            boot_gain,
            [0.025, 0.975]
        )

        boot_pct = np.where(
            boot_base > 1e-12,
            100.0 * boot_gain / boot_base,
            np.nan
        )

        pct_ci = np.nanquantile(
            boot_pct,
            [0.025, 0.975]
        )

        if np.all(np.abs(gain) < 1e-14):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(
                    gain,
                    alternative="two-sided",
                    zero_method="wilcox"
                ).pvalue
            )

        denominator = reconstructed[:, a].mean()

        main_rows.append({
            "comparison": f"{TARGET} vs {opponent}",
            "mean_opponent_EP": float(denominator),
            "mean_robust_EP": float(
                reconstructed[:, b].mean()
            ),
            "mean_EP_improvement": float(gain.mean()),
            "relative_improvement_pct": (
                100.0 * gain.mean() / denominator
            ),
            "bootstrap_CI_low": float(ci[0]),
            "bootstrap_CI_high": float(ci[1]),
            "relative_CI_low_pct": float(pct_ci[0]),
            "relative_CI_high_pct": float(pct_ci[1]),
            "wilcoxon_p": p_value,
            "robust_better_scenarios": int(
                np.sum(gain > 1e-12)
            ),
            "robust_worse_scenarios": int(
                np.sum(gain < -1e-12)
            ),
            "tied_scenarios": int(
                np.sum(np.abs(gain) <= 1e-12)
            ),
            "robust_worse_MC_rate": float(
                np.mean(pas[:, :, b] > pas[:, :, a] + 1e-12)
            ),
        })

    results = pd.DataFrame(main_rows)
    results["holm_adjusted_p"] = holm_two(
        results["wilcoxon_p"].to_numpy()
    )

    coupling_rows = []

    for pattern in sorted(set(patterns)):
        positions = np.where(patterns == pattern)[0]

        for opponent in COMPARISONS:
            a = idx[opponent]
            b = idx[TARGET]

            ep_a = reconstructed[positions, a]
            ep_b = reconstructed[positions, b]
            diff = ep_a - ep_b

            coupling_rows.append({
                "pattern": pattern,
                "comparison": f"{TARGET} vs {opponent}",
                "mean_opponent_EP": float(ep_a.mean()),
                "mean_robust_EP": float(ep_b.mean()),
                "EP_improvement": float(diff.mean()),
                "relative_improvement_pct": (
                    100.0 * diff.mean() / ep_a.mean()
                    if ep_a.mean() > 1e-12
                    else np.nan
                ),
                "better_scenarios": int(
                    (diff > 1e-12).sum()
                ),
                "worse_scenarios": int(
                    (diff < -1e-12).sum()
                ),
            })

    coupling_df = pd.DataFrame(coupling_rows)

    decisions = details.pivot(
        index=["scenario_id", "run_index"],
        columns="variant",
        values="selected_policy"
    )

    same_k10 = float(
        (
            decisions[TARGET]
            == decisions["Confirm15-K10"]
        ).mean()
    )

    scenario_df = meta[
        ["pattern", "info", "budget", "onset"]
    ].copy()

    scenario_df["CDP_EP"] = reconstructed[:, idx["CDP"]]
    scenario_df["Legacy_EP"] = reconstructed[
        :, idx["Legacy-K5"]
    ]
    scenario_df["Robust_EP"] = reconstructed[:, idx[TARGET]]

    scenario_df["gain_vs_CDP"] = (
        scenario_df["CDP_EP"] - scenario_df["Robust_EP"]
    )

    scenario_df["gain_vs_Legacy"] = (
        scenario_df["Legacy_EP"] - scenario_df["Robust_EP"]
    )

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    p1 = OUT / f"13_holdout_stats_{stamp}.csv"
    p2 = OUT / f"13_holdout_by_coupling_{stamp}.csv"
    p3 = OUT / f"13_holdout_scenarios_{stamp}.csv"

    results.to_csv(p1, index=False)
    coupling_df.to_csv(p2, index=False)
    scenario_df.to_csv(p3)

    print("\n" + "=" * 75)
    print("PRIMARY HOLDOUT STATISTICAL RESULTS")
    print("=" * 75)
    print(results.to_string(index=False))

    print("\n" + "=" * 75)
    print("COUPLING-SPECIFIC RESULTS")
    print("=" * 75)
    print(coupling_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("ROBUST VS K10 POLICY AGREEMENT")
    print("=" * 75)
    print(f"Same selected policy: {same_k10:.2%}")

    print("\n" + "=" * 75)
    print("FIVE WORST ROBUST CASES VS LEGACY")
    print("=" * 75)

    print(
        scenario_df.sort_values(
            "gain_vs_Legacy"
        ).head(5).to_string()
    )

    print("\nSaved:", p1)
    print("Saved:", p2)
    print("Saved:", p3)
    print("EXPERIMENT 13 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
