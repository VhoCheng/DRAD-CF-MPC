from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT
    / "revision_experiments/results/15_transmission_full"
)

OUT = ROOT / "revision_experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "CDP",
    "Legacy-K5",
    "Confirm15-Robust",
]

MECHANISMS = [
    "original",
    "degree_source",
    "degree_target",
    "temporal_burst",
]

COMPARISONS = [
    ("Confirm15-Robust", "CDP"),
    ("Confirm15-Robust", "Legacy-K5"),
]

BOOTSTRAPS = 3000
RNG = np.random.default_rng(20261019)


def holm_adjust(values):
    p = np.asarray(values, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    corrected = ranked * (
        n - np.arange(n)
    )

    corrected = np.maximum.accumulate(corrected)
    corrected = np.minimum(corrected, 1.0)

    result = np.empty_like(corrected)
    result[order] = corrected

    return result


def main():
    print("=" * 75)
    print("TNSE EXPERIMENT 17")
    print("TRANSMISSION-MECHANISM PAIRED STATISTICS")
    print("=" * 75)

    summary_path = SOURCE / "summary.csv"
    detail_path = SOURCE / "per_run.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    if not detail_path.exists():
        raise FileNotFoundError(detail_path)

    summary = pd.read_csv(summary_path)
    details = pd.read_csv(detail_path)

    if len(summary) != 480:
        raise RuntimeError(
            f"Expected 480 summary rows, got {len(summary)}"
        )

    if len(details) != 4800:
        raise RuntimeError(
            f"Expected 4800 per-run rows, got {len(details)}"
        )

    summary_key = [
        "base_scenario_id",
        "mechanism",
        "variant",
    ]

    detail_key = summary_key + ["run_index"]

    if summary.duplicated(summary_key).any():
        raise RuntimeError(
            "Duplicate scenario-method summary rows"
        )

    if details.duplicated(detail_key).any():
        raise RuntimeError(
            "Duplicate per-run rows"
        )

    if set(summary["variant"]) != set(VARIANTS):
        raise RuntimeError(
            "Unexpected defense methods"
        )

    if set(summary["mechanism"]) != set(MECHANISMS):
        raise RuntimeError(
            "Unexpected propagation mechanisms"
        )

    results = []
    coupling_results = []

    for mechanism in MECHANISMS:
        sm = summary[
            summary["mechanism"] == mechanism
        ].copy()

        dt = details[
            details["mechanism"] == mechanism
        ].copy()

        ids = sorted(
            sm["base_scenario_id"].unique()
        )

        if len(ids) != 40:
            raise RuntimeError(
                f"{mechanism}: expected 40 scenarios"
            )

        counts = dt.groupby(
            ["base_scenario_id", "variant"]
        )["run_index"].nunique()

        if not (counts == 10).all():
            raise RuntimeError(
                f"{mechanism}: missing MC runs"
            )

        pas_wide = dt.pivot(
            index=["base_scenario_id", "run_index"],
            columns="variant",
            values="PAS",
        )

        failed_wide = dt.pivot(
            index=["base_scenario_id", "run_index"],
            columns="variant",
            values="failed",
        )

        if (
            pas_wide.isna().any().any()
            or failed_wide.isna().any().any()
        ):
            raise RuntimeError(
                f"{mechanism}: incomplete paired observations"
            )

        pas = np.stack([
            pas_wide.loc[sid, VARIANTS].to_numpy(
                dtype=float
            )
            for sid in ids
        ])

        failed = np.stack([
            failed_wide.loc[sid, VARIANTS].to_numpy(
                dtype=float
            )
            for sid in ids
        ])

        if pas.shape != (40, 10, 3):
            raise RuntimeError(
                f"Invalid PAS array: {pas.shape}"
            )

        ep = (
            pas.mean(axis=1)
            * (1.0 - failed.mean(axis=1))
        )

        original_ep = (
            sm.pivot(
                index="base_scenario_id",
                columns="variant",
                values="EP",
            )
            .loc[ids, VARIANTS]
            .to_numpy(dtype=float)
        )

        max_error = float(
            np.max(np.abs(ep - original_ep))
        )

        if max_error > 1e-10:
            raise RuntimeError(
                f"{mechanism}: EP mismatch {max_error}"
            )

        print(
            f"EP RECONSTRUCTION PASSED: "
            f"{mechanism}, error={max_error:.3e}",
            flush=True,
        )

        metadata = (
            sm.drop_duplicates("base_scenario_id")
            .set_index("base_scenario_id")
            .loc[ids]
        )

        patterns = metadata["pattern"].to_numpy()

        pattern_groups = [
            np.where(patterns == name)[0]
            for name in sorted(set(patterns))
        ]

        if len(pattern_groups) != 5:
            raise RuntimeError(
                "Expected five coupling categories"
            )

        if any(
            len(indices) != 8
            for indices in pattern_groups
        ):
            raise RuntimeError(
                "Expected eight scenarios per coupling"
            )

        # Stratified paired bootstrap:
        # sample scenarios and then paired MC repetitions.
        scenario_draws = np.concatenate([
            RNG.choice(
                indices,
                size=(BOOTSTRAPS, len(indices)),
                replace=True,
            )
            for indices in pattern_groups
        ], axis=1)

        run_draws = RNG.integers(
            0,
            10,
            size=(BOOTSTRAPS, 40, 10),
        )

        sampled_pas = pas[
            scenario_draws[:, :, None],
            run_draws,
            :
        ].mean(axis=2)

        sampled_ps = (
            1.0
            - failed[
                scenario_draws[:, :, None],
                run_draws,
                :
            ].mean(axis=2)
        )

        sampled_ep = sampled_pas * sampled_ps

        index = {
            name: i
            for i, name in enumerate(VARIANTS)
        }

        for proposed, baseline in COMPARISONS:
            a = index[proposed]
            b = index[baseline]

            # Positive means proposed has lower EP.
            gain = ep[:, b] - ep[:, a]

            boot_baseline = sampled_ep[
                :, :, b
            ].mean(axis=1)

            boot_proposed = sampled_ep[
                :, :, a
            ].mean(axis=1)

            boot_gain = (
                boot_baseline - boot_proposed
            )

            interval = np.quantile(
                boot_gain,
                [0.025, 0.975],
            )

            if np.all(
                np.abs(gain) < 1e-14
            ):
                p = 1.0
            else:
                p = float(
                    wilcoxon(
                        gain,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )

            baseline_mean = float(
                ep[:, b].mean()
            )

            proposed_mean = float(
                ep[:, a].mean()
            )

            results.append({
                "mechanism": mechanism,
                "comparison": (
                    f"{proposed} vs {baseline}"
                ),
                "baseline_EP": baseline_mean,
                "proposed_EP": proposed_mean,
                "EP_improvement": float(
                    gain.mean()
                ),
                "relative_improvement_pct": (
                    100.0
                    * gain.mean()
                    / baseline_mean
                    if baseline_mean > 1e-12
                    else np.nan
                ),
                "CI_low": float(interval[0]),
                "CI_high": float(interval[1]),
                "wilcoxon_p": p,
                "better_scenarios": int(
                    np.sum(gain > 1e-12)
                ),
                "worse_scenarios": int(
                    np.sum(gain < -1e-12)
                ),
                "tied_scenarios": int(
                    np.sum(
                        np.abs(gain) <= 1e-12
                    )
                ),
                "negative_paired_PAS_rate": float(
                    np.mean(
                        pas[:, :, a]
                        > pas[:, :, b] + 1e-12
                    )
                ),
            })

            for pattern in sorted(set(patterns)):
                mask = patterns == pattern

                baseline_pattern = float(
                    ep[mask, b].mean()
                )

                proposed_pattern = float(
                    ep[mask, a].mean()
                )

                coupling_results.append({
                    "mechanism": mechanism,
                    "pattern": pattern,
                    "comparison": (
                        f"{proposed} vs {baseline}"
                    ),
                    "baseline_EP": baseline_pattern,
                    "proposed_EP": proposed_pattern,
                    "EP_improvement": (
                        baseline_pattern
                        - proposed_pattern
                    ),
                    "relative_improvement_pct": (
                        100.0
                        * (
                            baseline_pattern
                            - proposed_pattern
                        )
                        / baseline_pattern
                        if baseline_pattern > 1e-12
                        else np.nan
                    ),
                })

    result_df = pd.DataFrame(results)

    result_df["holm_adjusted_p"] = (
        holm_adjust(
            result_df["wilcoxon_p"].to_numpy()
        )
    )

    coupling_df = pd.DataFrame(
        coupling_results
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    stats_path = (
        OUT / f"17_transmission_stats_{stamp}.csv"
    )

    coupling_path = (
        OUT / f"17_transmission_by_coupling_{stamp}.csv"
    )

    result_df.to_csv(
        stats_path,
        index=False,
    )

    coupling_df.to_csv(
        coupling_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print("TRANSMISSION STATISTICAL RESULTS")
    print("=" * 75)

    print(
        result_df.to_string(index=False),
        flush=True,
    )

    print("\n" + "=" * 75)
    print("TRANSMISSION RESULTS BY COUPLING")
    print("=" * 75)

    print(
        coupling_df.to_string(index=False),
        flush=True,
    )

    print("\nPositive improvement = lower EP.")
    print(
        "Bootstrap intervals are conditional on "
        "the tested scenario design."
    )

    print("\nSaved statistics:", stats_path)
    print("Saved coupling results:", coupling_path)

    print(
        "EXPERIMENT 17 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
