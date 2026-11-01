from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"

BOOTSTRAPS = 5000
BOOT_SEED = 20261018

CONDITIONS = [
    (m, d)
    for m in [0.0, 0.2, 0.4]
    for d in [0, 15, 30]
]


def load_data():
    sources = sorted(
        OUT.glob("08_observation_full_*_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )

    if not sources:
        raise FileNotFoundError(
            "Experiment 08 full summary not found."
        )

    summary_path = sources[-1]
    prefix = summary_path.name.removesuffix("_summary.csv")
    detail_path = OUT / f"{prefix}_per_run.csv"

    if not detail_path.exists():
        raise FileNotFoundError(detail_path)

    summary = pd.read_csv(summary_path)
    details = pd.read_csv(detail_path)

    return summary, details, summary_path, detail_path


def validate(summary, details):
    if len(summary) != 360:
        raise RuntimeError(
            f"Expected 360 summary rows, got {len(summary)}"
        )

    if len(details) != 3600:
        raise RuntimeError(
            f"Expected 3600 per-run rows, got {len(details)}"
        )

    if summary["scenario_id"].nunique() != 40:
        raise RuntimeError("Expected 40 scenarios.")

    skey = [
        "scenario_id", "missing_rate", "delay_steps"
    ]

    rkey = skey + ["run_index"]

    if summary.duplicated(skey).any():
        raise RuntimeError("Duplicate summary rows.")

    if details.duplicated(rkey).any():
        raise RuntimeError("Duplicate per-run rows.")

    if summary.isna().any().any():
        raise RuntimeError("Summary contains missing data.")

    counts = details.groupby(skey)["run_index"].nunique()

    if not (counts == 10).all():
        raise RuntimeError("Some settings lack 10 MC runs.")

    reconstructed = (
        details.groupby(skey, as_index=False)
        .agg(
            pas=("DRAD_PAS", "mean"),
            ps=("DRAD_PS", "mean"),
        )
    )

    reconstructed["ep"] = (
        reconstructed["pas"] * reconstructed["ps"]
    )

    check = summary.merge(
        reconstructed,
        on=skey,
        validate="one_to_one",
    )

    max_error = float(
        np.max(np.abs(check["DRAD_EP"] - check["ep"]))
    )

    if max_error > 1e-10:
        raise RuntimeError(
            f"EP reconstruction failed: {max_error}"
        )

    print("DATA VALIDATION PASSED")
    print("Settings:", len(summary))
    print("Per-run records:", len(details))
    print("EP reconstruction error:", max_error)


def main():
    print("=" * 72)
    print("TNSE EXPERIMENT 09")
    print("PAIRED OBSERVATION ROBUSTNESS AUDIT")
    print("=" * 72)

    summary, details, source_s, source_d = load_data()

    validate(summary, details)

    clean_s = summary[
        (summary["missing_rate"] == 0.0)
        & (summary["delay_steps"] == 0)
    ].copy()

    clean_d = details[
        (details["missing_rate"] == 0.0)
        & (details["delay_steps"] == 0)
    ].copy()

    if len(clean_s) != 40 or len(clean_d) != 400:
        raise RuntimeError("Clean baseline incomplete.")

    clean_s = clean_s[
        [
            "scenario_id",
            "DRAD_EP",
            "DRAD_PAS",
            "CDP_EP",
            "switch_rate",
        ]
    ].rename(columns={
        "DRAD_EP": "clean_EP",
        "DRAD_PAS": "clean_PAS",
        "CDP_EP": "clean_CDP_EP",
        "switch_rate": "clean_switch_rate",
    })

    clean_d = clean_d[
        [
            "scenario_id",
            "run_index",
            "DRAD_PAS",
            "selected_policy",
            "switched",
        ]
    ].rename(columns={
        "DRAD_PAS": "clean_run_PAS",
        "selected_policy": "clean_policy",
        "switched": "clean_switched",
    })

    comparison = summary.merge(
        clean_s,
        on="scenario_id",
        validate="many_to_one",
    )

    paired_runs = details.merge(
        clean_d,
        on=["scenario_id", "run_index"],
        validate="many_to_one",
    )

    if len(comparison) != 360 or len(paired_runs) != 3600:
        raise RuntimeError("Paired merge incomplete.")

    if not np.allclose(
        comparison["CDP_EP"],
        comparison["clean_CDP_EP"],
        rtol=0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "CDP baseline changed across observation conditions."
        )

    comparison["delta_EP"] = (
        comparison["DRAD_EP"] - comparison["clean_EP"]
    )

    comparison["delta_PAS"] = (
        comparison["DRAD_PAS"] - comparison["clean_PAS"]
    )

    comparison["delta_switch_rate"] = (
        comparison["switch_rate"]
        - comparison["clean_switch_rate"]
    )

    paired_runs["policy_changed"] = (
        paired_runs["selected_policy"]
        != paired_runs["clean_policy"]
    )

    paired_runs["PAS_worsened_vs_clean"] = (
        paired_runs["DRAD_PAS"]
        > paired_runs["clean_run_PAS"] + 1e-12
    )

    paired_runs["delta_run_PAS"] = (
        paired_runs["DRAD_PAS"]
        - paired_runs["clean_run_PAS"]
    )

    rng = np.random.default_rng(BOOT_SEED)
    aggregate_rows = []

    for missing, delay in CONDITIONS:
        group = comparison[
            np.isclose(comparison["missing_rate"], missing)
            & (comparison["delay_steps"] == delay)
        ].sort_values("scenario_id")

        run_group = paired_runs[
            np.isclose(paired_runs["missing_rate"], missing)
            & (paired_runs["delay_steps"] == delay)
        ]

        if len(group) != 40 or len(run_group) != 400:
            raise RuntimeError(
                f"Incomplete setting: {missing}, {delay}"
            )

        delta = group["delta_EP"].to_numpy(dtype=float)
        patterns = group["pattern"].to_numpy()

        sampled_parts = []

        for pattern in sorted(set(patterns)):
            indices = np.where(patterns == pattern)[0]

            if len(indices) != 8:
                raise RuntimeError(
                    "Unexpected coupling group size."
                )

            sampled_parts.append(
                rng.choice(
                    indices,
                    size=(BOOTSTRAPS, len(indices)),
                    replace=True,
                )
            )

        sampled = np.concatenate(sampled_parts, axis=1)

        bootstrap_means = delta[sampled].mean(axis=1)

        ci_low, ci_high = np.quantile(
            bootstrap_means,
            [0.025, 0.975],
        )

        aggregate_rows.append({
            "missing_rate": missing,
            "delay_steps": delay,
            "scenarios": 40,
            "mean_clean_EP": float(group["clean_EP"].mean()),
            "mean_perturbed_EP": float(
                group["DRAD_EP"].mean()
            ),
            "mean_delta_EP": float(delta.mean()),
            "bootstrap_CI_low": float(ci_low),
            "bootstrap_CI_high": float(ci_high),
            "mean_delta_PAS": float(
                group["delta_PAS"].mean()
            ),
            "mean_switch_rate": float(
                group["switch_rate"].mean()
            ),
            "policy_change_rate": float(
                run_group["policy_changed"].mean()
            ),
            "PAS_worsening_rate_vs_clean": float(
                run_group["PAS_worsened_vs_clean"].mean()
            ),
            "mean_omitted_states": float(
                run_group["omitted_states"].mean()
            ),
            "mean_true_active": float(
                run_group["true_active"].mean()
            ),
            "mean_observed_active": float(
                run_group["observed_active"].mean()
            ),
        })

    result = pd.DataFrame(aggregate_rows)

    zero = result[
        (result["missing_rate"] == 0)
        & (result["delay_steps"] == 0)
    ].iloc[0]

    if (
        abs(zero["mean_delta_EP"]) > 1e-12
        or zero["policy_change_rate"] > 0
        or zero["PAS_worsening_rate_vs_clean"] > 0
    ):
        raise RuntimeError(
            "Zero-perturbation validation failed."
        )

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    base = f"09_observation_audit_{stamp}"

    result_path = OUT / f"{base}_aggregate.csv"
    scenario_path = OUT / f"{base}_by_scenario.csv"
    run_path = OUT / f"{base}_per_run.csv"
    config_path = OUT / f"{base}_config.json"

    result.to_csv(result_path, index=False)
    comparison.to_csv(scenario_path, index=False)
    paired_runs.to_csv(run_path, index=False)

    config_path.write_text(
        json.dumps({
            "experiment": "Paired observation audit",
            "bootstrap_iterations": BOOTSTRAPS,
            "bootstrap_seed": BOOT_SEED,
            "resampling": (
                "Stratified scenario bootstrap, "
                "8 scenarios within each coupling pattern"
            ),
            "source_summary": str(source_s),
            "source_per_run": str(source_d),
            "reference": "Same scenario and seed at 0% missing, 0 delay",
        }, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("PAIRED OBSERVATION ROBUSTNESS RESULTS")
    print("=" * 72)

    cols = [
        "missing_rate",
        "delay_steps",
        "mean_clean_EP",
        "mean_perturbed_EP",
        "mean_delta_EP",
        "bootstrap_CI_low",
        "bootstrap_CI_high",
        "policy_change_rate",
        "PAS_worsening_rate_vs_clean",
        "mean_omitted_states",
    ]

    print(result[cols].to_string(index=False))

    print("\nPositive Delta EP = deterioration.")
    print("Negative Delta EP = improvement.")
    print("Bootstrap intervals are conditional on tested scenarios.")

    print("\nAggregate:", result_path)
    print("By scenario:", scenario_path)
    print("Per-run:", run_path)
    print("Config:", config_path)
    print("EXPERIMENT 09 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
