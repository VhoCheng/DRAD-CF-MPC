from pathlib import Path
from itertools import product
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "revision_experiments"
RESULTS = EXPS / "results"
OUT = RESULTS / "35_37_final_evidence"

METHODS = [
    "CDP",
    "Legacy-K5",
    "Confirm15-Robust",
]

COMPARISONS = [
    ("Legacy-K5", "CDP"),
    ("Confirm15-Robust", "CDP"),
    ("Confirm15-Robust", "Legacy-K5"),
]

SEED = 20261020
BOOT = 3000


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def holm_adjust(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    n = len(p)

    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(
            p[order] * (
                n - np.arange(n)
            )
        ),
    )

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    return adjusted


def wilson_ci(successes, n):
    if n <= 0:
        return np.nan, np.nan

    z = 1.95996398454
    p = successes / n

    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den

    radius = (
        z / den
        * np.sqrt(
            p*(1-p)/n
            + z*z/(4*n*n)
        )
    )

    return (
        max(0.0, center-radius),
        min(1.0, center+radius),
    )


def experiment35():
    source = (
        RESULTS
        / "34_redispatched_physical_v2_full"
        / "physical_per_run.csv"
    )

    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)

    if len(df) != 5760:
        raise RuntimeError(
            f"Exp34 expected 5760 rows, got {len(df)}"
        )

    if set(df["variant"]) != set(METHODS):
        raise RuntimeError(
            "Unexpected defense methods"
        )

    if set(df["case_size"]) != {
        14, 57, 118, 300
    }:
        raise RuntimeError(
            "Missing IEEE test systems"
        )

    if not (
        df["baseline_pf_converged"] == 1
    ).all():
        raise RuntimeError(
            "At least one baseline PF failed"
        )

    keys = [
        "case_size",
        "scenario_id",
        "pattern",
        "budget",
        "onset",
        "run_index",
        "variant",
    ]

    counts = df.groupby(keys).size()

    if len(counts) != 960 or not (
        counts == 6
    ).all():
        raise RuntimeError(
            "Expected 960 paired run blocks "
            "with six physical conditions each"
        )

    per_run = (
        df.groupby(keys, as_index=False)
        .agg(
            mean_loss=(
                "modeled_unserved_fraction",
                "mean",
            ),
            pf_failure_rate=(
                "attack_pf_converged",
                lambda x: 1.0 - x.mean(),
            ),
        )
    )

    per_run.to_csv(
        OUT / "35_paired_run_blocks.csv",
        index=False,
    )

    overview = (
        df.groupby(
            ["case_size", "variant"],
            as_index=False,
        )
        .agg(
            physical_rows=(
                "modeled_unserved_fraction",
                "size",
            ),
            mean_unserved=(
                "modeled_unserved_fraction",
                "mean",
            ),
            p95_unserved=(
                "modeled_unserved_fraction",
                lambda x: np.quantile(
                    x, 0.95
                ),
            ),
            attacked_pf_success=(
                "attack_pf_converged",
                "mean",
            ),
            mean_new_voltage_violations=(
                "new_voltage_violations",
                "mean",
            ),
        )
    )

    overview.to_csv(
        OUT / "35_physical_overview.csv",
        index=False,
    )

    rng = np.random.default_rng(SEED)
    rows = []

    for n in [14, 57, 118, 300]:
        sub = per_run[
            per_run["case_size"] == n
        ]

        scenarios = sorted(
            sub["scenario_id"].unique()
        )

        if len(scenarios) != 20:
            raise RuntimeError(
                f"IEEE{n}: expected 20 scenarios"
            )

        arr = []
        patterns = []

        for sid in scenarios:
            ss = sub[
                sub["scenario_id"] == sid
            ]

            p = ss["pattern"].unique()

            if len(p) != 1:
                raise RuntimeError(
                    "Multiple coupling patterns"
                )

            patterns.append(p[0])

            wide = ss.pivot(
                index="run_index",
                columns="variant",
                values="mean_loss",
            ).reindex(
                index=range(4),
                columns=METHODS,
            )

            if wide.isna().any().any():
                raise RuntimeError(
                    f"Missing paired runs: {sid}"
                )

            arr.append(
                wide.to_numpy(dtype=float)
            )

        arr = np.stack(arr)

        if arr.shape != (20, 4, 3):
            raise RuntimeError(
                f"Unexpected tensor: {arr.shape}"
            )

        pattern_arr = np.asarray(patterns)

        strata = [
            np.where(
                pattern_arr == p
            )[0]
            for p in sorted(set(patterns))
        ]

        if (
            len(strata) != 5
            or any(len(x) != 4 for x in strata)
        ):
            raise RuntimeError(
                "Coupling strata are incomplete"
            )

        # Coupling-stratified scenario bootstrap.
        # Four MC runs are resampled within each
        # scenario. The six physical conditions
        # are NOT treated as independent trials.
        sampled_s = np.concatenate([
            rng.choice(
                x,
                size=(BOOT, len(x)),
                replace=True,
            )
            for x in strata
        ], axis=1)

        sampled_r = rng.integers(
            0, 4,
            size=(BOOT, 20, 4),
        )

        sampled = arr[
            sampled_s[:, :, None],
            sampled_r,
            :
        ]

        boot_means = sampled.mean(
            axis=(1, 2)
        )

        scenario_means = arr.mean(
            axis=1
        )

        index = {
            method: i
            for i, method in enumerate(
                METHODS
            )
        }

        for method, baseline in COMPARISONS:
            a = index[method]
            b = index[baseline]

            # Positive gain means lower loss.
            differences = (
                scenario_means[:, b]
                - scenario_means[:, a]
            )

            boot_differences = (
                boot_means[:, b]
                - boot_means[:, a]
            )

            low, high = np.quantile(
                boot_differences,
                [0.025, 0.975],
            )

            if np.allclose(
                differences,
                0,
                atol=1e-14,
                rtol=0,
            ):
                pvalue = 1.0
            else:
                pvalue = float(
                    wilcoxon(
                        differences,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )

            mean_base = float(
                scenario_means[:, b].mean()
            )

            mean_gain = float(
                differences.mean()
            )

            rows.append({
                "case_size": n,
                "comparison": (
                    f"{method} vs {baseline}"
                ),
                "mean_baseline_loss": mean_base,
                "mean_method_loss": float(
                    scenario_means[:, a].mean()
                ),
                "absolute_gain": mean_gain,
                "relative_gain_pct": (
                    100.0 * mean_gain / mean_base
                    if mean_base > 1e-12
                    else np.nan
                ),
                "CI_low": float(low),
                "CI_high": float(high),
                "wilcoxon_p": pvalue,
                "better_scenarios": int(
                    np.sum(
                        differences > 1e-12
                    )
                ),
                "worse_scenarios": int(
                    np.sum(
                        differences < -1e-12
                    )
                ),
                "tied_scenarios": int(
                    np.sum(
                        np.abs(differences)
                        <= 1e-12
                    )
                ),
            })

    result = pd.DataFrame(rows)

    result["holm_p_12_tests"] = (
        holm_adjust(
            result["wilcoxon_p"].to_numpy()
        )
    )

    result.to_csv(
        OUT / "35_physical_paired_statistics.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 35 - PHYSICAL PAIRED STATISTICS")
    print("=" * 75)

    print(
        result.to_string(index=False),
        flush=True,
    )

    return source


def experiment36():
    source = (
        RESULTS
        / "23_calibration_full"
        / "per_decision.csv"
    )

    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)

    if len(df) != 320:
        raise RuntimeError(
            f"Expected 320 calibration decisions, "
            f"got {len(df)}"
        )

    required = {
        "method",
        "budget",
        "onset",
        "switched",
        "predicted_PAS_gain",
        "validated_PAS_gain",
        "actual_PAS_gain",
        "validation_negative_gain",
        "actual_negative_gain",
    }

    if not required.issubset(df.columns):
        raise RuntimeError(
            "Calibration CSV schema incomplete"
        )

    summaries = []

    for method in ["Original", "Balanced"]:
        sub = df[df["method"] == method]
        selected = sub[
            sub["switched"] == 1
        ]

        if len(sub) != 160:
            raise RuntimeError(
                f"Unexpected decision count: {method}"
            )

        n = len(selected)

        negatives = int(
            selected[
                "validation_negative_gain"
            ].sum()
        )

        ci_low, ci_high = wilson_ci(
            negatives, n
        )

        summaries.append({
            "method": method,
            "total_decisions": len(sub),
            "switched": n,
            "switch_rate": float(
                sub["switched"].mean()
            ),
            "K5_predicted_gain": float(
                selected[
                    "predicted_PAS_gain"
                ].mean()
            ),
            "K20_independent_gain": float(
                selected[
                    "validated_PAS_gain"
                ].mean()
            ),
            "historical_actual_gain": float(
                selected[
                    "actual_PAS_gain"
                ].mean()
            ),
            "K20_negative_count": negatives,
            "K20_negative_rate": (
                negatives / n
                if n else np.nan
            ),
            "K20_negative_CI_low": ci_low,
            "K20_negative_CI_high": ci_high,
            "actual_negative_rate": float(
                selected[
                    "actual_negative_gain"
                ].mean()
            ),
        })

    overview = pd.DataFrame(summaries)

    selected = df[df["switched"] == 1]

    by_condition = (
        selected.groupby(
            ["method", "budget", "onset"],
            as_index=False,
        )
        .agg(
            switches=("switched", "size"),
            mean_K5_gain=(
                "predicted_PAS_gain",
                "mean",
            ),
            mean_K20_gain=(
                "validated_PAS_gain",
                "mean",
            ),
            K20_negative_rate=(
                "validation_negative_gain",
                "mean",
            ),
            actual_negative_rate=(
                "actual_negative_gain",
                "mean",
            ),
        )
    )

    overview.to_csv(
        OUT / "36_guard_calibration_summary.csv",
        index=False,
    )

    by_condition.to_csv(
        OUT / "36_guard_by_condition.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 36 - SAFEGUARD CALIBRATION")
    print("=" * 75)

    print(
        overview.to_string(index=False),
        flush=True,
    )

    print("\nCONDITIONAL FAILURE ANALYSIS")

    print(
        by_condition.to_string(index=False),
        flush=True,
    )

    return source


def load_experiment26():
    path = (
        EXPS / "26_multiscale_defense.py"
    )

    spec = importlib.util.spec_from_file_location(
        "tnse_latency_experiment26",
        path,
    )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    return mod, path


def experiment37():
    e26, source = load_experiment26()

    output = OUT / "37_serial_timing.csv"
    manifest_path = OUT / "37_timing_manifest.json"

    tasks = [
        (
            idx,
            n,
            pattern,
            0.50,
            fraction,
            onset,
            1,
            20261129,
        )
        for idx, (
            n, pattern, fraction, onset
        ) in enumerate(
            product(
                [14, 57, 118, 300],
                ["DDAC", "RC"],
                [0.10, 0.30],
                [120, 240],
            ),
            start=1,
        )
    ]

    manifest = {
        "experiment": "Serial phase timing",
        "tasks": tasks,
        "source_26_sha256": digest(source),
        "notes": (
            "Single Python process; one Monte Carlo "
            "replicate per configuration; stage timings "
            "are not network transport latency."
        ),
    }

    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if previous != manifest:
            raise RuntimeError(
                "Timing protocol changed"
            )
    else:
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    if output.exists():
        old = pd.read_csv(output)
        records = old.to_dict("records")
        complete = set(
            old["task_id"].astype(int)
        )
    else:
        records = []
        complete = set()

    print("\n" + "=" * 75)
    print("EXPERIMENT 37 - SERIAL TIMING")
    print("=" * 75)

    print(
        "Total tasks:", len(tasks),
        "completed:", len(complete),
        flush=True,
    )

    for task in tasks:
        task_id = task[0]

        if task_id in complete:
            continue

        began = time.perf_counter()

        result = e26.run_one(task)

        wall = (
            time.perf_counter() - began
        )

        summary = pd.DataFrame(
            result["summary"]
        )

        if set(summary["variant"]) != set(METHODS):
            raise RuntimeError(
                "Timing methods mismatch"
            )

        stage1 = float(
            summary[
                "mean_stage1_seconds"
            ].iloc[0]
        )

        stage2 = float(
            summary[
                "mean_stage2_seconds"
            ].iloc[0]
        )

        record = {
            "task_id": task_id,
            "case_size": task[1],
            "pattern": task[2],
            "budget_fraction": task[4],
            "onset": task[5],
            "stage1_seconds": stage1,
            "stage2_seconds": stage2,
            "three_method_total_wall_seconds": wall,
        }

        records.append(record)

        pd.DataFrame(records).to_csv(
            output,
            index=False,
        )

        print(
            f"[{len(records)}/{len(tasks)}] "
            f"IEEE{task[1]} {task[2]} "
            f"stage1={stage1:.2f}s "
            f"stage2={stage2:.2f}s "
            f"wall={wall:.2f}s",
            flush=True,
        )

    data = pd.read_csv(output)

    if len(data) != len(tasks):
        raise RuntimeError(
            "Timing results incomplete"
        )

    report = (
        data.groupby(
            "case_size",
            as_index=False,
        )
        .agg(
            mean_stage1=(
                "stage1_seconds", "mean"
            ),
            p95_stage1=(
                "stage1_seconds",
                lambda x: np.quantile(x, 0.95)
            ),
            mean_stage2=(
                "stage2_seconds", "mean"
            ),
            p95_stage2=(
                "stage2_seconds",
                lambda x: np.quantile(x, 0.95)
            ),
            mean_three_method_wall=(
                "three_method_total_wall_seconds",
                "mean"
            ),
        )
    )

    report.to_csv(
        OUT / "37_serial_timing_summary.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 37 - SERIAL COST RESULTS")
    print("=" * 75)

    print(
        report.to_string(index=False),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["audit", "latency", "all"],
        default="audit",
    )

    args = parser.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode in ("audit", "all"):
        physical = experiment35()
        calibration = experiment36()

        (OUT / "source_manifest.json").write_text(
            json.dumps({
                "physical_sha256": digest(
                    physical
                ),
                "calibration_sha256": digest(
                    calibration
                ),
                "driver_sha256": digest(
                    Path(__file__).resolve()
                ),
            }, indent=2),
            encoding="utf-8",
        )

        print("\nEXPERIMENTS 35-36 COMPLETED SUCCESSFULLY")

    if args.mode in ("latency", "all"):
        experiment37()
        print(
            "\nEXPERIMENT 37 COMPLETED SUCCESSFULLY"
        )


if __name__ == "__main__":
    main()
