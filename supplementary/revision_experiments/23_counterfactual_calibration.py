from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"
SRC = ROOT / "src"

os.chdir(ROOT)
sys.path.insert(0, str(SRC))

import run_experiments_core as core
import drad_mpc as mpc


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "revision_experiments" / path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e14 = import_script(
    "exp14_calibration",
    "14_parallel_replication.py",
)

e19 = import_script(
    "exp19_calibration",
    "19_dynamic_risk_weights.py",
)

PILOT = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

FULL = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.25, 1.00],
    [20, 80],
    [120, 270],
))

K_VALIDATION = 20
METHODS = ["Original", "Balanced"]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def nodes_key(nodes):
    return tuple(sorted(map(int, nodes)))


def build_balanced_candidates(graph, scenario, seed):
    pattern, info, budget, onset = scenario

    preview = core.get_preview_state(
        Gc=graph,
        onset_time=onset,
        attack_information_ratio=info,
        infection_rate=0.01,
        activation_rate=0.15,
        detection_rate=1e-5,
        seed=seed,
    )

    balanced_nodes = e19.weighted_drad_nodes(
        pattern,
        budget,
        info,
        preview["active_infectious_nodes"],
        preview["exposed_nodes"],
        seed,
        e19.PROFILES["Balanced"],
    )

    original_function = (
        mpc.candidate_drad_score_nodes
    )

    try:
        def replacement(*args, **kwargs):
            return list(balanced_nodes)

        mpc.candidate_drad_score_nodes = replacement

        candidates = mpc.generate_candidate_policies(
            pattern=pattern,
            defense_budget=budget,
            attack_information_ratio=info,
            active_nodes=preview[
                "active_infectious_nodes"
            ],
            exposed_nodes=preview["exposed_nodes"],
            table_dir="outputs/tables",
            seed=seed,
        )
    finally:
        mpc.candidate_drad_score_nodes = (
            original_function
        )

    if "CDP" not in candidates:
        candidates["CDP"] = (
            mpc.select_protected_nodes(
                strategy="CDP",
                defense_budget=budget,
                pattern=pattern,
                seed=seed,
                table_dir="outputs/tables",
            )
        )

    return candidates


def task_worker(task):
    sid, scenario, n_runs, base_seed, mode = task

    os.chdir(ROOT)

    graph = core.build_comm_graph(
        n_nodes=118,
        seed=42,
    )

    reference_path = (
        OUT
        / f"21_balanced_holdout_{mode}"
        / "per_run.csv"
    )

    reference = pd.read_csv(reference_path)

    pattern, info, budget, onset = scenario

    rows = []
    checks = 0

    for r in range(n_runs):
        seed = (
            base_seed + r
            + onset * 1000
            + budget * 17
        )

        # Reproduce the original K5 stage.
        original_candidates, metrics_original = (
            e14.create_metrics5(
                graph, scenario, seed
            )
        )

        # Reproduce the Balanced K5 stage.
        balanced_candidates = (
            build_balanced_candidates(
                graph, scenario, seed
            )
        )

        # Share K5 CF outcomes whenever node sets match.
        metric_cache = {
            nodes_key(original_candidates[name]): metric
            for name, metric in metrics_original.items()
        }

        k5_seeds = [
            seed + 777000 + 10007 * k
            + mpc.stable_int_hash(
                f"{pattern}-{onset}-{budget}",
                100000,
            )
            for k in range(5)
        ]

        metrics_balanced = {}

        for name, nodes in balanced_candidates.items():
            key = nodes_key(nodes)

            if key not in metric_cache:
                metric_cache[key] = (
                    mpc.evaluate_candidate_with_common_random_numbers(
                        candidate_nodes=list(key),
                        common_seeds=k5_seeds,
                        Gc=graph,
                        pattern=pattern,
                        attack_information_ratio=info,
                        defense_budget=budget,
                        onset_time=onset,
                        infection_rate=0.01,
                        activation_rate=0.15,
                        detection_rate=1e-5,
                    )
                )

            metrics_balanced[name] = metric_cache[key]

        # Independent seeds never used in K5 selection.
        independent_seeds = [
            seed + 20_000_000 + 10007 * k
            + mpc.stable_int_hash(
                f"calibration-{pattern}-{onset}-{budget}",
                100000,
            )
            for k in range(K_VALIDATION)
        ]

        validation_cache = {}

        def independent_metric(nodes):
            key = nodes_key(nodes)

            if key not in validation_cache:
                validation_cache[key] = (
                    mpc.evaluate_candidate_with_common_random_numbers(
                        candidate_nodes=list(key),
                        common_seeds=independent_seeds,
                        Gc=graph,
                        pattern=pattern,
                        attack_information_ratio=info,
                        defense_budget=budget,
                        onset_time=onset,
                        infection_rate=0.01,
                        activation_rate=0.15,
                        detection_rate=1e-5,
                    )
                )

            return validation_cache[key]

        configurations = {
            "Original": (
                original_candidates,
                metrics_original,
            ),
            "Balanced": (
                balanced_candidates,
                metrics_balanced,
            ),
        }

        stored_rows = reference[
            (reference["scenario_id"] == sid)
            & (reference["run_index"] == r)
        ]

        def stored(name):
            matched = stored_rows[
                stored_rows["variant"] == name
            ]

            if len(matched) != 1:
                raise RuntimeError(
                    f"Historical record missing: "
                    f"{sid}, {r}, {name}"
                )

            result = matched.iloc[0]

            if int(result["seed"]) != seed:
                raise RuntimeError(
                    "Historical random seed mismatch"
                )

            return result

        historical_cdp = stored("CDP")

        for method, (candidates, metrics) in (
            configurations.items()
        ):
            selected, _ = (
                e14.e11.helper.choose_policy(
                    metrics,
                    eta=0.0,
                    kappa=0.5,
                )
            )

            if selected not in candidates:
                raise RuntimeError(
                    "Selected candidate missing"
                )

            historical = stored(method)

            if selected != historical["selected_policy"]:
                raise RuntimeError(
                    "HISTORICAL POLICY REPRODUCTION FAILED: "
                    f"{sid}, {r}, {method}"
                )

            selected_nodes = nodes_key(
                candidates[selected]
            )

            stored_nodes = nodes_key(
                int(x)
                for x in str(
                    historical["protected_nodes"]
                ).split("|")
                if x.strip()
            )

            if selected_nodes != stored_nodes:
                raise RuntimeError(
                    "HISTORICAL NODE REPRODUCTION FAILED: "
                    f"{sid}, {r}, {method}"
                )

            checks += 1

            # K5 estimates used for actual policy choice.
            predicted_gain = (
                metrics["CDP"]["pred_PAS"]
                - metrics[selected]["pred_PAS"]
            )

            predicted_ep_gain = (
                metrics["CDP"]["pred_EP"]
                - metrics[selected]["pred_EP"]
            )

            # Independent validation of both policies,
            # with common random numbers.
            independent_cdp = independent_metric(
                candidates["CDP"]
            )

            independent_selected = independent_metric(
                candidates[selected]
            )

            validated_gain = (
                independent_cdp["pred_PAS"]
                - independent_selected["pred_PAS"]
            )

            validated_ep_gain = (
                independent_cdp["pred_EP"]
                - independent_selected["pred_EP"]
            )

            # Actual single-run outcomes, already preserved
            # in the frozen Experiment 21.
            actual_gain = (
                float(historical_cdp["PAS"])
                - float(historical["PAS"])
            )

            switched = selected != "CDP"

            rows.append({
                "scenario_id": sid,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "run_index": r,
                "seed": seed,
                "method": method,
                "selected_policy": selected,
                "switched": int(switched),
                "K_selection": 5,
                "K_validation": K_VALIDATION,
                "predicted_PAS_gain": float(
                    predicted_gain
                ),
                "validated_PAS_gain": float(
                    validated_gain
                ),
                "actual_PAS_gain": float(
                    actual_gain
                ),
                "predicted_EP_gain": float(
                    predicted_ep_gain
                ),
                "validated_EP_gain": float(
                    validated_ep_gain
                ),
                "prediction_error_PAS": float(
                    predicted_gain - validated_gain
                ),
                "validation_negative_gain": int(
                    switched and validated_gain < -1e-12
                ),
                "actual_negative_gain": int(
                    switched and actual_gain < -1e-12
                ),
                "nodes": "|".join(
                    map(str, selected_nodes)
                ),
            })

    return {
        "scenario_id": sid,
        "checks": checks,
        "records": rows,
    }


def save_json(path, data):
    tmp = path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    tmp.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    e14.e11.helper.check_original()

    if args.mode == "pilot":
        scenarios = PILOT
        n_runs = 3
        base_seed = 42
        workers = 2
    else:
        scenarios = FULL
        n_runs = 4
        base_seed = 20261219
        workers = max(
            1, min(args.workers, 12)
        )

    run_dir = (
        OUT / f"23_calibration_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(
        parents=True, exist_ok=True
    )

    reference_path = (
        OUT / f"21_balanced_holdout_{args.mode}"
        / "per_run.csv"
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            reference_path
        )

    manifest = {
        "experiment": "K5 independent CF calibration",
        "mode": args.mode,
        "scenarios": scenarios,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "K_selection": 5,
        "K_validation": K_VALIDATION,
        "methods": METHODS,
        "reference_sha256": sha(reference_path),
        "driver_sha256": sha(
            Path(__file__).resolve()
        ),
        "helpers_sha256": {
            name: sha(
                ROOT / "revision_experiments" / name
            )
            for name in [
                "10_cf_optimization_pilot.py",
                "11_crossfit_confirmation.py",
                "14_parallel_replication.py",
                "19_dynamic_risk_weights.py",
            ]
        },
        "source_sha256": {
            name: sha(SRC / name)
            for name in [
                "drad_mpc.py",
                "run_experiments_core.py",
                "seid_malware.py",
            ]
        },
    }

    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if previous != manifest:
            raise RuntimeError(
                "Protocol mismatch: refusing "
                "to mix checkpoints"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    tasks = [
        (
            i, scenario, n_runs,
            base_seed, args.mode
        )
        for i, scenario in enumerate(
            scenarios, 1
        )
    ]

    pending = [
        task for task in tasks
        if not (
            checkpoints
            / f"scenario_{task[0]:03d}.json"
        ).exists()
    ]

    print("=" * 74, flush=True)
    print("TNSE EXPERIMENT 23", flush=True)
    print("INDEPENDENT COUNTERFACTUAL CALIBRATION", flush=True)
    print("=" * 74, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(tasks), flush=True)
    print("Runs per scenario:", n_runs, flush=True)
    print("Workers:", workers, flush=True)
    print("Independent rollout count:", K_VALIDATION, flush=True)
    print("Pending:", len(pending), flush=True)

    started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:
        futures = {
            pool.submit(task_worker, task): task
            for task in pending
        }

        for future in as_completed(futures):
            task = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Calibration task failed: {task}"
                ) from exc

            path = (
                checkpoints
                / f"scenario_{task[0]:03d}.json"
            )

            save_json(path, result)

            print(
                f"[saved {task[0]:02d}/{len(tasks)}] "
                f"{task[1]} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    records = []
    total_checks = 0

    for task in tasks:
        path = (
            checkpoints
            / f"scenario_{task[0]:03d}.json"
        )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if data["scenario_id"] != task[0]:
            raise RuntimeError(
                "Checkpoint identity mismatch"
            )

        records.extend(data["records"])
        total_checks += data["checks"]

    expected = (
        len(tasks) * n_runs * len(METHODS)
    )

    if len(records) != expected:
        raise RuntimeError(
            f"Incomplete records: "
            f"{len(records)} != {expected}"
        )

    if total_checks != expected:
        raise RuntimeError(
            "Historical reproduction incomplete"
        )

    df = pd.DataFrame(records)

    if df.duplicated([
        "scenario_id",
        "run_index",
        "method",
    ]).any():
        raise RuntimeError(
            "Duplicate calibration records"
        )

    print(
        f"HISTORICAL REPRODUCTION PASSED: "
        f"{total_checks}/{expected} decisions and nodes",
        flush=True,
    )

    df.to_csv(
        run_dir / "per_decision.csv",
        index=False,
    )

    summary_rows = []

    for method in METHODS:
        all_rows = df[df["method"] == method]
        switched = all_rows[
            all_rows["switched"] == 1
        ]

        summary_rows.append({
            "method": method,
            "decisions": len(all_rows),
            "switched_decisions": len(switched),
            "switch_rate": float(
                all_rows["switched"].mean()
            ),
            "mean_predicted_gain_on_switch": (
                float(
                    switched["predicted_PAS_gain"].mean()
                ) if len(switched) else np.nan
            ),
            "mean_independent_gain_on_switch": (
                float(
                    switched["validated_PAS_gain"].mean()
                ) if len(switched) else np.nan
            ),
            "mean_actual_gain_on_switch": (
                float(
                    switched["actual_PAS_gain"].mean()
                ) if len(switched) else np.nan
            ),
            "independent_negative_gain_rate": (
                float(
                    switched["validation_negative_gain"].mean()
                ) if len(switched) else np.nan
            ),
            "actual_negative_gain_rate": (
                float(
                    switched["actual_negative_gain"].mean()
                ) if len(switched) else np.nan
            ),
            "mean_CF_prediction_error_on_switch": (
                float(
                    switched["prediction_error_PAS"].mean()
                ) if len(switched) else np.nan
            ),
            "mean_absolute_CF_error_on_switch": (
                float(
                    switched["prediction_error_PAS"].abs().mean()
                ) if len(switched) else np.nan
            ),
        })

    aggregate = pd.DataFrame(
        summary_rows
    )

    aggregate.to_csv(
        run_dir / "aggregate.csv",
        index=False,
    )

    # Analyze only actual switches for failure diagnostics.
    switched = df[df["switched"] == 1]

    if len(switched):
        by_condition = (
            switched.groupby(
                ["method", "budget", "onset"],
                as_index=False,
            )
            .agg(
                n_switches=(
                    "switched", "size"
                ),
                independent_negative_rate=(
                    "validation_negative_gain", "mean"
                ),
                actual_negative_rate=(
                    "actual_negative_gain", "mean"
                ),
                mean_predicted_gain=(
                    "predicted_PAS_gain", "mean"
                ),
                mean_independent_gain=(
                    "validated_PAS_gain", "mean"
                ),
            )
        )

        by_condition.to_csv(
            run_dir / "by_condition.csv",
            index=False,
        )
    else:
        by_condition = pd.DataFrame()

    print("\n" + "=" * 74)
    print("COUNTERFACTUAL CALIBRATION RESULTS")
    print("=" * 74)
    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\n" + "=" * 74)
    print("CALIBRATION BY BUDGET AND ATTACK ONSET")
    print("=" * 74)

    if len(by_condition):
        print(
            by_condition.to_string(index=False),
            flush=True,
        )
    else:
        print("No non-CDP switches.")

    print("\nOutput directory:", run_dir)
    print(
        "Runtime:",
        round(time.perf_counter()-started, 2),
        "seconds",
    )

    print(
        "EXPERIMENT 23 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
