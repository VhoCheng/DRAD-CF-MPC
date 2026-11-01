from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
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
SRC = ROOT / "src"
OUT = ROOT / "revision_experiments/results"

sys.path.insert(0, str(SRC))

import run_experiments_core as core
import drad_mpc as mpc

HELPER = (
    ROOT / "revision_experiments"
    / "14_parallel_replication.py"
)

spec = importlib.util.spec_from_file_location(
    "validated_experiment14",
    HELPER,
)

e14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e14)

PILOT = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

FULL = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.50],
    [40, 80],
    [90, 180],
))

# Coefficients are ordered as:
# predicted EP, PAS, normalized RT, RI.
WEIGHTS = {
    "Original": (0.68, 0.24, 0.06, -0.02),
    "EP-Focused": (0.85, 0.10, 0.04, -0.01),
    "PAS-Focused": (0.30, 0.60, 0.08, -0.02),
    "Recovery-Focused": (0.55, 0.20, 0.20, -0.05),
    "Resilience-Focused": (0.55, 0.20, 0.15, -0.10),
}

VARIANTS = ["CDP"] + list(WEIGHTS)


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def weighted_objective(metrics, weights):
    w_ep, w_pas, w_rt, w_ri = weights

    return (
        w_ep * float(metrics["pred_EP"])
        + w_pas * float(metrics["pred_PAS"])
        + w_rt * float(metrics["pred_RT"]) / 21.0
        + w_ri * float(metrics["pred_RI"])
    )


def select_policy(metrics, weights):
    scores = {
        name: weighted_objective(item, weights)
        for name, item in metrics.items()
    }

    if "CDP" not in scores:
        raise RuntimeError("Missing CDP candidate")

    best = min(scores, key=scores.get)

    if best == "CDP":
        return "CDP"

    cdp_score = float(scores["CDP"])
    best_score = float(scores[best])

    objective_improvement = (
        (cdp_score - best_score)
        / (abs(cdp_score) + 1e-9)
    )

    mean_diff, stderr = mpc.paired_improvement(
        metrics[best],
        metrics["CDP"],
    )

    conservative_gain = (
        mean_diff - 0.50 * stderr
    )

    if (
        objective_improvement >= 0.10
        and conservative_gain > 0.0
    ):
        return best

    return "CDP"


def worker(task):
    sid, scenario, n_runs, base_seed = task

    os.chdir(ROOT)

    pattern, info, budget, onset = scenario

    graph = core.build_comm_graph(
        n_nodes=118,
        seed=42,
    )

    details = []
    summary = []

    outcomes = {
        name: []
        for name in VARIANTS
    }

    policies = {
        name: []
        for name in VARIANTS
    }

    for r in range(n_runs):
        seed = (
            base_seed
            + r
            + onset * 1000
            + budget * 17
        )

        # One common CF sample set for every weight profile.
        candidates, metrics5 = e14.create_metrics5(
            graph,
            scenario,
            seed,
        )

        selections = {"CDP": "CDP"}

        for name, weights in WEIGHTS.items():
            selections[name] = select_policy(
                metrics5, weights
            )

        # The default-weight decision must reproduce
        # the validated original Legacy-K5 selector.
        expected, _ = e14.e11.helper.choose_policy(
            metrics5,
            eta=0.0,
            kappa=0.5,
        )

        if selections["Original"] != expected:
            raise RuntimeError(
                "DEFAULT OBJECTIVE SELECTOR MISMATCH: "
                f"scenario={sid}, run={r}, "
                f"expected={expected}, "
                f"actual={selections['Original']}"
            )

        actual_cache = {}

        for variant, selected in selections.items():
            nodes = tuple(
                sorted(candidates[selected])
            )

            if nodes not in actual_cache:
                actual_cache[nodes] = (
                    e14.e11.helper.actual_outcome(
                        graph,
                        scenario,
                        list(nodes),
                        seed,
                    )
                )

            outcome = actual_cache[nodes]

            outcomes[variant].append(outcome)
            policies[variant].append(selected)

            details.append({
                "scenario_id": sid,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "run_index": r,
                "seed": seed,
                "variant": variant,
                "selected_policy": selected,
                "PAS": float(outcome["PAS"]),
                "failed": int(outcome["failed"]),
                "RT": float(outcome["RT"]),
                "RI": float(outcome["RI"]),
                "RE": float(outcome["RE"]),
                "candidate_count": len(candidates),
            })

    baseline = e14.e11.helper.aggregate(
        outcomes["CDP"]
    )

    for variant in VARIANTS:
        metrics = e14.e11.helper.aggregate(
            outcomes[variant]
        )

        summary.append({
            "scenario_id": sid,
            "pattern": pattern,
            "info": info,
            "budget": budget,
            "onset": onset,
            "variant": variant,
            "n_runs": n_runs,
            "EP": float(metrics["EP"]),
            "PAS": float(metrics["PAS"]),
            "PS": float(metrics["PS"]),
            "RT": float(metrics["RT"]),
            "RI": float(metrics["RI"]),
            "RE": float(metrics["RE"]),
            "switch_rate": float(np.mean([
                policy != "CDP"
                for policy in policies[variant]
            ])),
            "EP_gain_pct": (
                100.0
                * (baseline["EP"] - metrics["EP"])
                / baseline["EP"]
                if baseline["EP"] > 1e-12
                else None
            ),
            "policy_change_vs_original": float(
                np.mean([
                    x != y
                    for x, y in zip(
                        policies[variant],
                        policies["Original"],
                    )
                ])
            ),
        })

    return {
        "scenario_id": sid,
        "summary": summary,
        "details": details,
    }


def check_pilot(summary, details):
    ref_path = (
        OUT / "14_replication_pilot" / "per_run.csv"
    )

    if not ref_path.exists():
        raise RuntimeError(
            "Validated Experiment 14 pilot not found"
        )

    reference = pd.read_csv(ref_path)

    checks = 0

    for row in details:
        if row["variant"] not in {"CDP", "Original"}:
            continue

        ref_variant = (
            "CDP"
            if row["variant"] == "CDP"
            else "Legacy-K5"
        )

        matched = reference[
            (reference["scenario_id"] == row["scenario_id"])
            & (reference["run_index"] == row["run_index"])
            & (reference["variant"] == ref_variant)
        ]

        if len(matched) != 1:
            raise RuntimeError(
                "Pilot reference record missing"
            )

        old = matched.iloc[0]

        if row["selected_policy"] != old["selected_policy"]:
            raise RuntimeError(
                "Pilot selected-policy mismatch"
            )

        for metric in [
            "PAS", "failed", "RT", "RI", "RE"
        ]:
            if not np.isclose(
                float(row[metric]),
                float(old[metric]),
                atol=1e-10,
                rtol=0,
            ):
                raise RuntimeError(
                    f"PILOT METRIC MISMATCH: {metric}"
                )

        checks += 1

    if checks != 12:
        raise RuntimeError(
            f"Expected 12 pilot checks, got {checks}"
        )

    print(
        "OBJECTIVE WEIGHT PILOT EQUIVALENCE PASSED: "
        "12/12 decisions and outcomes match.",
        flush=True,
    )


def atomic_json(path, result):
    tmp = path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    tmp.write_text(
        json.dumps(result),
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
        n_runs = 8
        base_seed = 20261201
        workers = max(
            1, min(args.workers, 12)
        )

    run_dir = (
        OUT / f"18_objective_weights_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"

    checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "experiment": "Objective weight sensitivity",
        "mode": args.mode,
        "scenarios": scenarios,
        "weights": WEIGHTS,
        "variants": VARIANTS,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "gamma": 0.10,
        "kappa": 0.50,
        "K": 5,
        "source_hashes": {
            name: digest(SRC / name)
            for name in [
                "run_experiments_core.py",
                "drad_mpc.py",
                "seid_malware.py",
            ]
        },
        "helper_hashes": {
            name: digest(
                ROOT / "revision_experiments" / name
            )
            for name in [
                "10_cf_optimization_pilot.py",
                "11_crossfit_confirmation.py",
                "14_parallel_replication.py",
            ]
        },
        "driver_hash": digest(
            Path(__file__).resolve()
        ),
    }

    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        old = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if old != manifest:
            raise RuntimeError(
                "Experiment protocol changed. "
                "Refusing to mix checkpoints."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    tasks = [
        (sid, scenario, n_runs, base_seed)
        for sid, scenario in enumerate(
            scenarios, start=1
        )
    ]

    pending = [
        task for task in tasks
        if not (
            checkpoints
            / f"scenario_{task[0]:03d}.json"
        ).exists()
    ]

    print("=" * 75, flush=True)
    print("TNSE EXPERIMENT 18", flush=True)
    print("COUNTERFACTUAL OBJECTIVE WEIGHT SENSITIVITY", flush=True)
    print("=" * 75, flush=True)

    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Weights:", list(WEIGHTS), flush=True)
    print("Workers:", workers, flush=True)
    print("Monte Carlo:", n_runs, flush=True)
    print("Pending:", len(pending), flush=True)

    start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(worker, task): task
            for task in pending
        }

        for future in as_completed(futures):
            task = futures[future]
            sid = task[0]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"OBJECTIVE TASK FAILED: {task}"
                ) from exc

            checkpoint = (
                checkpoints
                / f"scenario_{sid:03d}.json"
            )

            atomic_json(
                checkpoint,
                result,
            )

            print(
                f"[saved {sid:02d}/{len(tasks)}] "
                f"{task[1]} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    all_summary = []
    all_details = []

    for sid, _, _, _ in tasks:
        checkpoint = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        record = json.loads(
            checkpoint.read_text(
                encoding="utf-8"
            )
        )

        if record["scenario_id"] != sid:
            raise RuntimeError(
                "Checkpoint identity mismatch"
            )

        all_summary.extend(
            record["summary"]
        )

        all_details.extend(
            record["details"]
        )

    summary = pd.DataFrame(all_summary)
    details = pd.DataFrame(all_details)

    if len(summary) != len(tasks) * len(VARIANTS):
        raise RuntimeError(
            "Incomplete summary results"
        )

    if len(details) != (
        len(tasks) * n_runs * len(VARIANTS)
    ):
        raise RuntimeError(
            "Incomplete per-run results"
        )

    if summary.duplicated(
        ["scenario_id", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate summary records"
        )

    if details.duplicated(
        ["scenario_id", "run_index", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate per-run records"
        )

    if args.mode == "pilot":
        check_pilot(all_summary, all_details)

    summary_path = run_dir / "summary.csv"
    detail_path = run_dir / "per_run.csv"
    aggregate_path = run_dir / "aggregate.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    details.to_csv(
        detail_path,
        index=False,
    )

    aggregate_df = (
        summary.groupby(
            "variant",
            as_index=False,
        )
        .agg(
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_PS=("PS", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
            mean_EP_gain_pct=("EP_gain_pct", "mean"),
            mean_policy_change=(
                "policy_change_vs_original", "mean"
            ),
        )
        .sort_values("mean_EP")
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print("OBJECTIVE WEIGHT SENSITIVITY RESULTS")
    print("=" * 75)

    print(
        aggregate_df.to_string(index=False),
        flush=True,
    )

    print("\nCSV summary:", summary_path)
    print("CSV per-run:", detail_path)
    print("CSV aggregate:", aggregate_path)

    print(
        "Total runtime:",
        round(time.perf_counter()-start, 2),
        "seconds",
        flush=True,
    )

    print(
        "EXPERIMENT 18 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
