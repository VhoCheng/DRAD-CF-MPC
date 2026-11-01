from pathlib import Path
from datetime import datetime, timezone
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

helper_path = (
    ROOT / "revision_experiments"
    / "11_crossfit_confirmation.py"
)

spec = importlib.util.spec_from_file_location(
    "experiment11_validated", helper_path
)

e11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e11)

PILOT = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

TEST = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.50],
    [40, 80],
    [180, 270],
))

VARIANTS = [
    "CDP",
    "Legacy-K5",
    "Confirm15-Robust",
]


def digest(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def create_metrics5(graph, scenario, seed):
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

    candidates = mpc.generate_candidate_policies(
        pattern=pattern,
        defense_budget=budget,
        attack_information_ratio=info,
        active_nodes=preview["active_infectious_nodes"],
        exposed_nodes=preview["exposed_nodes"],
        table_dir="outputs/tables",
        seed=seed,
    )

    if "CDP" not in candidates:
        candidates["CDP"] = mpc.select_protected_nodes(
            strategy="CDP",
            defense_budget=budget,
            pattern=pattern,
            seed=seed,
            table_dir="outputs/tables",
        )

    common_seeds = [
        seed
        + 777000
        + 10007 * k
        + mpc.stable_int_hash(
            f"{pattern}-{onset}-{budget}",
            100000,
        )
        for k in range(5)
    ]

    metrics = {}

    for name, nodes in candidates.items():
        metrics[name] = (
            mpc.evaluate_candidate_with_common_random_numbers(
                candidate_nodes=nodes,
                common_seeds=common_seeds,
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

    return candidates, metrics


def run_scenario(task):
    scenario_id, scenario, n_runs, base_seed = task

    os.chdir(ROOT)

    pattern, info, budget, onset = scenario

    graph = core.build_comm_graph(
        n_nodes=118,
        seed=42,
    )

    outcomes = {
        name: [] for name in VARIANTS
    }

    selections = {
        name: [] for name in VARIANTS
    }

    detail_rows = []

    selection_times = []

    for r in range(n_runs):
        seed = (
            base_seed
            + r
            + onset * 1000
            + budget * 17
        )

        # Stage 1: candidate generation and 5 CF samples.
        t1 = time.perf_counter()

        candidates, metrics5 = create_metrics5(
            graph,
            scenario,
            seed,
        )

        legacy, _ = e11.helper.choose_policy(
            metrics5,
            eta=0.0,
            kappa=0.5,
        )

        stage1_seconds = time.perf_counter() - t1

        # Stage 2: fresh independent confirmation samples.
        t2 = time.perf_counter()

        shortlist, confirmed = e11.confirm_candidates(
            graph,
            scenario,
            seed,
            candidates,
            metrics5,
        )

        robust = e11.select_confirmed(
            confirmed,
            eta=0.5,
            kappa=1.0,
        )

        stage2_seconds = time.perf_counter() - t2

        decisions = {
            "CDP": "CDP",
            "Legacy-K5": legacy,
            "Confirm15-Robust": robust,
        }

        cache = {}

        for variant, selected in decisions.items():
            nodes = tuple(
                sorted(candidates[selected])
            )

            if nodes not in cache:
                cache[nodes] = (
                    e11.helper.actual_outcome(
                        graph,
                        scenario,
                        list(nodes),
                        seed,
                    )
                )

            outcome = cache[nodes]

            outcomes[variant].append(outcome)
            selections[variant].append(selected)

            detail_rows.append({
                "scenario_id": scenario_id,
                "pattern": pattern,
                "info": info,
                "budget": budget,
                "onset": onset,
                "run_index": r,
                "seed": seed,
                "variant": variant,
                "selected_policy": selected,
                "shortlist": "|".join(shortlist),
                "PAS": float(outcome["PAS"]),
                "failed": int(outcome["failed"]),
                "RT": float(outcome["RT"]),
                "RI": float(outcome["RI"]),
                "RE": float(outcome["RE"]),
                "stage1_seconds": stage1_seconds,
                "stage2_seconds": stage2_seconds,
            })

        selection_times.append(
            [stage1_seconds, stage2_seconds]
        )

    baseline = e11.helper.aggregate(
        outcomes["CDP"]
    )

    summary_rows = []

    for variant in VARIANTS:
        metrics = e11.helper.aggregate(
            outcomes[variant]
        )

        summary_rows.append({
            "scenario_id": scenario_id,
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
                for policy in selections[variant]
            ])),
            "EP_gain_pct": (
                100.0
                * (
                    baseline["EP"]
                    - metrics["EP"]
                )
                / baseline["EP"]
                if baseline["EP"] > 1e-12
                else None
            ),
            "mean_stage1_seconds": float(
                np.mean(selection_times, axis=0)[0]
            ),
            "mean_stage2_seconds": float(
                np.mean(selection_times, axis=0)[1]
            ),
        })

    return {
        "scenario_id": scenario_id,
        "summary": summary_rows,
        "details": detail_rows,
    }


def validate_pilot(summary):
    sources = sorted(
        OUT.glob("11_crossfit_pilot_*_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )

    if not sources:
        raise RuntimeError(
            "Validated Experiment 11 pilot not found"
        )

    reference = pd.read_csv(sources[-1])

    for sid in [1, 2]:
        for variant in VARIANTS:
            old = reference[
                (reference["scenario_id"] == sid)
                & (reference["variant"] == variant)
            ]

            new = summary[
                (summary["scenario_id"] == sid)
                & (summary["variant"] == variant)
            ]

            if len(old) != 1 or len(new) != 1:
                raise RuntimeError(
                    "Pilot reference row missing"
                )

            for metric in [
                "EP", "PAS", "PS",
                "RT", "RI", "RE", "switch_rate"
            ]:
                a = float(old.iloc[0][metric])
                b = float(new.iloc[0][metric])

                if not np.isclose(
                    a, b, atol=1e-10, rtol=0
                ):
                    raise RuntimeError(
                        "PILOT REFERENCE FAILED: "
                        f"{sid} {variant} {metric} "
                        f"old={a}, new={b}"
                    )

    print(
        "PARALLEL PILOT REFERENCE PASSED",
        flush=True
    )


def save_checkpoint(path, result):
    temp = path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    temp.write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "test"],
        required=True
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8
    )

    args = parser.parse_args()

    e11.helper.check_original()

    if args.mode == "pilot":
        scenarios = PILOT
        n_runs = 3
        base_seed = 42
        workers = 2
    else:
        scenarios = TEST
        n_runs = 15
        base_seed = 20261115
        workers = max(
            1, min(args.workers, 12)
        )

    run_dir = OUT / (
        f"14_replication_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(
        parents=True, exist_ok=True
    )

    manifest = {
        "mode": args.mode,
        "scenarios": scenarios,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "variants": VARIANTS,
        "frozen_parameters": {
            "K1": 5,
            "K2": 15,
            "eta": 0.5,
            "gamma": 0.10,
            "kappa": 1.00,
        },
        "source_hashes": {
            name: digest(SRC / name)
            for name in [
                "drad_mpc.py",
                "run_experiments_core.py",
                "seid_malware.py",
            ]
        },
        "experiment_helper_hashes": {
            name: digest(
                ROOT / "revision_experiments" / name
            )
            for name in [
                "10_cf_optimization_pilot.py",
                "11_crossfit_confirmation.py",
            ]
        },
        "parallel_driver_hash": digest(
            Path(__file__).resolve()
        ),
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
                "EXPERIMENT PROTOCOL CHANGED. "
                "Refusing to reuse old checkpoints."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    tasks = [
        (i, scenario, n_runs, base_seed)
        for i, scenario in enumerate(
            scenarios, start=1
        )
    ]

    pending = []

    for task in tasks:
        sid = task[0]

        ckpt = (
            checkpoints / f"scenario_{sid:03d}.json"
        )

        if not ckpt.exists():
            pending.append(task)

    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 14", flush=True)
    print("PARALLEL FROZEN-METHOD REPLICATION", flush=True)
    print("=" * 72, flush=True)

    print("Mode:", args.mode, flush=True)
    print("CPU visible:", os.cpu_count(), flush=True)
    print("Workers:", workers, flush=True)
    print("Scenarios:", len(tasks), flush=True)
    print("Completed checkpoints:",
          len(tasks) - len(pending), flush=True)
    print("Remaining scenarios:", len(pending), flush=True)
    print("Monte Carlo:", n_runs, flush=True)
    print("Base seed:", base_seed, flush=True)

    start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(run_scenario, task): task
            for task in pending
        }

        for future in as_completed(futures):
            task = futures[future]
            sid = task[0]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"SCENARIO FAILED: {task}"
                ) from exc

            ckpt = (
                checkpoints
                / f"scenario_{sid:03d}.json"
            )

            save_checkpoint(ckpt, result)

            print(
                f"[saved {sid:02d}/{len(tasks)}] "
                f"{task[1]} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    all_summary = []
    all_details = []

    for task in tasks:
        sid = task[0]

        ckpt = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        if not ckpt.exists():
            raise RuntimeError(
                f"Missing completed checkpoint: {sid}"
            )

        result = json.loads(
            ckpt.read_text(encoding="utf-8")
        )

        if int(result["scenario_id"]) != sid:
            raise RuntimeError(
                "Checkpoint scenario ID mismatch"
            )

        all_summary.extend(
            result["summary"]
        )

        all_details.extend(
            result["details"]
        )

    summary = pd.DataFrame(all_summary)
    details = pd.DataFrame(all_details)

    expected_summary = (
        len(tasks) * len(VARIANTS)
    )

    expected_details = (
        len(tasks) * n_runs * len(VARIANTS)
    )

    if len(summary) != expected_summary:
        raise RuntimeError(
            "Incomplete summary records"
        )

    if len(details) != expected_details:
        raise RuntimeError(
            "Incomplete per-run records"
        )

    if args.mode == "pilot":
        validate_pilot(summary)

    summary_path = run_dir / "summary.csv"
    detail_path = run_dir / "per_run.csv"
    aggregate_path = run_dir / "aggregate.csv"

    summary.to_csv(
        summary_path, index=False
    )

    details.to_csv(
        detail_path, index=False
    )

    aggregate_df = (
        summary.groupby(
            "variant", as_index=False
        )
        .agg(
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_PS=("PS", "mean"),
            mean_EP_gain_pct=(
                "EP_gain_pct", "mean"
            ),
            mean_switch_rate=(
                "switch_rate", "mean"
            ),
            mean_stage1_seconds=(
                "mean_stage1_seconds", "mean"
            ),
            mean_stage2_seconds=(
                "mean_stage2_seconds", "mean"
            ),
        )
        .sort_values("mean_EP")
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False
    )

    print("\n" + "=" * 72)
    print("PARALLEL REPLICATION RESULTS")
    print("=" * 72)

    print(
        aggregate_df.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run:", detail_path)
    print("Aggregate:", aggregate_path)
    print("Checkpoints:", checkpoints)

    print(
        "Runtime:",
        round(
            time.perf_counter() - start,
            2
        ),
        "seconds",
        flush=True,
    )

    print(
        "EXPERIMENT 14 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
