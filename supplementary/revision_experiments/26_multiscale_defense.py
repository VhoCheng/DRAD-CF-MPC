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
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "revision_experiments/results"
NETWORKS = OUT / "25_multiscale_networks"

sys.path.insert(0, str(SRC))

import run_experiments_core as core
import drad_mpc as mpc
import dynamic_risk_defense as risk

HELPER = (
    ROOT
    / "revision_experiments"
    / "14_parallel_replication.py"
)

spec = importlib.util.spec_from_file_location(
    "validated_experiment14_multiscale",
    HELPER,
)

e14 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e14
spec.loader.exec_module(e14)

SIZES = [14, 57, 118, 300]

PATTERNS = [
    "DDAC", "DDDC", "DCAC", "DCDC", "RC"
]

BUDGET_RATIOS = [0.10, 0.30]
ONSETS = [120, 240]

METHODS = [
    "CDP",
    "Legacy-K5",
    "Confirm15-Robust",
]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def budget_for(n, fraction):
    return max(
        1,
        min(n, int(np.rint(n * fraction))),
    )


def validate_network(n):
    case_dir = NETWORKS / f"case{n}"
    tables = case_dir / "outputs/tables"

    files = [
        tables / "comm_features.csv",
        tables / "power_features.csv",
    ]

    files.extend(
        tables / f"coupling_{p}.csv"
        for p in PATTERNS
    )

    missing = [
        str(p) for p in files
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Experiment 25 files:\n"
            + "\n".join(missing)
        )

    comm = pd.read_csv(files[0])
    power = pd.read_csv(files[1])

    if len(comm) != n or len(power) != n:
        raise RuntimeError(
            f"Incorrect table sizes for IEEE {n}"
        )

    expected = set(range(n))

    if set(comm["comm_node"]) != expected:
        raise RuntimeError(
            f"Invalid communication IDs for IEEE {n}"
        )

    if set(power["power_node"]) != expected:
        raise RuntimeError(
            f"Invalid physical IDs for IEEE {n}"
        )

    for pattern in PATTERNS:
        coupling = pd.read_csv(
            tables / f"coupling_{pattern}.csv"
        )

        if len(coupling) != n:
            raise RuntimeError(
                f"Bad coupling length: {n}, {pattern}"
            )

        if (
            set(coupling["comm_node"]) != expected
            or set(coupling["power_node"]) != expected
        ):
            raise RuntimeError(
                f"Coupling is not one-to-one: "
                f"{n}, {pattern}"
            )

    summary_path = (
        case_dir / "network_summary.json"
    )

    if not summary_path.exists():
        raise RuntimeError(
            f"Missing power-flow summary: {n}"
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if not summary["ac_pf_converged"]:
        raise RuntimeError(
            f"Baseline AC PF failed for IEEE {n}"
        )

    return case_dir, files


def run_one(task):
    (
        job_id,
        n,
        pattern,
        info,
        fraction,
        onset,
        n_runs,
        base_seed,
    ) = task

    case_dir, _ = validate_network(n)

    budget = budget_for(n, fraction)

    scenario = (
        pattern,
        info,
        budget,
        onset,
    )

    # Every worker is a separate process.
    # Patching here cannot modify the parent process
    # or the original source files.
    original_cwd = Path.cwd()

    old_root = e14.ROOT
    old_core_graph = core.build_comm_graph
    old_mpc_graph = mpc.build_comm_graph
    old_risk_graph = risk.build_comm_graph

    old_mpc_resilience = (
        mpc.evaluate_resilience_proxy_mpc
    )

    old_core_resilience = (
        core.evaluate_resilience_metrics
    )

    def consistent_graph(n_nodes=118, seed=42):
        return nx.barabasi_albert_graph(
            n=n, m=2, seed=42
        )

    def mpc_resilience(
        pas, defense_budget, max_budget=100
    ):
        return old_mpc_resilience(
            pas=pas,
            defense_budget=defense_budget,
            max_budget=n,
        )

    def core_resilience(
        pas, defense_budget, max_budget=100
    ):
        return old_core_resilience(
            pas=pas,
            defense_budget=defense_budget,
            max_budget=n,
        )

    started = time.perf_counter()

    try:
        os.chdir(case_dir)

        # Experiment 14 calls os.chdir(e14.ROOT).
        # Redirect it to the isolated case workspace.
        e14.ROOT = case_dir

        # Every component now references the same
        # fixed BA communication graph for this case.
        core.build_comm_graph = consistent_graph
        mpc.build_comm_graph = consistent_graph
        risk.build_comm_graph = consistent_graph

        # Scale-normalized defense budget.
        mpc.evaluate_resilience_proxy_mpc = (
            mpc_resilience
        )

        core.evaluate_resilience_metrics = (
            core_resilience
        )

        # Reuse fully validated Experiment 14 logic:
        # CDP / Legacy-K5 / Confirm15-Robust.
        result = e14.run_scenario((
            job_id,
            scenario,
            n_runs,
            base_seed,
        ))

    finally:
        core.build_comm_graph = old_core_graph
        mpc.build_comm_graph = old_mpc_graph
        risk.build_comm_graph = old_risk_graph

        mpc.evaluate_resilience_proxy_mpc = (
            old_mpc_resilience
        )

        core.evaluate_resilience_metrics = (
            old_core_resilience
        )

        e14.ROOT = old_root
        os.chdir(original_cwd)

    result["case_size"] = n
    result["budget_fraction"] = fraction

    result["runtime_seconds"] = (
        time.perf_counter() - started
    )

    for row in result["summary"]:
        row["case_size"] = n
        row["budget_fraction"] = fraction
        row["adaptation"] = (
            "consistent_BA42_scale_normalized_budget"
        )

    for row in result["details"]:
        row["case_size"] = n
        row["budget_fraction"] = fraction
        row["adaptation"] = (
            "consistent_BA42_scale_normalized_budget"
        )

    return result


def atomic_save(path, record):
    temporary = path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    temporary.write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    temporary.replace(path)


def verify_result(result, task):
    (
        job_id,
        n,
        pattern,
        info,
        fraction,
        onset,
        n_runs,
        base_seed,
    ) = task

    if result["scenario_id"] != job_id:
        raise RuntimeError(
            "Scenario ID mismatch"
        )

    if result["case_size"] != n:
        raise RuntimeError(
            "Case size mismatch"
        )

    if len(result["summary"]) != 3:
        raise RuntimeError(
            "Expected three summary rows"
        )

    if len(result["details"]) != 3 * n_runs:
        raise RuntimeError(
            "Missing Monte Carlo records"
        )

    summary = pd.DataFrame(
        result["summary"]
    )

    details = pd.DataFrame(
        result["details"]
    )

    if set(summary["variant"]) != set(METHODS):
        raise RuntimeError(
            "Unexpected defense methods"
        )

    if details.duplicated(
        ["run_index", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate Monte Carlo records"
        )

    for name in METHODS:
        sub = details[
            details["variant"] == name
        ]

        if len(sub) != n_runs:
            raise RuntimeError(
                f"Incomplete runs: {name}"
            )

        pas = float(sub["PAS"].mean())

        ps = (
            1.0
            - float(sub["failed"].mean())
        )

        ep = pas * ps

        ref = summary[
            summary["variant"] == name
        ].iloc[0]

        for metric, value in [
            ("PAS", pas),
            ("PS", ps),
            ("EP", ep),
        ]:
            if not np.isclose(
                value,
                float(ref[metric]),
                atol=1e-10,
                rtol=0,
            ):
                raise RuntimeError(
                    "Metric reconstruction mismatch: "
                    f"IEEE{n}, {name}, {metric}"
                )

    return True


def create_tasks(mode):
    if mode == "pilot":
        # Each of the four new network sizes is tested.
        # The pilot is not a historical 118-node
        # equivalence test: these are rebuilt tables.
        configs = [
            (14, "DDAC", 0.50, 0.10, 120),
            (57, "DDDC", 0.50, 0.30, 120),
            (118, "DCAC", 0.50, 0.10, 120),
            (300, "RC", 0.50, 0.30, 120),
        ]

        n_runs = 2
        base_seed = 42

    else:
        configs = list(product(
            SIZES,
            PATTERNS,
            [0.50],
            BUDGET_RATIOS,
            ONSETS,
        ))

        n_runs = 4
        base_seed = 20261031

    tasks = [
        (
            i,
            n,
            pattern,
            info,
            fraction,
            onset,
            n_runs,
            base_seed,
        )
        for i, (
            n,
            pattern,
            info,
            fraction,
            onset,
        ) in enumerate(
            configs, start=1
        )
    ]

    return tasks


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

    # Check original source files have not changed.
    e14.e11.helper.check_original()

    tasks = create_tasks(args.mode)

    workers = (
        2 if args.mode == "pilot"
        else max(1, min(args.workers, 12))
    )

    for n in SIZES:
        validate_network(n)

    run_dir = (
        OUT / f"26_multiscale_{args.mode}"
    )

    checkpoints = (
        run_dir / "checkpoints"
    )

    checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_files = [
        "drad_mpc.py",
        "dynamic_risk_defense.py",
        "run_experiments_core.py",
        "seid_malware.py",
        "baseline_defense.py",
    ]

    network_hashes = {}

    for n in SIZES:
        _, files = validate_network(n)

        network_hashes[str(n)] = {
            str(p.relative_to(ROOT)): sha(p)
            for p in files
        }

    manifest = {
        "experiment": "IEEE multiscale CPPS defense",
        "mode": args.mode,
        "cases": SIZES,
        "tasks": tasks,
        "methods": METHODS,
        "budget_ratios": BUDGET_RATIOS,
        "fixed_information_ratio": 0.50,
        "attack_onsets": ONSETS,
        "K_stage1": 5,
        "K_stage2": 15,
        "gamma": 0.10,
        "kappa_confirm": 1.0,
        "eta_confirm": 0.5,
        "adaptations": {
            "communication_graph": (
                "same BA(n,2,seed=42) for propagation "
                "and dynamic-risk estimation"
            ),
            "budget_normalization": (
                "defense_budget / number_of_comm_nodes"
            ),
            "physical_impact": (
                "original PAS proxy with new "
                "IEEE-case-specific physical tables"
            ),
            "power_flow_scope": (
                "baseline AC PF verified by Exp25; "
                "NO post-attack AC PF in Exp26"
            ),
        },
        "source_hashes": {
            name: sha(SRC / name)
            for name in source_files
        },
        "helper_hashes": {
            name: sha(
                ROOT
                / "revision_experiments"
                / name
            )
            for name in [
                "10_cf_optimization_pilot.py",
                "11_crossfit_confirmation.py",
                "14_parallel_replication.py",
            ]
        },
        "network_table_hashes": network_hashes,
        "driver_sha256": sha(
            Path(__file__).resolve()
        ),
    }

    manifest_path = (
        run_dir / "manifest.json"
    )

    if manifest_path.exists():
        previous = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if previous != manifest:
            raise RuntimeError(
                "EXPERIMENT PROTOCOL CHANGED. "
                "Refusing to mix checkpoints."
            )

    else:
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    pending = []

    for task in tasks:
        sid = task[0]

        checkpoint = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        if checkpoint.exists():
            old = json.loads(
                checkpoint.read_text(
                    encoding="utf-8"
                )
            )

            verify_result(old, task)

        else:
            pending.append(task)

    print("=" * 75, flush=True)
    print("TNSE EXPERIMENT 26", flush=True)
    print("IEEE MULTISCALE DEFENSE AND COMPLEXITY", flush=True)
    print("=" * 75, flush=True)

    print("Mode:", args.mode, flush=True)
    print("IEEE cases:", SIZES, flush=True)
    print("Total scenarios:", len(tasks), flush=True)
    print("Pending scenarios:", len(pending), flush=True)
    print("Workers:", workers, flush=True)

    print(
        "Monte Carlo per scenario:",
        tasks[0][6],
        flush=True,
    )

    print(
        "Scale normalization: B / N_comm",
        flush=True,
    )

    started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(run_one, task): task
            for task in pending
        }

        completed = len(tasks) - len(pending)

        for future in as_completed(futures):
            task = futures[future]

            try:
                result = future.result()

                verify_result(
                    result,
                    task,
                )

            except Exception as exc:
                raise RuntimeError(
                    f"FAILED SCENARIO: {task}"
                ) from exc

            sid = task[0]

            checkpoint = (
                checkpoints
                / f"scenario_{sid:03d}.json"
            )

            atomic_save(
                checkpoint,
                result,
            )

            completed += 1

            print(
                f"[{completed}/{len(tasks)}] "
                f"IEEE{task[1]} "
                f"{task[2]} "
                f"budget={budget_for(task[1], task[4])} "
                f"onset={task[5]} "
                f"elapsed="
                f"{time.perf_counter()-started:.1f}s",
                flush=True,
            )

    summaries = []
    details = []
    timing = []

    for task in tasks:
        checkpoint = (
            checkpoints
            / f"scenario_{task[0]:03d}.json"
        )

        data = json.loads(
            checkpoint.read_text(
                encoding="utf-8"
            )
        )

        verify_result(data, task)

        summaries.extend(
            data["summary"]
        )

        details.extend(
            data["details"]
        )

        timing.append({
            "scenario_id": task[0],
            "case_size": task[1],
            "runtime_seconds": (
                data["runtime_seconds"]
            ),
        })

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(details)
    timing_df = pd.DataFrame(timing)

    expected_summary = len(tasks) * 3

    expected_detail = (
        len(tasks) * tasks[0][6] * 3
    )

    if len(summary_df) != expected_summary:
        raise RuntimeError(
            "Incomplete summary table"
        )

    if len(detail_df) != expected_detail:
        raise RuntimeError(
            "Incomplete per-run table"
        )

    aggregate = (
        summary_df.groupby(
            ["case_size", "variant"],
            as_index=False,
        )
        .agg(
            scenarios=("scenario_id", "nunique"),
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_PS=("PS", "mean"),
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
        .sort_values(
            ["case_size", "mean_EP"]
        )
    )

    wide = aggregate.pivot(
        index="case_size",
        columns="variant",
        values="mean_EP",
    )

    gains = pd.DataFrame({
        "case_size": wide.index,
        "Legacy_vs_CDP_pct": (
            100.0
            * (
                wide["CDP"]
                - wide["Legacy-K5"]
            )
            / wide["CDP"]
        ),
        "Robust_vs_CDP_pct": (
            100.0
            * (
                wide["CDP"]
                - wide["Confirm15-Robust"]
            )
            / wide["CDP"]
        ),
        "Robust_vs_Legacy_pct": (
            100.0
            * (
                wide["Legacy-K5"]
                - wide["Confirm15-Robust"]
            )
            / wide["Legacy-K5"]
        ),
    }).reset_index(drop=True)

    summary_path = run_dir / "summary.csv"
    detail_path = run_dir / "per_run.csv"
    timing_path = run_dir / "scenario_timing.csv"
    aggregate_path = run_dir / "aggregate.csv"
    gain_path = run_dir / "relative_gain.csv"

    summary_df.to_csv(
        summary_path, index=False
    )

    detail_df.to_csv(
        detail_path, index=False
    )

    timing_df.to_csv(
        timing_path, index=False
    )

    aggregate.to_csv(
        aggregate_path, index=False
    )

    gains.to_csv(
        gain_path, index=False
    )

    print("\n" + "=" * 75)
    print("IEEE MULTISCALE DEFENSE RESULTS")
    print("=" * 75)

    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\n" + "=" * 75)
    print("IEEE MULTISCALE RELATIVE GAINS")
    print("=" * 75)

    print(
        gains.to_string(index=False),
        flush=True,
    )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("Saved:", aggregate_path)
    print("Saved:", gain_path)
    print("Saved:", timing_path)

    print(
        "Elapsed seconds:",
        round(
            time.perf_counter() - started,
            2,
        ),
        flush=True,
    )

    print(
        "IMPORTANT: decision timings were measured "
        "under parallel workload."
    )

    print(
        "IMPORTANT: physical consequences use "
        "the PAS surrogate, not post-attack AC PF."
    )

    print(
        "EXPERIMENT 26 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
