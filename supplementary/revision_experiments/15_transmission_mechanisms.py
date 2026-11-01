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
OUT = ROOT / "revision_experiments" / "results"

sys.path.insert(0, str(SRC))

import run_experiments_core as core
import drad_mpc as mpc
import seid_malware as seid

HELPER_PATH = (
    ROOT / "revision_experiments"
    / "14_parallel_replication.py"
)

spec = importlib.util.spec_from_file_location(
    "validated_experiment14",
    HELPER_PATH,
)

e14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e14)

MECHANISMS = [
    "original",
    "degree_source",
    "degree_target",
    "temporal_burst",
]

PILOT_SCENARIOS = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

FULL_SCENARIOS = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.25, 1.00],
    [40, 80],
    [150, 240],
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


def alternate_simulator(
    Gc,
    onset_time,
    infection_rate=0.01,
    activation_rate=0.15,
    detection_rate=0.00001,
    attack_information_ratio=0.5,
    protected_nodes=None,
    strict_detection_failure=True,
    seed=42,
    mechanism="degree_source",
):
    """
    Research extension of the original SEID simulator.

    Only the S->E transmission hazard changes.

    Initial attacker choice, protected-node treatment,
    E->I activation and I->D detection follow the
    original implementation.

    This is an experimental alternative, not a change
    to the original repository's simulator.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(mechanism)

    if mechanism == "original":
        return seid.simulate_seid_once(
            Gc=Gc,
            onset_time=onset_time,
            infection_rate=infection_rate,
            activation_rate=activation_rate,
            detection_rate=detection_rate,
            attack_information_ratio=attack_information_ratio,
            protected_nodes=protected_nodes,
            strict_detection_failure=strict_detection_failure,
            seed=seed,
        )

    rng = np.random.default_rng(seed)

    n = Gc.number_of_nodes()

    if set(Gc.nodes()) != set(range(n)):
        raise RuntimeError(
            "Communication graph must use node IDs 0..n-1."
        )

    protected = set(
        protected_nodes
        if protected_nodes is not None
        else []
    )

    states = np.zeros(n, dtype=int)

    initial_node, observed_nodes = (
        seid.choose_initial_infection_node(
            Gc=Gc,
            attack_information_ratio=attack_information_ratio,
            protected_nodes=protected,
            seed=seed,
        )
    )

    states[initial_node] = seid.EXPOSED

    degrees = dict(Gc.degree())
    mean_degree = (
        2.0 * Gc.number_of_edges() / n
    )

    trajectory = []
    attack_failed = False
    detection_time = None

    for t in range(onset_time + 1):
        counts = {
            "t": t,
            "S": int(np.sum(states == seid.SUSCEPTIBLE)),
            "E": int(np.sum(states == seid.EXPOSED)),
            "I": int(np.sum(states == seid.INFECTIOUS)),
            "D": int(np.sum(states == seid.DETECTED)),
        }

        trajectory.append(counts)

        if t == onset_time:
            break

        new_states = states.copy()

        # S -> E: the only mechanism-specific stage.
        for node in range(n):
            if states[node] != seid.SUSCEPTIBLE:
                continue

            hardening = (
                0.2 if node in protected else 1.0
            )

            sources = [
                int(nb)
                for nb in Gc.neighbors(node)
                if states[int(nb)] == seid.INFECTIOUS
            ]

            if not sources:
                continue

            survival = 1.0

            for source in sources:
                multiplier = 1.0

                if mechanism == "degree_source":
                    ratio = (
                        degrees[source]
                        / max(mean_degree, 1e-12)
                    )

                    multiplier = float(np.clip(
                        0.5 + 0.5 * ratio,
                        0.5,
                        2.0,
                    ))

                elif mechanism == "degree_target":
                    ratio = (
                        degrees[node]
                        / max(mean_degree, 1e-12)
                    )

                    multiplier = float(np.clip(
                        0.5 + 0.5 * ratio,
                        0.5,
                        2.0,
                    ))

                elif mechanism == "temporal_burst":
                    if t < onset_time / 2:
                        multiplier = 0.6
                    else:
                        multiplier = 1.4

                hazard = float(np.clip(
                    infection_rate
                    * hardening
                    * multiplier,
                    0.0,
                    1.0,
                ))

                survival *= 1.0 - hazard

            infection_probability = 1.0 - survival

            if rng.random() < infection_probability:
                new_states[node] = seid.EXPOSED

        # E -> I: unchanged.
        for node in range(n):
            if states[node] == seid.EXPOSED:
                if rng.random() < activation_rate:
                    new_states[node] = seid.INFECTIOUS

        # I -> D: unchanged.
        for node in range(n):
            if states[node] == seid.INFECTIOUS:
                if rng.random() < detection_rate:
                    new_states[node] = seid.DETECTED

                    if strict_detection_failure:
                        attack_failed = True
                        detection_time = t + 1

        states = new_states

        if attack_failed and strict_detection_failure:
            for tt in range(t + 1, onset_time + 1):
                trajectory.append({
                    "t": tt,
                    "S": int(np.sum(
                        states == seid.SUSCEPTIBLE
                    )),
                    "E": int(np.sum(
                        states == seid.EXPOSED
                    )),
                    "I": int(np.sum(
                        states == seid.INFECTIOUS
                    )),
                    "D": int(np.sum(
                        states == seid.DETECTED
                    )),
                })
            break

    active_nodes = np.where(
        states == seid.INFECTIOUS
    )[0].astype(int).tolist()

    exposed_nodes = np.where(
        states == seid.EXPOSED
    )[0].astype(int).tolist()

    detected_nodes = np.where(
        states == seid.DETECTED
    )[0].astype(int).tolist()

    if (
        len(active_nodes)
        + len(exposed_nodes)
        + len(detected_nodes)
        > n
    ):
        raise RuntimeError(
            "Invalid SEID state partition."
        )

    return {
        "onset_time": onset_time,
        "initial_node": int(initial_node),
        "observed_nodes": sorted(
            map(int, observed_nodes)
        ),
        "protected_nodes": sorted(
            map(int, protected)
        ),
        "attack_information_ratio": (
            attack_information_ratio
        ),
        "infection_rate": infection_rate,
        "activation_rate": activation_rate,
        "detection_rate": detection_rate,
        "strict_detection_failure": (
            strict_detection_failure
        ),
        "attack_failed": bool(attack_failed),
        "detection_time": detection_time,
        "n_active_infectious": len(active_nodes),
        "n_exposed": len(exposed_nodes),
        "n_detected": len(detected_nodes),
        "active_infectious_nodes": active_nodes,
        "exposed_nodes": exposed_nodes,
        "detected_nodes": detected_nodes,
        "trajectory": trajectory,
    }


def worker(task):
    (
        job_id,
        base_scenario_id,
        scenario,
        mechanism,
        n_runs,
        base_seed,
    ) = task

    os.chdir(ROOT)

    original_core = core.simulate_seid_once
    original_mpc = mpc.simulate_seid_once

    # Monkey-patching is confined to this worker process.
    # No source file is changed.
    if mechanism != "original":
        def simulation_for_mechanism(*args, **kwargs):
            return alternate_simulator(
                *args,
                **kwargs,
                mechanism=mechanism,
            )

        core.simulate_seid_once = (
            simulation_for_mechanism
        )

        mpc.simulate_seid_once = (
            simulation_for_mechanism
        )

    try:
        result = e14.run_scenario((
            job_id,
            scenario,
            n_runs,
            base_seed,
        ))

        result["base_scenario_id"] = (
            base_scenario_id
        )
        result["mechanism"] = mechanism

        for row in result["summary"]:
            row["base_scenario_id"] = (
                base_scenario_id
            )
            row["mechanism"] = mechanism

        for row in result["details"]:
            row["base_scenario_id"] = (
                base_scenario_id
            )
            row["mechanism"] = mechanism

        return result

    finally:
        core.simulate_seid_once = original_core
        mpc.simulate_seid_once = original_mpc


def save_checkpoint(path, result):
    temporary = path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    temporary.write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    temporary.replace(path)


def validate_original_pilot(summary):
    reference_path = (
        OUT
        / "14_replication_pilot"
        / "summary.csv"
    )

    if not reference_path.exists():
        raise RuntimeError(
            "Experiment 14 pilot reference missing."
        )

    reference = pd.read_csv(reference_path)

    original = summary[
        summary["mechanism"] == "original"
    ]

    if len(original) != 6:
        raise RuntimeError(
            "Expected six original pilot rows."
        )

    checks = 0

    for _, row in original.iterrows():
        matched = reference[
            (
                reference["scenario_id"]
                == row["base_scenario_id"]
            )
            & (
                reference["variant"]
                == row["variant"]
            )
        ]

        if len(matched) != 1:
            raise RuntimeError(
                "Original pilot reference row missing."
            )

        previous = matched.iloc[0]

        for metric in [
            "EP",
            "PAS",
            "PS",
            "RT",
            "RI",
            "RE",
            "switch_rate",
        ]:
            if not np.isclose(
                row[metric],
                previous[metric],
                atol=1e-10,
                rtol=0,
            ):
                raise RuntimeError(
                    "ORIGINAL REFERENCE FAILED: "
                    f"{row['base_scenario_id']} "
                    f"{row['variant']} "
                    f"{metric}"
                )

        checks += 1

    if checks != 6:
        raise RuntimeError(
            "Incomplete reference checks."
        )

    print(
        "ORIGINAL PROPAGATION REFERENCE PASSED",
        flush=True,
    )


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

    # Original repository remains untouched.
    e14.e11.helper.check_original()

    if args.mode == "pilot":
        scenarios = PILOT_SCENARIOS
        n_runs = 3
        base_seed = 42
        workers = 2
    else:
        scenarios = FULL_SCENARIOS
        n_runs = 10
        base_seed = 20261125
        workers = max(
            1,
            min(args.workers, 12),
        )

    run_dir = (
        OUT
        / f"15_transmission_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "experiment": "15_transmission",
        "mode": args.mode,
        "scenarios": scenarios,
        "mechanisms": MECHANISMS,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "variants": VARIANTS,
        "mechanism_details": {
            "original": "Unmodified SEID",
            "degree_source": (
                "Source-degree-dependent infection hazard"
            ),
            "degree_target": (
                "Target-degree-dependent infection hazard"
            ),
            "temporal_burst": (
                "Hazard multiplier 0.6/1.4 "
                "before/after half of onset"
            ),
        },
        "model_parameters": {
            "K1": 5,
            "K2": 15,
            "eta": 0.5,
            "gamma": 0.10,
            "kappa": 1.00,
        },
        "source_hashes": {
            name: digest(SRC / name)
            for name in [
                "seid_malware.py",
                "run_experiments_core.py",
                "drad_mpc.py",
            ]
        },
        "helper_hashes": {
            name: digest(
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
        "driver_hash": digest(
            Path(__file__).resolve()
        ),
    }

    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        existing = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if existing != manifest:
            raise RuntimeError(
                "Protocol differs from saved experiment. "
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

    tasks = []

    for base_scenario_id, scenario in enumerate(
        scenarios,
        start=1,
    ):
        for mechanism in MECHANISMS:
            tasks.append((
                len(tasks) + 1,
                base_scenario_id,
                scenario,
                mechanism,
                n_runs,
                base_seed,
            ))

    pending = []

    for task in tasks:
        job_id = task[0]

        checkpoint = (
            checkpoints
            / f"job_{job_id:03d}.json"
        )

        if not checkpoint.exists():
            pending.append(task)

    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 15", flush=True)
    print("TRANSMISSION-MECHANISM ROBUSTNESS", flush=True)
    print("=" * 72, flush=True)

    print("Mode:", args.mode, flush=True)
    print("Workers:", workers, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Mechanisms:", MECHANISMS, flush=True)
    print("Total tasks:", len(tasks), flush=True)
    print("Pending:", len(pending), flush=True)
    print("Monte Carlo:", n_runs, flush=True)

    start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(worker, task): task
            for task in pending
        }

        completed = len(tasks) - len(pending)

        for future in as_completed(futures):
            task = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"MECHANISM TASK FAILED: {task}"
                ) from exc

            checkpoint = (
                checkpoints
                / f"job_{task[0]:03d}.json"
            )

            save_checkpoint(
                checkpoint,
                result,
            )

            completed += 1

            print(
                f"[{completed:03d}/{len(tasks)}] "
                f"scenario={task[1]} "
                f"mechanism={task[3]} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    summary_rows = []
    detail_rows = []

    for task in tasks:
        checkpoint = (
            checkpoints
            / f"job_{task[0]:03d}.json"
        )

        result = json.loads(
            checkpoint.read_text(
                encoding="utf-8"
            )
        )

        if (
            result["scenario_id"] != task[0]
            or result["base_scenario_id"] != task[1]
            or result["mechanism"] != task[3]
        ):
            raise RuntimeError(
                "Checkpoint identity mismatch."
            )

        summary_rows.extend(
            result["summary"]
        )

        detail_rows.extend(
            result["details"]
        )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    if len(summary) != len(tasks) * 3:
        raise RuntimeError(
            "Incomplete mechanism summary."
        )

    if len(details) != len(tasks) * n_runs * 3:
        raise RuntimeError(
            "Incomplete mechanism per-run records."
        )

    if summary.duplicated(
        ["base_scenario_id", "mechanism", "variant"]
    ).any():
        raise RuntimeError(
            "Duplicate mechanism summary."
        )

    if args.mode == "pilot":
        validate_original_pilot(summary)

    summary_path = run_dir / "summary.csv"
    details_path = run_dir / "per_run.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    details.to_csv(
        details_path,
        index=False,
    )

    aggregate = (
        summary.groupby(
            ["mechanism", "variant"],
            as_index=False,
        )
        .agg(
            mean_EP=("EP", "mean"),
            mean_PAS=("PAS", "mean"),
            mean_PS=("PS", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
        )
    )

    wide = aggregate.pivot(
        index="mechanism",
        columns="variant",
        values="mean_EP",
    )

    mechanism_gain = pd.DataFrame({
        "mechanism": wide.index,
        "Robust_vs_CDP_pct": (
            100.0
            * (
                wide["CDP"]
                - wide["Confirm15-Robust"]
            )
            / wide["CDP"]
        ),
        "Legacy_vs_CDP_pct": (
            100.0
            * (
                wide["CDP"]
                - wide["Legacy-K5"]
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

    aggregate_path = (
        run_dir / "aggregate.csv"
    )

    gain_path = (
        run_dir / "mechanism_gain.csv"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    mechanism_gain.to_csv(
        gain_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("TRANSMISSION MECHANISM RESULTS")
    print("=" * 72)

    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\n" + "=" * 72)
    print("MECHANISM-LEVEL EP GAINS")
    print("=" * 72)

    print(
        mechanism_gain.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run:", details_path)
    print("Aggregate:", aggregate_path)
    print("Gains:", gain_path)

    print(
        "Runtime:",
        round(time.perf_counter() - start, 2),
        "seconds",
        flush=True,
    )

    print(
        "EXPERIMENT 15 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
