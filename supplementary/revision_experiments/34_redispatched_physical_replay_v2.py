from pathlib import Path
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

from pypower.api import case14, case57, case118, case300
from pypower.idx_bus import PD, QD, VM, VA
from pypower.idx_gen import PG

ROOT = Path(__file__).resolve().parents[1]
EXPS = ROOT / "revision_experiments"
RESULTS = EXPS / "results"

SOURCE = RESULTS / "26_multiscale_full"
DISPATCH = RESULTS / "33_dispatch_pf"

CASE_FUNCS = {
    14: case14,
    57: case57,
    118: case118,
    300: case300,
}

REFERENCE = EXPS / "27_29_joint_suite.py"

spec = importlib.util.spec_from_file_location(
    "tnse_verified_replay_34",
    REFERENCE,
)
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)

METHODS = list(joint.METHODS)
PROFILES = list(joint.OPERATING_PROFILES)
SEVERITIES = list(joint.ATTACK_SEVERITIES)


def file_hash(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def dispatch_file(n, profile):
    return (
        RESULTS
        / "34_staged_dispatch"
        / f"case{n}"
        / f"{profile}.npz"
    )


def load_redispatched_case(n, profile):
    path = dispatch_file(n, profile)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing validated dispatch snapshot: {path}"
        )

    case = CASE_FUNCS[n]()

    # Gross positive load is the reference used by
    # the original physical impact mapping.
    original_pd = np.maximum(
        case["bus"][:, PD].copy(),
        0.0,
    )

    original_qd = case["bus"][:, QD].copy()

    total_gross_load = float(
        original_pd.sum()
    )

    with np.load(path, allow_pickle=False) as data:
        pd_target = data["bus_pd"].copy()
        qd_target = data["bus_qd"].copy()
        pg_target = data["gen_pg"].copy()
        vm_target = data["bus_vm"].copy()
        va_target = data["bus_va"].copy()

    if pd_target.shape != case["bus"][:, PD].shape:
        raise RuntimeError(
            f"Incorrect PD shape: IEEE{n}"
        )

    if qd_target.shape != case["bus"][:, QD].shape:
        raise RuntimeError(
            f"Incorrect QD shape: IEEE{n}"
        )

    if pg_target.shape != case["gen"][:, PG].shape:
        raise RuntimeError(
            f"Incorrect PG shape: IEEE{n}"
        )

    for arr in (pd_target, qd_target, pg_target):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(
                "Dispatch snapshot contains NaN/Inf"
            )

    case["bus"][:, PD] = pd_target
    case["bus"][:, QD] = qd_target
    case["gen"][:, PG] = pg_target
    case["bus"][:, VM] = vm_target
    case["bus"][:, VA] = va_target

    load_factor = float(
        joint.OPERATING_PROFILES[profile][0]
    )

    return (
        case,
        original_pd,
        original_qd,
        load_factor,
        total_gross_load,
    )


def check_inputs():
    for n in CASE_FUNCS:
        for profile in PROFILES:
            load_redispatched_case(n, profile)

    summary_path = DISPATCH / "summary_full.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            "Experiment 33 full summary not found"
        )

    dispatch_summary = pd.read_csv(summary_path)

    for n in CASE_FUNCS:
        for profile in PROFILES:
            row = dispatch_summary[
                (dispatch_summary["case_size"] == n)
                & (dispatch_summary["profile"] == profile)
            ]

            if len(row) != 1:
                raise RuntimeError(
                    f"Missing dispatch summary: "
                    f"IEEE{n}, {profile}"
                )

            x = row.iloc[0]

            if (
                int(x["redispatch_converged"]) != 1
                or False
                or int(x["Pg_limits_satisfied"]) != 1
            ):
                raise RuntimeError(
                    "Unvalidated dispatch snapshot: "
                    f"IEEE{n}, {profile}"
                )

    details = pd.read_csv(
        SOURCE / "per_run.csv"
    )

    if len(details) != 960:
        raise RuntimeError(
            "Expected 960 historical defense records"
        )

    if set(details["variant"]) != set(METHODS):
        raise RuntimeError(
            "Defense methods do not match history"
        )

    return details, dispatch_summary


def worker(task):
    # Use the previously validated Experiment 27-29
    # historical SEID replay and candidate selection.
    # Only the operating point is replaced.
    previous = joint.operating_case

    joint.operating_case = load_redispatched_case

    try:
        return joint.replay_worker(task)
    finally:
        joint.operating_case = previous


def atomic_json(path, data):
    temp = path.with_suffix(
        f".{os.getpid()}.tmp"
    )

    temp.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    temp.replace(path)


def validate_checkpoint(data, sid, n_runs):
    if int(data["scenario_id"]) != sid:
        raise RuntimeError(
            "Checkpoint scenario mismatch"
        )

    expected_replays = n_runs * len(METHODS)

    expected_rows = (
        expected_replays
        * len(PROFILES)
        * len(SEVERITIES)
    )

    if int(data["replay_checks"]) != expected_replays:
        raise RuntimeError(
            f"SEID replay count mismatch: {sid}"
        )

    if len(data["records"]) != expected_rows:
        raise RuntimeError(
            f"Physical row count mismatch: {sid}"
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

    os.chdir(ROOT)

    details, dispatch_summary = check_inputs()

    run_dir = RESULTS / (
        f"34_redispatched_physical_v2_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"

    checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )

    tasks = [
        (
            int(sid),
            sub.to_dict("records"),
        )
        for sid, sub in details.groupby(
            "scenario_id",
            sort=True,
        )
    ]

    if len(tasks) != 80:
        raise RuntimeError(
            "Expected 80 historical scenarios"
        )

    if args.mode == "pilot":
        selected = {}

        for task in tasks:
            n = int(task[1][0]["case_size"])
            selected.setdefault(n, task)

        tasks = [
            (
                sid,
                [
                    row
                    for row in records
                    if int(row["run_index"]) == 0
                ],
            )
            for sid, records in selected.values()
        ]

        workers = 4
        n_runs = 1

    else:
        workers = max(
            1,
            min(args.workers, 12),
        )

        n_runs = 4

    manifest = {
        "experiment": "Exp34 redispatched physical replay",
        "mode": args.mode,
        "reference_exp26_hash": file_hash(
            SOURCE / "per_run.csv"
        ),
        "reference_exp27_29_hash": file_hash(
            REFERENCE
        ),
        "exp33_summary_hash": file_hash(
            DISPATCH / "summary_full.csv"
        ),
        "dispatch_snapshots": {
            f"{n}_{profile}": file_hash(
                dispatch_file(n, profile)
            )
            for n in CASE_FUNCS
            for profile in PROFILES
        },
        "driver_hash": file_hash(
            Path(__file__).resolve()
        ),
        "methods": METHODS,
        "profiles": PROFILES,
        "severities": SEVERITIES,
        "limitations": [
            "Historical SEID attack replay",
            "Synthetic load and PV scenarios",
            "Prescribed cyber-to-load interruption mapping",
            "IEEE standard branch RATE_A may be placeholder",
            "Reactive generation limits not enforced by PF",
        ],
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
                "Protocol changed: cannot mix checkpoints"
            )
    else:
        manifest_path.write_text(
            json.dumps(
                manifest, indent=2
            ),
            encoding="utf-8",
        )

    pending = []

    for sid, rows in tasks:
        checkpoint = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        if checkpoint.exists():
            data = json.loads(
                checkpoint.read_text(
                    encoding="utf-8"
                )
            )

            validate_checkpoint(
                data, sid, n_runs
            )
        else:
            pending.append(
                (sid, rows)
            )

    print("=" * 75, flush=True)
    print("TNSE EXPERIMENT 34", flush=True)
    print("REDISPATCHED POST-ATTACK AC POWER FLOW", flush=True)
    print("=" * 75, flush=True)

    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(tasks), flush=True)
    print("Workers:", workers, flush=True)
    print("Pending:", len(pending), flush=True)

    started = time.perf_counter()

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
                validate_checkpoint(
                    result,
                    task[0],
                    n_runs,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Exp34 scenario failed: {task[0]}"
                ) from exc

            sid = task[0]

            atomic_json(
                checkpoints
                / f"scenario_{sid:03d}.json",
                result,
            )

            completed += 1

            print(
                f"[{completed}/{len(tasks)}] "
                f"IEEE{task[1][0]['case_size']} "
                f"scenario={sid} "
                f"elapsed="
                f"{time.perf_counter()-started:.1f}s",
                flush=True,
            )

    all_rows = []
    replay_checks = 0

    for sid, _ in tasks:
        path = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        validate_checkpoint(
            data, sid, n_runs
        )

        all_rows.extend(data["records"])
        replay_checks += data["replay_checks"]

    df = pd.DataFrame(all_rows)

    expected_replays = (
        len(tasks)
        * n_runs
        * len(METHODS)
    )

    expected_rows = (
        expected_replays
        * len(PROFILES)
        * len(SEVERITIES)
    )

    if replay_checks != expected_replays:
        raise RuntimeError(
            "Incomplete historical replication"
        )

    if len(df) != expected_rows:
        raise RuntimeError(
            "Incomplete physical evaluations"
        )

    if df.duplicated([
        "scenario_id",
        "run_index",
        "variant",
        "profile",
        "attack_severity",
    ]).any():
        raise RuntimeError(
            "Duplicate physical evaluations"
        )

    # Append pre-attack operational violations
    # from the saved Exp33 dispatch audit.
    baseline = dispatch_summary[[
        "case_size",
        "profile",
        "voltage_violations",
        "Pg_violations",
        "Qg_violations",
    ]].rename(columns={
        "voltage_violations": (
            "dispatch_baseline_voltage_violations"
        ),
        "Pg_violations": (
            "dispatch_baseline_Pg_violations"
        ),
        "Qg_violations": (
            "dispatch_baseline_Qg_violations"
        ),
    })

    df = df.merge(
        baseline,
        on=["case_size", "profile"],
        how="left",
        validate="many_to_one",
    )

    df.to_csv(
        run_dir / "physical_per_run.csv",
        index=False,
    )

    aggregate = (
        df.groupby(
            ["case_size", "variant"],
            as_index=False,
        )
        .agg(
            n_rows=(
                "modeled_unserved_fraction", "size"
            ),
            mean_unserved_fraction=(
                "modeled_unserved_fraction", "mean"
            ),
            p95_unserved_fraction=(
                "modeled_unserved_fraction",
                lambda x: np.quantile(x, 0.95)
            ),
            baseline_pf_success=(
                "baseline_pf_converged", "mean"
            ),
            attacked_pf_success=(
                "attack_pf_converged", "mean"
            ),
            mean_new_voltage_violations=(
                "new_voltage_violations", "mean"
            ),
            mean_new_line_overloads=(
                "new_line_overloads", "mean"
            ),
            mean_baseline_Qg_violations=(
                "dispatch_baseline_Qg_violations",
                "mean"
            ),
        )
        .sort_values([
            "case_size",
            "mean_unserved_fraction",
        ])
    )

    aggregate.to_csv(
        run_dir / "physical_aggregate.csv",
        index=False,
    )

    condition = (
        df.groupby(
            [
                "case_size",
                "profile",
                "attack_severity",
                "variant",
            ],
            as_index=False,
        )
        .agg(
            mean_unserved_fraction=(
                "modeled_unserved_fraction", "mean"
            ),
            baseline_pf_success=(
                "baseline_pf_converged", "mean"
            ),
            attacked_pf_success=(
                "attack_pf_converged", "mean"
            ),
            mean_new_voltage_violations=(
                "new_voltage_violations", "mean"
            ),
        )
    )

    condition.to_csv(
        run_dir / "physical_by_condition.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("HISTORICAL REPLAY CHECK")
    print("=" * 75)
    print(
        f"{replay_checks}/{expected_replays} "
        f"PAS/failed records reproduced",
        flush=True,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 34 - PHYSICAL EVALUATION RESULTS")
    print("=" * 75)
    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 34 - BY OPERATING CONDITION")
    print("=" * 75)
    print(
        condition.to_string(index=False),
        flush=True,
    )

    print(
        "\nRuntime:",
        round(
            time.perf_counter() - started,
            2,
        ),
        "seconds",
        flush=True,
    )

    print(
        "EXPERIMENT 34 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
