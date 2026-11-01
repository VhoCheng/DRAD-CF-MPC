from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon

from pypower.api import (
    case14, case57, case118, case300,
    runpf, ppoption,
)
from pypower.idx_bus import (
    BUS_I, PD, QD, VM, VMIN, VMAX,
)
from pypower.idx_brch import (
    RATE_A, BR_STATUS, PF, QF, PT, QT,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "revision_experiments/results"
SOURCE = RESULTS / "26_multiscale_full"
NETWORKS = RESULTS / "25_multiscale_networks"

sys.path.insert(0, str(SRC))

import seid_malware as seid
import run_experiments_core as core
import drad_mpc as mpc
import dynamic_risk_defense as risk

CASE_FUNCTIONS = {
    14: case14,
    57: case57,
    118: case118,
    300: case300,
}

METHODS = [
    "CDP",
    "Legacy-K5",
    "Confirm15-Robust",
]

OPERATING_PROFILES = {
    "Morning": (0.85, 0.05),
    "Solar-Peak": (1.05, 0.30),
    "Evening-Peak": (1.20, 0.02),
}

ATTACK_SEVERITIES = [0.10, 0.30]

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(20261019)


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def validate_original_records(summary, details):
    if len(summary) != 240:
        raise RuntimeError(
            "Expected 240 Exp26 summary records"
        )

    if len(details) != 960:
        raise RuntimeError(
            "Expected 960 Exp26 per-run records"
        )

    if set(details["variant"]) != set(METHODS):
        raise RuntimeError(
            "Unexpected defense methods"
        )

    keys = [
        "scenario_id", "run_index", "variant"
    ]

    if details.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate historical observations"
        )

    counts = details.groupby(
        ["scenario_id", "variant"]
    )["run_index"].nunique()

    if not (counts == 4).all():
        raise RuntimeError(
            "Missing historical Monte Carlo runs"
        )

    for (sid, variant), sub in details.groupby(
        ["scenario_id", "variant"]
    ):
        ref = summary[
            (summary["scenario_id"] == sid)
            & (summary["variant"] == variant)
        ]

        if len(ref) != 1:
            raise RuntimeError(
                "Missing historical summary"
            )

        pas = float(sub["PAS"].mean())
        ps = 1.0 - float(sub["failed"].mean())
        ep = pas * ps

        for col, value in [
            ("PAS", pas),
            ("PS", ps),
            ("EP", ep),
        ]:
            if not np.isclose(
                value,
                float(ref.iloc[0][col]),
                atol=1e-10,
                rtol=0,
            ):
                raise RuntimeError(
                    f"History mismatch: {sid} "
                    f"{variant} {col}"
                )

    print(
        "EXPERIMENT 26 DATA RECONSTRUCTION PASSED",
        flush=True,
    )


def experiment27_statistics(summary, details, out_dir):
    result_rows = []

    for n in sorted(summary["case_size"].unique()):
        sm = summary[
            summary["case_size"] == n
        ]

        dt = details[
            details["case_size"] == n
        ]

        ids = sorted(
            sm["scenario_id"].unique()
        )

        if len(ids) != 20:
            raise RuntimeError(
                f"IEEE{n}: expected 20 scenarios"
            )

        pas_wide = dt.pivot(
            index=["scenario_id", "run_index"],
            columns="variant",
            values="PAS",
        )

        fail_wide = dt.pivot(
            index=["scenario_id", "run_index"],
            columns="variant",
            values="failed",
        )

        pas = np.stack([
            pas_wide.loc[
                [(sid, r) for r in range(4)],
                METHODS,
            ].to_numpy(dtype=float)
            for sid in ids
        ])

        failed = np.stack([
            fail_wide.loc[
                [(sid, r) for r in range(4)],
                METHODS,
            ].to_numpy(dtype=float)
            for sid in ids
        ])

        ep = (
            pas.mean(axis=1)
            * (1.0 - failed.mean(axis=1))
        )

        meta = (
            sm.drop_duplicates("scenario_id")
            .set_index("scenario_id")
            .loc[ids]
        )

        patterns = meta["pattern"].to_numpy()

        groups = [
            np.where(patterns == x)[0]
            for x in sorted(set(patterns))
        ]

        if (
            len(groups) != 5
            or any(len(g) != 4 for g in groups)
        ):
            raise RuntimeError(
                "Invalid coupling distribution"
            )

        # Paired, coupling-stratified bootstrap.
        draw_s = np.concatenate([
            RNG.choice(
                group,
                size=(N_BOOTSTRAP, len(group)),
                replace=True,
            )
            for group in groups
        ], axis=1)

        draw_r = RNG.integers(
            0, 4,
            size=(N_BOOTSTRAP, 20, 4),
        )

        sampled_pas = pas[
            draw_s[:, :, None],
            draw_r,
            :
        ].mean(axis=2)

        sampled_ps = (
            1.0
            - failed[
                draw_s[:, :, None],
                draw_r,
                :
            ].mean(axis=2)
        )

        sampled_ep = sampled_pas * sampled_ps

        idx = {
            x: i for i, x in enumerate(METHODS)
        }

        for method, baseline in [
            ("Legacy-K5", "CDP"),
            ("Confirm15-Robust", "CDP"),
            ("Confirm15-Robust", "Legacy-K5"),
        ]:
            a = idx[method]
            b = idx[baseline]

            # Positive = proposed method is better.
            gain = ep[:, b] - ep[:, a]

            boot_gain = (
                sampled_ep[:, :, b]
                - sampled_ep[:, :, a]
            ).mean(axis=1)

            low, high = np.quantile(
                boot_gain,
                [0.025, 0.975],
            )

            if np.all(np.abs(gain) < 1e-14):
                p = 1.0
            else:
                p = float(
                    wilcoxon(
                        gain,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )

            baseline_mean = float(ep[:, b].mean())

            result_rows.append({
                "case_size": n,
                "comparison": (
                    f"{method} vs {baseline}"
                ),
                "baseline_EP": baseline_mean,
                "method_EP": float(
                    ep[:, a].mean()
                ),
                "EP_gain": float(
                    gain.mean()
                ),
                "gain_pct_of_means": (
                    100.0 * gain.mean()
                    / baseline_mean
                    if baseline_mean > 1e-12
                    else np.nan
                ),
                "CI_low": float(low),
                "CI_high": float(high),
                "exploratory_wilcoxon_p": p,
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
            })

    result = pd.DataFrame(result_rows)

    result.to_csv(
        out_dir / "27_multiscale_paired_statistics.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 27 - MULTISCALE PAIRED STATISTICS")
    print("=" * 75)

    print(
        result.to_string(index=False),
        flush=True,
    )


def operating_case(case_size, profile):
    load_factor, solar_fraction = (
        OPERATING_PROFILES[profile]
    )

    ppc = CASE_FUNCTIONS[case_size]()

    original_pd = np.maximum(
        ppc["bus"][:, PD].astype(float),
        0.0,
    )

    original_qd = ppc["bus"][:, QD].astype(float)

    # Scale gross demand and reactive demand together.
    ppc["bus"][:, PD] = (
        original_pd * load_factor
    )

    ppc["bus"][:, QD] = (
        original_qd * load_factor
    )

    total_original_load = float(
        original_pd.sum()
    )

    # Synthetic behind-the-meter solar:
    # represented as negative active-power demand,
    # injected at the highest-load 20% of load buses.
    load_ids = np.where(
        original_pd > 1e-12
    )[0]

    if len(load_ids) > 0:
        ordered = sorted(
            load_ids,
            key=lambda i: (
                -original_pd[i], int(i)
            ),
        )

        n_solar = max(
            1,
            int(np.ceil(0.20 * len(ordered))),
        )

        solar_buses = np.array(
            ordered[:n_solar],
            dtype=int,
        )

        weights = original_pd[solar_buses]
        weights = weights / weights.sum()

        solar_mw = (
            total_original_load
            * solar_fraction
        )

        ppc["bus"][solar_buses, PD] -= (
            solar_mw * weights
        )

    return (
        ppc,
        original_pd,
        original_qd,
        load_factor,
        total_original_load,
    )


def solve_ac(ppc):
    options = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=1,
        PF_MAX_IT=50,
    )

    try:
        solved, success = runpf(
            copy.deepcopy(ppc),
            options,
        )
    except Exception as exc:
        return {
            "converged": False,
            "error": str(exc)[:200],
            "voltage_violating_buses": set(),
            "overloaded_branches": set(),
            "voltage_min": np.nan,
            "voltage_max": np.nan,
            "rated_branches": 0,
        }

    if not bool(success):
        return {
            "converged": False,
            "error": "PF did not converge",
            "voltage_violating_buses": set(),
            "overloaded_branches": set(),
            "voltage_min": np.nan,
            "voltage_max": np.nan,
            "rated_branches": 0,
        }

    buses = solved["bus"]
    branches = solved["branch"]

    vm = buses[:, VM]

    violation_mask = (
        (vm < buses[:, VMIN] - 1e-3)
        | (vm > buses[:, VMAX] + 1e-3)
    )

    violated_buses = {
        int(bus_id)
        for bus_id in buses[
            violation_mask, BUS_I
        ]
    }

    rate = branches[:, RATE_A]

    rated_mask = (
        (branches[:, BR_STATUS] > 0)
        & (rate > 1e-12)
    )

    apparent_from = np.hypot(
        branches[:, PF],
        branches[:, QF],
    )

    apparent_to = np.hypot(
        branches[:, PT],
        branches[:, QT],
    )

    apparent = np.maximum(
        apparent_from,
        apparent_to,
    )

    overloaded = {
        int(i)
        for i in np.where(
            rated_mask
            & (apparent > rate * 1.00001)
        )[0]
    }

    return {
        "converged": True,
        "error": "",
        "voltage_violating_buses": violated_buses,
        "overloaded_branches": overloaded,
        "voltage_min": float(vm.min()),
        "voltage_max": float(vm.max()),
        "rated_branches": int(
            rated_mask.sum()
        ),
    }


def replay_worker(task):
    job_id, rows = task

    # Each process has an isolated working directory
    # and Python module state.
    df = pd.DataFrame(rows)

    if df["scenario_id"].nunique() != 1:
        raise RuntimeError(
            "Worker received mixed scenario IDs"
        )

    first = df.iloc[0]

    n = int(first["case_size"])
    pattern = str(first["pattern"])
    info = float(first["info"])
    budget = int(first["budget"])
    onset = int(first["onset"])

    case_dir = NETWORKS / f"case{n}"
    table_dir = case_dir / "outputs/tables"

    if not table_dir.exists():
        raise RuntimeError(
            f"Missing case directory: {n}"
        )

    coupling = pd.read_csv(
        table_dir / f"coupling_{pattern}.csv"
    )

    comm_to_power = {
        int(a): int(b)
        for a, b in zip(
            coupling["comm_node"],
            coupling["power_node"],
        )
    }

    graph = nx.barabasi_albert_graph(
        n=n, m=2, seed=42
    )

    previous_dir = Path.cwd()
    old_risk_graph = risk.build_comm_graph

    def fixed_risk_graph(n_nodes=118, seed=42):
        return graph

    started = time.perf_counter()

    result_rows = []
    checks = 0

    try:
        os.chdir(case_dir)

        # Match Experiment 26 risk topology.
        risk.build_comm_graph = fixed_risk_graph

        baseline_profiles = {}

        for profile in OPERATING_PROFILES:
            (
                base,
                original_pd,
                original_qd,
                load_factor,
                original_total_load,
            ) = operating_case(n, profile)

            baseline_profiles[profile] = (
                base,
                original_pd,
                original_qd,
                load_factor,
                original_total_load,
                solve_ac(base),
            )

        for run_index, group in df.groupby(
            "run_index"
        ):
            if set(group["variant"]) != set(METHODS):
                raise RuntimeError(
                    "Incomplete historical methods"
                )

            seed_values = group["seed"].unique()

            if len(seed_values) != 1:
                raise RuntimeError(
                    "Historical seeds differ"
                )

            seed = int(seed_values[0])

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
                active_nodes=preview[
                    "active_infectious_nodes"
                ],
                exposed_nodes=preview[
                    "exposed_nodes"
                ],
                table_dir="outputs/tables",
                seed=seed,
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

            infection_cache = {}

            for _, original in group.iterrows():
                method = str(original["variant"])
                policy = str(
                    original["selected_policy"]
                )

                if policy not in candidates:
                    raise RuntimeError(
                        f"Missing policy: {policy}"
                    )

                protected = tuple(sorted(
                    int(x)
                    for x in candidates[policy]
                ))

                if len(protected) != budget:
                    raise RuntimeError(
                        "Defense budget mismatch"
                    )

                if protected not in infection_cache:
                    state = seid.simulate_seid_once(
                        Gc=graph,
                        onset_time=onset,
                        infection_rate=0.01,
                        activation_rate=0.15,
                        detection_rate=1e-5,
                        attack_information_ratio=info,
                        protected_nodes=list(protected),
                        strict_detection_failure=True,
                        seed=seed,
                    )

                    infection_cache[protected] = state

                state = infection_cache[protected]

                failed = bool(
                    state["attack_failed"]
                )

                proxy_pas = (
                    0.0
                    if failed
                    else mpc.evaluate_pas_proxy_mpc(
                        active_infectious_nodes=state[
                            "active_infectious_nodes"
                        ],
                        pattern=pattern,
                        table_dir="outputs/tables",
                    )
                )

                if (
                    int(failed)
                    != int(original["failed"])
                ):
                    raise RuntimeError(
                        "HISTORICAL FAILURE REPLAY FAILED: "
                        f"IEEE{n}, scenario={job_id}, "
                        f"run={run_index}, {method}"
                    )

                if not np.isclose(
                    proxy_pas,
                    float(original["PAS"]),
                    atol=1e-10,
                    rtol=0,
                ):
                    raise RuntimeError(
                        "HISTORICAL PAS REPLAY FAILED: "
                        f"IEEE{n}, scenario={job_id}, "
                        f"run={run_index}, {method}; "
                        f"replayed={proxy_pas}; "
                        f"reference={original['PAS']}"
                    )

                checks += 1

                infected = (
                    set()
                    if failed
                    else set(
                        map(
                            int,
                            state[
                                "active_infectious_nodes"
                            ],
                        )
                    )
                )

                affected_physical = {
                    comm_to_power[node]
                    for node in infected
                    if node in comm_to_power
                }

                for profile in OPERATING_PROFILES:
                    (
                        base,
                        original_pd,
                        original_qd,
                        load_factor,
                        original_total_load,
                        base_pf,
                    ) = baseline_profiles[profile]

                    gross_demand = (
                        original_total_load
                        * load_factor
                    )

                    for severity in ATTACK_SEVERITIES:
                        attacked = copy.deepcopy(base)

                        modeled_lost_load = 0.0

                        for physical_node in affected_physical:
                            lost_p = (
                                severity
                                * original_pd[physical_node]
                                * load_factor
                            )

                            lost_q = (
                                severity
                                * original_qd[physical_node]
                                * load_factor
                            )

                            attacked["bus"][
                                physical_node, PD
                            ] -= lost_p

                            attacked["bus"][
                                physical_node, QD
                            ] -= lost_q

                            modeled_lost_load += lost_p

                        if modeled_lost_load < -1e-10:
                            raise RuntimeError(
                                "Negative service interruption"
                            )

                        # If no load is interrupted, the
                        # attacked operating point is identical
                        # to the baseline operating point.
                        if modeled_lost_load <= 1e-12:
                            attack_pf = base_pf
                        else:
                            attack_pf = solve_ac(
                                attacked
                            )

                        both_converged = (
                            base_pf["converged"]
                            and attack_pf["converged"]
                        )

                        if both_converged:
                            new_v = len(
                                attack_pf[
                                    "voltage_violating_buses"
                                ]
                                - base_pf[
                                    "voltage_violating_buses"
                                ]
                            )

                            new_line = len(
                                attack_pf[
                                    "overloaded_branches"
                                ]
                                - base_pf[
                                    "overloaded_branches"
                                ]
                            )
                        else:
                            new_v = np.nan
                            new_line = np.nan

                        result_rows.append({
                            "scenario_id": job_id,
                            "case_size": n,
                            "pattern": pattern,
                            "info": info,
                            "budget": budget,
                            "onset": onset,
                            "run_index": int(run_index),
                            "seed": seed,
                            "variant": method,
                            "selected_policy": policy,
                            "profile": profile,
                            "attack_severity": severity,
                            "replayed_PAS": float(proxy_pas),
                            "historical_failed": int(failed),
                            "infectious_comm_nodes": len(infected),
                            "affected_power_buses": len(
                                affected_physical
                            ),
                            "gross_demand_mw": float(
                                gross_demand
                            ),
                            "modeled_unserved_mw": float(
                                modeled_lost_load
                            ),
                            "modeled_unserved_fraction": (
                                float(
                                    modeled_lost_load
                                    / gross_demand
                                )
                                if gross_demand > 1e-12
                                else np.nan
                            ),
                            "baseline_pf_converged": int(
                                base_pf["converged"]
                            ),
                            "attack_pf_converged": int(
                                attack_pf["converged"]
                            ),
                            "baseline_voltage_violations": (
                                len(
                                    base_pf[
                                        "voltage_violating_buses"
                                    ]
                                )
                                if base_pf["converged"]
                                else np.nan
                            ),
                            "attack_voltage_violations": (
                                len(
                                    attack_pf[
                                        "voltage_violating_buses"
                                    ]
                                )
                                if attack_pf["converged"]
                                else np.nan
                            ),
                            "new_voltage_violations": new_v,
                            "baseline_line_overloads": (
                                len(
                                    base_pf[
                                        "overloaded_branches"
                                    ]
                                )
                                if base_pf["converged"]
                                else np.nan
                            ),
                            "attack_line_overloads": (
                                len(
                                    attack_pf[
                                        "overloaded_branches"
                                    ]
                                )
                                if attack_pf["converged"]
                                else np.nan
                            ),
                            "new_line_overloads": new_line,
                            "rated_branches": (
                                base_pf["rated_branches"]
                            ),
                        })

    finally:
        risk.build_comm_graph = old_risk_graph
        os.chdir(previous_dir)

    return {
        "scenario_id": job_id,
        "replay_checks": checks,
        "records": result_rows,
        "runtime_seconds": (
            time.perf_counter() - started
        ),
    }


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
        choices=["pilot", "full"],
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    summary = pd.read_csv(
        SOURCE / "summary.csv"
    )

    details = pd.read_csv(
        SOURCE / "per_run.csv"
    )

    validate_original_records(
        summary,
        details,
    )

    run_dir = RESULTS / (
        f"27_29_joint_{args.mode}"
    )

    checkpoints = run_dir / "checkpoints"

    checkpoints.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exp27 uses the entire completed Exp26 dataset.
    experiment27_statistics(
        summary,
        details,
        run_dir,
    )

    groups = [
        (
            int(sid),
            sub.to_dict("records"),
        )
        for sid, sub in details.groupby(
            "scenario_id",
            sort=True,
        )
    ]

    if args.mode == "pilot":
        # One scenario from each IEEE scale.
        selected = {}

        for task in groups:
            n = int(
                task[1][0]["case_size"]
            )

            selected.setdefault(n, task)

        groups = [
            (
                sid,
                [
                    row for row in records
                    if int(row["run_index"]) == 0
                ],
            )
            for sid, records in selected.values()
        ]

        workers = 4

    else:
        workers = max(
            1,
            min(args.workers, 12),
        )

    manifest = {
        "experiment": "TNSE Joint Experiments 27-29",
        "mode": args.mode,
        "profiles": OPERATING_PROFILES,
        "attack_severities": ATTACK_SEVERITIES,
        "method_names": METHODS,
        "reference_exp26_per_run_sha256": sha(
            SOURCE / "per_run.csv"
        ),
        "driver_sha256": sha(
            Path(__file__).resolve()
        ),
        "source_hashes": {
            name: sha(SRC / name)
            for name in [
                "seid_malware.py",
                "run_experiments_core.py",
                "drad_mpc.py",
                "dynamic_risk_defense.py",
            ]
        },
        "scope": (
            "Historical defense replay; synthetic "
            "time-of-day gross load and behind-the-meter "
            "solar; 10/30 percent interruption of infected "
            "coupled-bus gross demand; post-attack AC PF."
        ),
        "case_table_hashes": {
            str(n): {
                name: sha(
                    NETWORKS
                    / f"case{n}"
                    / "outputs/tables"
                    / name
                )
                for name in [
                    "power_features.csv",
                    "comm_features.csv",
                    "coupling_DDAC.csv",
                    "coupling_DDDC.csv",
                    "coupling_DCAC.csv",
                    "coupling_DCDC.csv",
                    "coupling_RC.csv",
                ]
            }
            for n in CASE_FUNCTIONS
        },
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
                "Experiment protocol changed; "
                "cannot reuse checkpoints"
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

    for sid, records in groups:
        checkpoint = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        if not checkpoint.exists():
            pending.append(
                (sid, records)
            )

    print("\n" + "=" * 75, flush=True)
    print("TNSE JOINT EXPERIMENTS 27-29", flush=True)
    print("PAIRED STATISTICS / TIME-VARYING AC PF / PHYSICAL RISK")
    print("=" * 75, flush=True)

    print("Mode:", args.mode, flush=True)
    print("Workers:", workers, flush=True)
    print("Scenarios:", len(groups), flush=True)
    print("Pending:", len(pending), flush=True)
    print(
        "Operating profiles:",
        list(OPERATING_PROFILES),
        flush=True,
    )
    print(
        "Attack load-interruption fractions:",
        ATTACK_SEVERITIES,
        flush=True,
    )

    started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:

        futures = {
            pool.submit(replay_worker, task): task
            for task in pending
        }

        completed = (
            len(groups) - len(pending)
        )

        for future in as_completed(futures):
            task = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"AC PF REPLAY FAILED: "
                    f"scenario={task[0]}"
                ) from exc

            sid = task[0]

            expected_runs = (
                1 if args.mode == "pilot" else 4
            )

            expected_rows = (
                expected_runs
                * len(METHODS)
                * len(OPERATING_PROFILES)
                * len(ATTACK_SEVERITIES)
            )

            if (
                result["replay_checks"]
                != expected_runs * len(METHODS)
            ):
                raise RuntimeError(
                    "Incomplete historical replay"
                )

            if (
                len(result["records"])
                != expected_rows
            ):
                raise RuntimeError(
                    "Incomplete physical results"
                )

            path = (
                checkpoints
                / f"scenario_{sid:03d}.json"
            )

            save_checkpoint(
                path,
                result,
            )

            completed += 1

            print(
                f"[{completed}/{len(groups)}] "
                f"IEEE{task[1][0]['case_size']} "
                f"scenario={sid} "
                f"elapsed="
                f"{time.perf_counter()-started:.1f}s",
                flush=True,
            )

    all_rows = []
    total_checks = 0

    for sid, _ in groups:
        path = (
            checkpoints
            / f"scenario_{sid:03d}.json"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing checkpoint {sid}"
            )

        result = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if int(result["scenario_id"]) != sid:
            raise RuntimeError(
                "Checkpoint identity mismatch"
            )

        all_rows.extend(
            result["records"]
        )

        total_checks += int(
            result["replay_checks"]
        )

    df = pd.DataFrame(all_rows)

    expected_replays = (
        len(groups)
        * (1 if args.mode == "pilot" else 4)
        * len(METHODS)
    )

    if total_checks != expected_replays:
        raise RuntimeError(
            "Historical replay count mismatch"
        )

    print(
        "\nHISTORICAL PAS/FAILURE REPLAY PASSED:",
        f"{total_checks}/{expected_replays}",
        flush=True,
    )

    expected_physical_rows = (
        expected_replays
        * len(OPERATING_PROFILES)
        * len(ATTACK_SEVERITIES)
    )

    if len(df) != expected_physical_rows:
        raise RuntimeError(
            "Missing physical evaluation rows"
        )

    if df.duplicated([
        "scenario_id",
        "run_index",
        "variant",
        "profile",
        "attack_severity",
    ]).any():
        raise RuntimeError(
            "Duplicate physical observations"
        )

    result_path = run_dir / "28_29_physical_per_run.csv"

    df.to_csv(
        result_path,
        index=False,
    )

    grouped = (
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
            n_observations=(
                "modeled_unserved_fraction", "size"
            ),
            mean_unserved_fraction=(
                "modeled_unserved_fraction", "mean"
            ),
            p95_unserved_fraction=(
                "modeled_unserved_fraction",
                lambda x: np.quantile(x, 0.95)
            ),
            baseline_pf_convergence_rate=(
                "baseline_pf_converged", "mean"
            ),
            attack_pf_convergence_rate=(
                "attack_pf_converged", "mean"
            ),
            mean_new_voltage_violations=(
                "new_voltage_violations", "mean"
            ),
            mean_new_line_overloads=(
                "new_line_overloads", "mean"
            ),
        )
    )

    grouped_path = (
        run_dir / "28_29_physical_by_condition.csv"
    )

    grouped.to_csv(
        grouped_path,
        index=False,
    )

    overall = (
        df.groupby(
            [
                "case_size",
                "variant",
            ],
            as_index=False,
        )
        .agg(
            n_observations=(
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
        )
        .sort_values(
            [
                "case_size",
                "mean_unserved_fraction",
            ]
        )
    )

    overall_path = (
        run_dir / "28_29_physical_overall.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 28-29 - PHYSICAL EVALUATION RESULTS")
    print("=" * 75)

    print(
        overall.to_string(index=False),
        flush=True,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 28-29 - CONDITIONS (FIRST 20 ROWS)")
    print("=" * 75)

    print(
        grouped.head(20).to_string(index=False),
        flush=True,
    )

    print("\nSaved:", result_path)
    print("Saved:", grouped_path)
    print("Saved:", overall_path)

    print(
        "Total runtime:",
        round(
            time.perf_counter()-started,
            2,
        ),
        "seconds",
    )

    print(
        "\nPHYSICAL MODEL NOTE: "
        "Results describe a specified cyber-induced "
        "load-unavailability model and synthetic operating "
        "profiles, not measured grid incidents."
    )

    print(
        "JOINT EXPERIMENTS 27-29 "
        "COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
