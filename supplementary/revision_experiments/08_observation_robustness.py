from pathlib import Path
from datetime import datetime, timezone
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiments_core as core

OUT = ROOT / "revision_experiments/results"
OUT.mkdir(parents=True, exist_ok=True)

PILOT_SCENARIOS = [
    ("DDAC", 0.50, 60, 150),
    ("RC", 0.50, 40, 120),
]

FULL_SCENARIOS = list(product(
    ["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
    [0.25, 1.00],
    [40, 80],
    [90, 210],
))

MISSING_RATES = [0.0, 0.20, 0.40]
DELAYS = [0, 15, 30]

BETA = 0.01
ALPHA = 0.15
DELTA = 1e-5

LOOKAHEAD = 5


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_original_source():
    names = [
        "run_experiments_core.py",
        "drad_mpc.py",
        "seid_malware.py",
        "dynamic_risk_defense.py",
    ]

    for name in names:
        current = ROOT / "src" / name
        original = ROOT.parent / "DRAD-CF-MPC" / "src" / name

        if not current.exists() or not original.exists():
            raise RuntimeError(
                f"Missing original source: {name}"
            )

        if sha256(current) != sha256(original):
            raise RuntimeError(
                f"Source mismatch: {name}"
            )

    print(
        "ORIGINAL SOURCE CHECK PASSED",
        flush=True
    )


def run_original(scenario, strategy, seed_argument):
    pattern, info, budget, onset = scenario

    return core.run_single_setting(
        pattern=pattern,
        strategy=strategy,
        attack_information_ratio=info,
        defense_budget=budget,
        onset_time=onset,
        n_runs=1,
        lookahead_runs=LOOKAHEAD,
        infection_rate=BETA,
        activation_rate=ALPHA,
        detection_rate=DELTA,
        seed=seed_argument,
    )


def worker(task):
    (
        scenario_id,
        scenario,
        missing_rate,
        delay_steps,
        n_runs,
        base_seed,
    ) = task

    pattern, info, budget, onset = scenario

    original_preview = core.get_preview_state

    records = []

    for r in range(n_runs):
        seed_argument = base_seed + r

        actual_seed = (
            seed_argument
            + onset * 1000
            + budget * 17
        )

        # CDP receives no dynamically estimated SEID state.
        # Therefore observation corruption does not alter CDP.
        baseline = run_original(
            scenario,
            "CDP",
            seed_argument,
        )

        observation_metadata = {}

        def imperfect_preview(
            Gc,
            onset_time,
            attack_information_ratio,
            infection_rate,
            activation_rate,
            detection_rate,
            seed,
        ):
            # Original current observation.
            truth = original_preview(
                Gc=Gc,
                onset_time=onset_time,
                attack_information_ratio=attack_information_ratio,
                infection_rate=infection_rate,
                activation_rate=activation_rate,
                detection_rate=detection_rate,
                seed=seed,
            )

            current_time = max(
                1, int(onset_time * 0.4)
            )

            observed_time = max(
                1, current_time - delay_steps
            )

            if delay_steps == 0:
                measured = truth

            else:
                # Observe the same underlying SEID process
                # at an earlier simulation time.
                measured = core.simulate_seid_once(
                    Gc=Gc,
                    onset_time=observed_time,
                    infection_rate=infection_rate,
                    activation_rate=activation_rate,
                    detection_rate=detection_rate,
                    attack_information_ratio=attack_information_ratio,
                    protected_nodes=None,
                    strict_detection_failure=False,
                    seed=seed,
                )

            observed = dict(measured)

            active = list(
                measured["active_infectious_nodes"]
            )

            exposed = list(
                measured["exposed_nodes"]
            )

            # Deterministic nested dropout:
            # 40% missingness includes the 20% omissions
            # for the same seed whenever thresholds permit.
            rng = np.random.default_rng(
                int(seed) + 550021
            )

            omit_mask = (
                rng.random(Gc.number_of_nodes())
                < missing_rate
            )

            active_observed = [
                int(node)
                for node in active
                if not omit_mask[int(node)]
            ]

            exposed_observed = [
                int(node)
                for node in exposed
                if not omit_mask[int(node)]
            ]

            observed["active_infectious_nodes"] = (
                active_observed
            )

            observed["exposed_nodes"] = (
                exposed_observed
            )

            observed["n_active_infectious"] = (
                len(active_observed)
            )

            observed["n_exposed"] = (
                len(exposed_observed)
            )

            observation_metadata.update({
                "nominal_preview_time": current_time,
                "actual_observed_time": observed_time,
                "true_active": len(
                    truth["active_infectious_nodes"]
                ),
                "true_exposed": len(
                    truth["exposed_nodes"]
                ),
                "delayed_active": len(active),
                "delayed_exposed": len(exposed),
                "observed_active": len(
                    active_observed
                ),
                "observed_exposed": len(
                    exposed_observed
                ),
                "omitted_states": (
                    len(active) + len(exposed)
                    - len(active_observed)
                    - len(exposed_observed)
                ),
            })

            if missing_rate == 0 and delay_steps == 0:
                if (
                    active_observed
                    != truth["active_infectious_nodes"]
                    or exposed_observed
                    != truth["exposed_nodes"]
                ):
                    raise RuntimeError(
                        "Zero-corruption preview mismatch."
                    )

            return observed

        # Patch only the defender's input state.
        # Actual propagation and CF simulation are untouched.
        core.get_preview_state = imperfect_preview

        try:
            result = run_original(
                scenario,
                "DRAD-CF-MPC",
                seed_argument,
            )

        finally:
            core.get_preview_state = original_preview

        if not observation_metadata:
            raise RuntimeError(
                "Observation wrapper was not called."
            )

        for name, values in [
            ("CDP", baseline),
            ("DRAD-CF-MPC", result),
        ]:
            for metric in [
                "PAS", "PS", "RT", "RI", "RE"
            ]:
                if not np.isfinite(
                    float(values[metric])
                ):
                    raise RuntimeError(
                        f"Invalid {name} {metric}"
                    )

        row = {
            "scenario_id": scenario_id,
            "pattern": pattern,
            "info": info,
            "budget": budget,
            "onset": onset,
            "missing_rate": missing_rate,
            "delay_steps": delay_steps,
            "run_index": r,
            "seed_argument": seed_argument,
            "actual_seed": actual_seed,
            "CDP_PAS": float(baseline["PAS"]),
            "CDP_PS": float(baseline["PS"]),
            "CDP_RT": float(baseline["RT"]),
            "CDP_RI": float(baseline["RI"]),
            "CDP_RE": float(baseline["RE"]),
            "DRAD_PAS": float(result["PAS"]),
            "DRAD_PS": float(result["PS"]),
            "DRAD_RT": float(result["RT"]),
            "DRAD_RI": float(result["RI"]),
            "DRAD_RE": float(result["RE"]),
            "selected_policy": result[
                "selected_policy_mode"
            ],
            "safe_guard": result[
                "safe_guard_mode"
            ],
            **observation_metadata,
        }

        row["paired_PAS_gain"] = (
            row["CDP_PAS"] - row["DRAD_PAS"]
        )

        row["negative_PAS_gain"] = int(
            row["paired_PAS_gain"] < -1e-12
        )

        row["switched"] = int(
            row["selected_policy"] != "CDP"
        )

        records.append(row)

    df = pd.DataFrame(records)

    cdp_pas = float(df["CDP_PAS"].mean())
    cdp_ps = float(df["CDP_PS"].mean())

    drad_pas = float(df["DRAD_PAS"].mean())
    drad_ps = float(df["DRAD_PS"].mean())

    cdp_ep = cdp_pas * cdp_ps
    drad_ep = drad_pas * drad_ps

    summary = {
        "scenario_id": scenario_id,
        "pattern": pattern,
        "info": info,
        "budget": budget,
        "onset": onset,
        "missing_rate": missing_rate,
        "delay_steps": delay_steps,
        "n_runs": n_runs,
        "lookahead_runs": LOOKAHEAD,
        "CDP_EP": cdp_ep,
        "DRAD_EP": drad_ep,
        "CDP_PAS": cdp_pas,
        "DRAD_PAS": drad_pas,
        "CDP_PS": cdp_ps,
        "DRAD_PS": drad_ps,
        "CDP_RT": float(df["CDP_RT"].mean()),
        "DRAD_RT": float(df["DRAD_RT"].mean()),
        "CDP_RI": float(df["CDP_RI"].mean()),
        "DRAD_RI": float(df["DRAD_RI"].mean()),
        "CDP_RE": float(df["CDP_RE"].mean()),
        "DRAD_RE": float(df["DRAD_RE"].mean()),
        "switch_rate": float(
            df["switched"].mean()
        ),
        "negative_PAS_gain_rate": float(
            df["negative_PAS_gain"].mean()
        ),
        "mean_true_active": float(
            df["true_active"].mean()
        ),
        "mean_observed_active": float(
            df["observed_active"].mean()
        ),
        "mean_omitted_states": float(
            df["omitted_states"].mean()
        ),
        "EP_gain_pct": (
            100.0 * (cdp_ep - drad_ep) / cdp_ep
            if cdp_ep > 1e-12
            else np.nan
        ),
    }

    return summary, records


def validate_pilot(summary):
    sources = sorted(
        OUT.glob(
            "03_ablation_pilot_*_summary.csv"
        ),
        key=lambda p: p.stat().st_mtime,
    )

    if not sources:
        raise FileNotFoundError(
            "Validated ablation pilot not found."
        )

    reference = pd.read_csv(sources[-1])

    clean = summary[
        (summary["missing_rate"] == 0)
        & (summary["delay_steps"] == 0)
    ]

    if len(clean) != 2:
        raise RuntimeError(
            "Expected two clean pilot scenarios."
        )

    for row in clean.itertuples():
        for variant, prefix in [
            ("CDP", "CDP"),
            ("Full", "DRAD"),
        ]:
            selected = reference[
                (reference["scenario_id"]
                 == row.scenario_id)
                & (reference["variant"] == variant)
            ]

            if len(selected) != 1:
                raise RuntimeError(
                    "Reference scenario missing."
                )

            expected = selected.iloc[0]

            for metric in [
                "EP", "PAS", "PS",
                "RT", "RI", "RE"
            ]:
                actual = getattr(
                    row, f"{prefix}_{metric}"
                )

                if not np.isclose(
                    actual,
                    float(expected[metric]),
                    atol=1e-10,
                    rtol=0,
                ):
                    raise RuntimeError(
                        "PILOT EQUIVALENCE FAILED: "
                        f"scenario={row.scenario_id}, "
                        f"variant={variant}, "
                        f"metric={metric}, "
                        f"new={actual}, "
                        f"old={expected[metric]}"
                    )

    print(
        "\nZERO-PERTURBATION EQUIVALENCE PASSED",
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
        default=6,
    )

    args = parser.parse_args()

    verify_original_source()

    if args.mode == "pilot":
        scenarios = PILOT_SCENARIOS
        n_runs = 3
        base_seed = 42
        workers = 2
    else:
        scenarios = FULL_SCENARIOS
        n_runs = 10
        base_seed = 20261022
        workers = max(
            1, min(args.workers, 8)
        )

    tasks = []

    for scenario_id, scenario in enumerate(
        scenarios, start=1
    ):
        for missing_rate, delay_steps in product(
            MISSING_RATES, DELAYS
        ):
            tasks.append((
                scenario_id,
                scenario,
                missing_rate,
                delay_steps,
                n_runs,
                base_seed,
            ))

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    prefix = (
        f"08_observation_{args.mode}_{stamp}"
    )

    summary_path = OUT / f"{prefix}_summary.csv"
    detail_path = OUT / f"{prefix}_per_run.csv"
    config_path = OUT / f"{prefix}_config.json"

    config = {
        "experiment": (
            "Imperfect defender-state observation"
        ),
        "mode": args.mode,
        "scenarios": scenarios,
        "missing_rates": MISSING_RATES,
        "delay_steps": DELAYS,
        "n_runs": n_runs,
        "lookahead_runs": LOOKAHEAD,
        "base_seed": base_seed,
        "workers": workers,
        "observation_time_definition": (
            "max(1, floor(0.4 * onset) - delay_steps)"
        ),
        "missingness_definition": (
            "Independent per-node omission of observed "
            "I and E state reports using a fixed "
            "seeded mask; true SEID dynamics unchanged"
        ),
        "source_hashes": {
            name: sha256(ROOT / "src" / name)
            for name in [
                "run_experiments_core.py",
                "drad_mpc.py",
                "seid_malware.py",
            ]
        },
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 72, flush=True)
    print("TNSE EXPERIMENT 08", flush=True)
    print("IMPERFECT OBSERVATION ROBUSTNESS", flush=True)
    print("=" * 72, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Scenarios:", len(scenarios), flush=True)
    print("Missingness:", MISSING_RATES, flush=True)
    print("Delays:", DELAYS, flush=True)
    print("Tasks:", len(tasks), flush=True)
    print("Monte Carlo:", n_runs, flush=True)
    print("Workers:", workers, flush=True)

    start = time.perf_counter()
    summaries = []
    completed = 0

    with ProcessPoolExecutor(
        max_workers=workers
    ) as pool:
        futures = {
            pool.submit(worker, task): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]

            try:
                summary, details = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"OBSERVATION TASK FAILED: {task}"
                ) from exc

            pd.DataFrame([summary]).to_csv(
                summary_path,
                mode="a",
                index=False,
                header=not summary_path.exists(),
            )

            pd.DataFrame(details).to_csv(
                detail_path,
                mode="a",
                index=False,
                header=not detail_path.exists(),
            )

            summaries.append(summary)
            completed += 1

            print(
                f"[{completed:03d}/{len(tasks)}] "
                f"scenario={task[0]} "
                f"missing={task[2]:.0%} "
                f"delay={task[3]} "
                f"EP_gain={summary['EP_gain_pct']:+.2f}% "
                f"switch={summary['switch_rate']:.1%} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(summaries)

    if len(df) != len(tasks):
        raise RuntimeError(
            "Incomplete observation experiment."
        )

    if df.duplicated(
        ["scenario_id", "missing_rate", "delay_steps"]
    ).any():
        raise RuntimeError(
            "Duplicate observation combinations."
        )

    if args.mode == "pilot":
        validate_pilot(df)

    aggregate = (
        df.groupby(
            ["missing_rate", "delay_steps"],
            as_index=False,
        )
        .agg(
            mean_CDP_EP=("CDP_EP", "mean"),
            mean_DRAD_EP=("DRAD_EP", "mean"),
            mean_EP_gain_pct=("EP_gain_pct", "mean"),
            mean_switch_rate=("switch_rate", "mean"),
            mean_negative_PAS_gain_rate=(
                "negative_PAS_gain_rate", "mean"
            ),
            mean_omitted_states=(
                "mean_omitted_states", "mean"
            ),
        )
    )

    aggregate_path = (
        OUT / f"{prefix}_aggregate.csv"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("OBSERVATION ROBUSTNESS RESULTS")
    print("=" * 72)

    print(
        aggregate.to_string(index=False),
        flush=True,
    )

    print("\nSummary:", summary_path)
    print("Per-run records:", detail_path)
    print("Aggregate:", aggregate_path)
    print("Configuration:", config_path)

    print(
        "Total runtime:",
        round(time.perf_counter()-start, 2),
        "seconds",
    )

    print(
        "EXPERIMENT 08 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
