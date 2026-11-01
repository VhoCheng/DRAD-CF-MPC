from pathlib import Path
import sys
import time
import json
import itertools
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiments_core as core


# ============================================================
# Pilot settings
# ============================================================

GAMMAS = [0.00, 0.10, 0.20]
KAPPAS = [0.00, 0.50, 1.00]

SCENARIOS = [
    {
        "pattern": "DDAC",
        "defense_budget": 60,
        "onset_time": 150,
        "attack_information_ratio": 0.50,
    },
    {
        "pattern": "RC",
        "defense_budget": 40,
        "onset_time": 120,
        "attack_information_ratio": 0.50,
    },
]

N_RUNS = 3
N_CF = 5
SEED = 42


def evaluate_controller_with_parameters(
    scenario, gamma, kappa
):
    """
    Inject Safe-Guard parameters without modifying src files.

    All other settings remain unchanged.
    """

    original = core.select_drad_mpc_protected_nodes

    def parameterized_selector(*args, **kwargs):
        kwargs["switch_margin"] = float(gamma)
        kwargs["confidence_k"] = float(kappa)
        return original(*args, **kwargs)

    core.select_drad_mpc_protected_nodes = (
        parameterized_selector
    )

    try:
        result = core.run_single_setting(
            strategy="DRAD-CF-MPC",
            n_runs=N_RUNS,
            lookahead_runs=N_CF,
            infection_rate=0.01,
            activation_rate=0.15,
            detection_rate=1e-5,
            seed=SEED,
            **scenario,
        )
    finally:
        core.select_drad_mpc_protected_nodes = original

    return result


def evaluate_cdp(scenario):

    return core.run_single_setting(
        strategy="CDP",
        n_runs=N_RUNS,
        lookahead_runs=N_CF,
        infection_rate=0.01,
        activation_rate=0.15,
        detection_rate=1e-5,
        seed=SEED,
        **scenario,
    )


def main():

    print("=" * 75, flush=True)
    print("TNSE MAJOR REVISION", flush=True)
    print("EXPERIMENT 01: SAFE-GUARD SENSITIVITY PILOT", flush=True)
    print("=" * 75, flush=True)

    print("Gamma grid:", GAMMAS, flush=True)
    print("Kappa grid:", KAPPAS, flush=True)
    print("Monte Carlo:", N_RUNS, flush=True)
    print("Counterfactual rollouts:", N_CF, flush=True)
    print("Scenarios:", len(SCENARIOS), flush=True)

    output_dir = ROOT / "revision_experiments/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    rows = []

    total = (
        len(SCENARIOS)
        * len(GAMMAS)
        * len(KAPPAS)
    )

    task_id = 0

    for scenario_id, scenario in enumerate(
        SCENARIOS, start=1
    ):

        print(
            f"\nEvaluating CDP baseline: scenario {scenario_id}",
            flush=True,
        )

        baseline = evaluate_cdp(scenario)

        for gamma, kappa in itertools.product(
            GAMMAS, KAPPAS
        ):

            task_id += 1

            print(
                f"\n[{task_id}/{total}] "
                f"scenario={scenario_id} "
                f"gamma={gamma:.2f} "
                f"kappa={kappa:.2f}",
                flush=True,
            )

            start = time.perf_counter()

            result = evaluate_controller_with_parameters(
                scenario=scenario,
                gamma=gamma,
                kappa=kappa,
            )

            elapsed = time.perf_counter() - start

            for metric in ["EP", "PAS", "RT", "RI", "RE"]:
                if not np.isfinite(float(result[metric])):
                    raise RuntimeError(
                        f"Invalid metric {metric}: {result[metric]}"
                    )

            cdp_ep = float(baseline["EP"])
            drad_ep = float(result["EP"])

            if abs(cdp_ep) > 1e-12:
                ep_gain_pct = (
                    100.0 * (cdp_ep - drad_ep) / cdp_ep
                )
            else:
                ep_gain_pct = float("nan")

            safe_guard = result["safe_guard_mode"]
            selected_policy = result[
                "selected_policy_mode"
            ]

            row = {
                "scenario_id": scenario_id,
                **scenario,

                "gamma": gamma,
                "kappa": kappa,

                "n_runs": N_RUNS,
                "n_cf": N_CF,
                "seed": SEED,

                "CDP_EP": cdp_ep,
                "DRAD_EP": drad_ep,

                "CDP_PAS": float(baseline["PAS"]),
                "DRAD_PAS": float(result["PAS"]),

                "CDP_RT": float(baseline["RT"]),
                "DRAD_RT": float(result["RT"]),

                "CDP_RI": float(baseline["RI"]),
                "DRAD_RI": float(result["RI"]),

                "CDP_RE": float(baseline["RE"]),
                "DRAD_RE": float(result["RE"]),

                "EP_gain_pct": ep_gain_pct,

                "selected_policy_mode": selected_policy,
                "safe_guard_mode": safe_guard,

                "wall_time_sec": elapsed,
            }

            rows.append(row)

            print(
                f"CDP_EP={cdp_ep:.6f} | "
                f"DRAD_EP={drad_ep:.6f} | "
                f"Gain={ep_gain_pct:.2f}% | "
                f"Policy={selected_policy} | "
                f"Guard={safe_guard} | "
                f"Time={elapsed:.2f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)

    csv_path = (
        output_dir
        / f"01_safeguard_pilot_{timestamp}.csv"
    )

    df.to_csv(csv_path, index=False)

    metadata = {
        "experiment": "Safe-Guard sensitivity pilot",
        "status": "pilot_not_publication_statistics",
        "gamma_grid": GAMMAS,
        "kappa_grid": KAPPAS,
        "scenarios": SCENARIOS,
        "n_runs": N_RUNS,
        "n_cf": N_CF,
        "seed": SEED,
        "python": platform.python_version(),
        "timestamp_utc": timestamp,
    }

    json_path = (
        output_dir
        / f"01_safeguard_pilot_{timestamp}.json"
    )

    json_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("PILOT RESULTS")
    print("=" * 75)

    print(
        df[
            [
                "scenario_id",
                "gamma",
                "kappa",
                "CDP_EP",
                "DRAD_EP",
                "EP_gain_pct",
                "selected_policy_mode",
                "safe_guard_mode",
            ]
        ].to_string(index=False)
    )

    print("\nSaved CSV:", csv_path)
    print("Saved metadata:", json_path)

    print("\nEXPERIMENT 01 PILOT COMPLETED")


if __name__ == "__main__":
    main()
