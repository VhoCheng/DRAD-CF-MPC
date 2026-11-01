from pathlib import Path
import argparse
import contextlib
import copy
import json
import io
import time
import warnings

import numpy as np
import pandas as pd

from pypower.api import (
    case14, case57, case118, case300,
    runpf, ppoption,
)

from pypower.idx_bus import (
    BUS_I, BUS_TYPE, PD, QD,
    VM, VA, VMIN, VMAX, REF,
)

from pypower.idx_gen import (
    GEN_BUS, GEN_STATUS,
    PG, QG, PMAX, PMIN, QMAX, QMIN,
)

ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "revision_experiments/results/33_dispatch_pf"
)

CASES = {
    14: case14,
    57: case57,
    118: case118,
    300: case300,
}

PROFILES = {
    "Morning": (0.85, 0.05),
    "Solar-Peak": (1.05, 0.30),
    "Evening-Peak": (1.20, 0.02),
}


def make_profile(n, name):
    load_factor, solar_fraction = PROFILES[name]

    case = CASES[n]()

    original_pd = case["bus"][:, PD].copy()
    original_qd = case["bus"][:, QD].copy()

    positive_pd = np.maximum(original_pd, 0.0)
    positive_qd = np.maximum(original_qd, 0.0)

    # Preserve negative demand entries in the case.
    case["bus"][:, PD] = (
        original_pd
        + (load_factor - 1.0) * positive_pd
    )

    case["bus"][:, QD] = (
        original_qd
        + (load_factor - 1.0) * positive_qd
    )

    # Synthetic distributed PV:
    # active-power injection at load buses,
    # represented as negative active demand.
    eligible = np.where(
        positive_pd > 1e-9
    )[0]

    if len(eligible):
        weights = (
            positive_pd[eligible]
            / positive_pd[eligible].sum()
        )

        pv_mw = (
            positive_pd.sum() * solar_fraction
        )

        case["bus"][eligible, PD] -= (
            pv_mw * weights
        )

    return case


def solve(case, warm=None, algorithm=1):
    ppc = copy.deepcopy(case)

    if warm is not None:
        if not np.array_equal(
            ppc["bus"][:, BUS_I],
            warm["bus"][:, BUS_I],
        ):
            raise RuntimeError(
                "Warm-start bus IDs mismatch"
            )

        ppc["bus"][:, VM] = (
            warm["bus"][:, VM]
        )

        ppc["bus"][:, VA] = (
            warm["bus"][:, VA]
        )

    options = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=algorithm,
        PF_MAX_IT=80,
        PF_MAX_IT_FD=160,
        PF_TOL=1e-7,
    )

    started = time.perf_counter()

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter("always")

        try:
            with contextlib.redirect_stdout(
                io.StringIO()
            ):
                solved, success = runpf(
                    ppc, options
                )

            finite = np.all(
                np.isfinite(
                    solved["bus"][:, [VM, VA]]
                )
            )

            converged = bool(success and finite)

            error = (
                ""
                if converged
                else "Unconverged or nonfinite voltage"
            )

        except Exception as exc:
            solved = None
            converged = False
            error = (
                f"{type(exc).__name__}: {exc}"
            )[:200]

    record = {
        "converged": int(converged),
        "solver_algorithm": algorithm,
        "seconds": round(
            time.perf_counter() - started,
            4,
        ),
        "warnings": "|".join(sorted(set(
            type(w.message).__name__
            for w in caught
        ))),
        "error": error,
    }

    return record, (
        solved if converged else None
    )


def allocate_generation(original, target, solved_base):
    gen = target["gen"]

    baseline_pg = (
        solved_base["gen"][:, PG].copy()
    )

    online = gen[:, GEN_STATUS] > 0

    slack_buses = set(
        int(x)
        for x in original["bus"][
            original["bus"][:, BUS_TYPE] == REF,
            BUS_I,
        ]
    )

    non_slack = np.array([
        int(bus) not in slack_buses
        for bus in gen[:, GEN_BUS]
    ])

    valid_limits = (
        np.isfinite(gen[:, PMAX])
        & np.isfinite(gen[:, PMIN])
        & (gen[:, PMAX] >= gen[:, PMIN])
    )

    baseline_within_limits = (
        (baseline_pg >= gen[:, PMIN] - 1e-5)
        & (baseline_pg <= gen[:, PMAX] + 1e-5)
    )

    eligible = (
        online
        & non_slack
        & valid_limits
        & baseline_within_limits
    )

    delta_load = float(np.sum(
        target["bus"][:, PD]
        - original["bus"][:, PD]
    ))

    direction = (
        1.0 if delta_load >= 0 else -1.0
    )

    if delta_load >= 0:
        capacity = np.where(
            eligible,
            np.maximum(
                gen[:, PMAX] - baseline_pg,
                0.0,
            ),
            0.0,
        )
    else:
        capacity = np.where(
            eligible,
            np.maximum(
                baseline_pg - gen[:, PMIN],
                0.0,
            ),
            0.0,
        )

    weights = np.where(
        eligible,
        np.maximum(
            np.abs(baseline_pg),
            1.0,
        ),
        0.0,
    )

    remaining = abs(delta_load)

    allocation = np.zeros(
        len(gen), dtype=float
    )

    for _ in range(len(gen) + 2):
        if remaining < 1e-7:
            break

        available = np.maximum(
            capacity - allocation,
            0.0,
        )

        active = available > 1e-9

        if not active.any():
            break

        w = weights * active

        if w.sum() <= 0:
            w = available

        proposal = (
            remaining * w / w.sum()
        )

        step = np.minimum(
            proposal,
            available,
        )

        amount = float(step.sum())

        if amount <= 1e-9:
            break

        allocation += step

        remaining = max(
            0.0,
            remaining - amount,
        )

    # Restore baseline operating-point generation
    # and apply bounded non-slack redispatch.
    gen[:, PG] = baseline_pg
    gen[:, PG] += direction * allocation

    report = {
        "net_load_change_mw": delta_load,
        "redispatched_mw": (
            direction * float(allocation.sum())
        ),
        "slack_residual_target_mw": (
            direction * remaining
        ),
        "available_headroom_mw": float(
            capacity.sum()
        ),
        "online_generators": int(
            online.sum()
        ),
        "eligible_generators": int(
            eligible.sum()
        ),
        "excluded_outside_limits": int(
            np.sum(
                online
                & non_slack
                & ~baseline_within_limits
            )
        ),
    }

    return target, report


def check_constraints(solution, original):
    if solution is None:
        return {
            "voltage_min": np.nan,
            "voltage_max": np.nan,
            "voltage_violations": np.nan,
            "Pg_violations": np.nan,
            "Qg_violations": np.nan,
            "slack_Pg_mw": np.nan,
        }

    bus = solution["bus"]
    gen = solution["gen"]
    gen_limits = original["gen"]

    voltage_bad = (
        (bus[:, VM] < bus[:, VMIN] - 1e-3)
        | (bus[:, VM] > bus[:, VMAX] + 1e-3)
    )

    online = gen[:, GEN_STATUS] > 0

    pg_bad = (
        (gen[:, PG] < gen_limits[:, PMIN] - 1e-4)
        | (gen[:, PG] > gen_limits[:, PMAX] + 1e-4)
    )

    qg_bad = (
        (gen[:, QG] < gen_limits[:, QMIN] - 1e-4)
        | (gen[:, QG] > gen_limits[:, QMAX] + 1e-4)
    )

    refs = set(
        int(x)
        for x in bus[
            bus[:, BUS_TYPE] == REF,
            BUS_I,
        ]
    )

    slack_pg = sum(
        float(gen[i, PG])
        for i in range(len(gen))
        if online[i]
        and int(gen[i, GEN_BUS]) in refs
    )

    return {
        "voltage_min": float(
            bus[:, VM].min()
        ),
        "voltage_max": float(
            bus[:, VM].max()
        ),
        "voltage_violations": int(
            voltage_bad.sum()
        ),
        "Pg_violations": int(
            np.sum(pg_bad & online)
        ),
        "Qg_violations": int(
            np.sum(qg_bad & online)
        ),
        "slack_Pg_mw": float(slack_pg),
    }


def continuation(original, solved_base, steps=20):
    if solved_base is None:
        return None

    previous = solved_base

    for step in range(1, steps + 1):
        # Target case is supplied through a global
        # variable local to the current process.
        alpha = step / steps

        intermediate = copy.deepcopy(original)

        intermediate["bus"][:, PD] = (
            original["bus"][:, PD]
            + alpha * (
                TARGET["bus"][:, PD]
                - original["bus"][:, PD]
            )
        )

        intermediate["bus"][:, QD] = (
            original["bus"][:, QD]
            + alpha * (
                TARGET["bus"][:, QD]
                - original["bus"][:, QD]
            )
        )

        intermediate, _ = allocate_generation(
            original,
            intermediate,
            solved_base,
        )

        solution = None

        for algorithm in (1, 2, 3):
            _, trial = solve(
                intermediate,
                warm=previous,
                algorithm=algorithm,
            )

            if trial is not None:
                solution = trial
                break

        if solution is None:
            return None

        previous = solution

    return previous


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        required=True,
    )

    args = parser.parse_args()

    sizes = (
        [118, 300]
        if args.mode == "pilot"
        else [14, 57, 118, 300]
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 75)
    print("TNSE EXPERIMENT 33")
    print("BOUNDED GENERATION REDISPATCH AND AC POWER FLOW")
    print("=" * 75)

    for n in sizes:
        original = CASES[n]()

        base_result, solved_base = solve(
            original
        )

        if solved_base is None:
            raise RuntimeError(
                f"IEEE{n} original AC PF failed"
            )

        for profile in PROFILES:
            target = make_profile(
                n, profile
            )

            # Reference: altered demand with
            # original generator dispatch.
            original_dispatch_result, _ = solve(
                target,
                warm=solved_base,
            )

            target, dispatch = allocate_generation(
                original,
                target,
                solved_base,
            )

            global TARGET
            TARGET = target

            solution = None
            method = "NONE"

            for name, algorithm, warm in [
                ("redispatch_newton", 1, None),
                ("redispatch_warm", 1, solved_base),
                ("redispatch_fast_XB", 2, solved_base),
                ("redispatch_fast_BX", 3, solved_base),
            ]:
                result, trial = solve(
                    target,
                    warm=warm,
                    algorithm=algorithm,
                )

                if trial is not None:
                    solution = trial
                    method = name
                    break

            if solution is None:
                solution = continuation(
                    original,
                    solved_base,
                    steps=20,
                )

                if solution is not None:
                    method = "redispatch_continuation"

            constraints = check_constraints(
                solution,
                original,
            )

            residual_ok = (
                abs(
                    dispatch[
                        "slack_residual_target_mw"
                    ]
                ) < 1.0
            )

            pg_ok = (
                solution is not None
                and constraints["Pg_violations"] == 0
            )

            # Q-limit violations are reported separately.
            # Standard runpf does not enforce all operational
            # constraints like an AC OPF solver.
            record = {
                "case_size": n,
                "profile": profile,
                "without_redispatch_converged": (
                    original_dispatch_result[
                        "converged"
                    ]
                ),
                "redispatch_converged": int(
                    solution is not None
                ),
                "solver_method": method,
                "net_demand_mw": float(
                    target["bus"][:, PD].sum()
                ),
                **dispatch,
                **constraints,
                "small_slack_residual": int(
                    residual_ok
                ),
                "Pg_limits_satisfied": int(
                    pg_ok
                ),
            }

            rows.append(record)

            if (
                solution is not None
                and residual_ok
                and pg_ok
            ):
                case_dir = (
                    OUT / f"case{n}"
                )

                case_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.savez_compressed(
                    case_dir / f"{profile}.npz",
                    bus_pd=target["bus"][:, PD],
                    bus_qd=target["bus"][:, QD],
                    gen_pg=target["gen"][:, PG],
                )

            print(
                f"[IEEE{n} {profile}] "
                f"converged={int(solution is not None)} "
                f"method={method} "
                f"residual="
                f"{dispatch['slack_residual_target_mw']:.2f} MW "
                f"Pg_bad={constraints['Pg_violations']} "
                f"Qg_bad={constraints['Qg_violations']}",
                flush=True,
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT / f"summary_{args.mode}.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 33 - AC PF DISPATCH RESULTS")
    print("=" * 75)

    print(
        df.to_string(index=False),
        flush=True,
    )

    print(
        "\nEXPERIMENT 33 COMPUTATION COMPLETED; "
        "CHECK CONVERGENCE AND LIMITS",
        flush=True,
    )


if __name__ == "__main__":
    main()
