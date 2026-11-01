from pathlib import Path
import argparse
import importlib
import json

import numpy as np
import pandas as pd

from pypower.ppoption import ppoption
from pypower.runopf import runopf

from pypower.idx_bus import (
    PD, QD, VM, VA, VMIN, VMAX, BUS_I
)
from pypower.idx_gen import (
    PG, QG, PMIN, PMAX, QMIN, QMAX, VG, GEN_BUS
)
from pypower.idx_brch import (
    RATE_A, PF, QF, PT, QT
)

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revision_experiments/results"
STAGED = R / "34_staged_dispatch"
OUT = R / "45_acopf_corrective_dispatch"

OUT.mkdir(parents=True, exist_ok=True)

CASES = [14, 57, 118, 300]
PROFILES = [
    "Morning",
    "Solar-Peak",
    "Evening-Peak"
]

def casefun(n):
    module = importlib.import_module(
        f"pypower.case{n}"
    )
    return getattr(module, f"case{n}")

def finite_violations(x, lo, hi, tol=1e-5):
    return int(np.sum(
        (x < lo - tol) | (x > hi + tol)
    ))

def run(n, profile):
    inp = (
        STAGED
        / f"case{n}"
        / f"{profile}.npz"
    )

    if not inp.exists():
        raise FileNotFoundError(inp)

    pp = casefun(n)()

    with np.load(
        inp, allow_pickle=False
    ) as x:
        data = {
            k: np.asarray(x[k], dtype=float)
            for k in [
                "bus_pd",
                "bus_qd",
                "bus_vm",
                "bus_va",
                "gen_pg"
            ]
        }

    for k in data:
        if not np.isfinite(data[k]).all():
            raise ValueError(
                f"{inp}: nonfinite {k}"
            )

    if (
        len(data["bus_pd"]) != len(pp["bus"])
        or len(data["gen_pg"]) != len(pp["gen"])
    ):
        raise ValueError(
            f"{inp}: dimension mismatch"
        )

    pp["bus"][:, PD] = data["bus_pd"]
    pp["bus"][:, QD] = data["bus_qd"]
    pp["bus"][:, VM] = data["bus_vm"]
    pp["bus"][:, VA] = data["bus_va"]

    pp["gen"][:, PG] = data["gen_pg"]

    bus_id_to_row = {
        int(v): i
        for i, v in enumerate(
            pp["bus"][:, BUS_I]
        )
    }

    for g in pp["gen"]:
        g[VG] = data["bus_vm"][
            bus_id_to_row[int(g[GEN_BUS])]
        ]

    rates = pp["branch"][:, RATE_A]

    ratings_defined = int(np.sum(
        (rates > 0) & (rates < 9000)
    ))

    ratings_placeholder = int(
        np.sum(rates >= 9000)
    )

    answer = {
        "case_size": n,
        "profile": profile,
        "input_file": str(
            inp.relative_to(ROOT)
        ),
        "input_bus_count": len(pp["bus"]),
        "input_gen_count": len(pp["gen"]),
        "branches_with_nonplaceholder_RATING_A": (
            ratings_defined
        ),
        "branches_with_placeholder_RATING_A": (
            ratings_placeholder
        ),
        "branches_without_RATING_A": int(
            np.sum(rates <= 0)
        ),
        "opf_success": False,
        "solver_error": "",
        "comment": (
            "Corrective redispatch permitted; "
            "not original fixed-dispatch feasibility."
        )
    }

    try:
        solved = runopf(
            pp,
            ppoption(
                VERBOSE=0,
                OUT_ALL=0
            )
        )

        ok = bool(solved["success"])
        answer["opf_success"] = ok

        if ok:
            bus = solved["bus"]
            gen = solved["gen"]
            branch = solved["branch"]

            rated = (
                (branch[:, RATE_A] > 0)
                & (branch[:, RATE_A] < 9000)
            )

            Sfrom = np.hypot(
                branch[:, PF],
                branch[:, QF]
            )

            Sto = np.hypot(
                branch[:, PT],
                branch[:, QT]
            )

            answer.update({
                "new_voltage_violations":
                    finite_violations(
                        bus[:, VM],
                        bus[:, VMIN],
                        bus[:, VMAX]
                    ),

                "new_Pg_violations":
                    finite_violations(
                        gen[:, PG],
                        gen[:, PMIN],
                        gen[:, PMAX]
                    ),

                "new_Qg_violations":
                    finite_violations(
                        gen[:, QG],
                        gen[:, QMIN],
                        gen[:, QMAX]
                    ),

                "rated_thermal_violations": int(
                    np.sum(
                        (
                            (Sfrom > branch[:, RATE_A] + 1e-4)
                            | (Sto > branch[:, RATE_A] + 1e-4)
                        ) & rated
                    )
                ),

                "generator_Pg_redispatch_L1_MW":
                    float(
                        np.sum(
                            abs(
                                gen[:, PG]
                                - data["gen_pg"]
                            )
                        )
                    ),

                "total_net_load_MW":
                    float(
                        bus[:, PD].sum()
                    ),

                "opf_objective":
                    float(
                        solved.get("f", np.nan)
                    )
            })

    except Exception as err:
        answer["solver_error"] = repr(err)[:450]

    return answer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["pilot", "full"],
    required=True
)

args = parser.parse_args()

tasks = [
    (n, p)
    for n in CASES
    for p in PROFILES
]

if args.mode == "pilot":
    tasks = [
        (14, "Morning"),
        (300, "Morning")
    ]

manifest = OUT / "manifest.json"

protocol = {
    "scope": (
        "Corrective AC OPF with staged synthetic "
        "net-demand profiles, not original fixed dispatch"
    ),
    "ratings_rule": (
        "0 or >=9000 is treated as an unspecified "
        "or placeholder rating for coverage reporting; "
        "input constraints remain unchanged"
    ),
    "cases": CASES,
    "profiles": PROFILES
}

if (
    manifest.exists()
    and json.loads(
        manifest.read_text()
    ) != protocol
):
    raise RuntimeError(
        "Protocol mismatch"
    )

manifest.write_text(
    json.dumps(protocol, indent=2)
)

output = OUT / f"{args.mode}_summary.csv"

records = []

for i, (n, profile) in enumerate(
    tasks, start=1
):
    row = run(n, profile)
    records.append(row)

    pd.DataFrame(records).to_csv(
        output,
        index=False
    )

    print(
        f"[{i}/{len(tasks)}] "
        f"IEEE{n} {profile} "
        f"OPF={row['opf_success']} "
        f"thermal_covered="
        f"{row['branches_with_nonplaceholder_RATING_A']} "
        f"error={row['solver_error']}",
        flush=True
    )

summary = pd.DataFrame(records)

print("\n=== AC OPF FEASIBILITY RESULTS ===")

print(summary[
    [
        "case_size",
        "profile",
        "opf_success",
        "branches_with_nonplaceholder_RATING_A",
        "solver_error"
    ]
].to_string(index=False))

print("\nEXP45 OPF CHECK COMPLETED")
print("Original results unchanged")
print("Output:", output)
