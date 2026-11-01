from pathlib import Path
import importlib
import json
import hashlib
import zipfile
import traceback

import numpy as np
import pandas as pd

from pypower.runopf import runopf
from pypower.ppoption import ppoption
from pypower.idx_bus import (
    BUS_I, PD, QD, VM, VA, VMIN, VMAX
)
from pypower.idx_gen import (
    GEN_BUS, GEN_STATUS, PG, QG,
    PMIN, PMAX, QMIN, QMAX, VG
)
from pypower.idx_brch import RATE_A

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "revision_experiments"
R = EXP / "results"

OUT = R / "46_final_closeout"
OUT.mkdir(parents=True, exist_ok=True)

FULL45 = (
    R / "45_acopf_corrective_dispatch"
    / "full_summary.csv"
)

STAGED = R / "34_staged_dispatch"

DIFFICULT = [
    (57, "Evening-Peak"),
    (300, "Evening-Peak"),
]

SCALES = [
    1.00, 0.95, 0.90, 0.85, 0.80,
    0.75, 0.70, 0.60, 0.50, 0.40,
    0.30, 0.20, 0.10, 0.00
]


def digest(p):
    return hashlib.sha256(
        Path(p).read_bytes()
    ).hexdigest()


def load_case(n, profile, scale):
    mod = importlib.import_module(
        f"pypower.case{n}"
    )

    case = getattr(
        mod, f"case{n}"
    )()

    file = (
        STAGED
        / f"case{n}"
        / f"{profile}.npz"
    )

    if not file.exists():
        raise FileNotFoundError(file)

    with np.load(
        file, allow_pickle=False
    ) as x:
        pd0 = np.asarray(
            x["bus_pd"], dtype=float
        )
        qd0 = np.asarray(
            x["bus_qd"], dtype=float
        )
        vm0 = np.asarray(
            x["bus_vm"], dtype=float
        )
        va0 = np.asarray(
            x["bus_va"], dtype=float
        )
        pg0 = np.asarray(
            x["gen_pg"], dtype=float
        )

    if len(pd0) != len(case["bus"]):
        raise RuntimeError(
            "Bus dimension mismatch"
        )

    if len(pg0) != len(case["gen"]):
        raise RuntimeError(
            "Generator dimension mismatch"
        )

    for v in [pd0, qd0, vm0, va0, pg0]:
        if not np.isfinite(v).all():
            raise RuntimeError(
                "Nonfinite staged operating point"
            )

    # Curtail only positive net active demand.
    # Preserve negative net injections.
    positive_load = pd0 > 0

    new_pd = pd0.copy()
    new_qd = qd0.copy()

    new_pd[positive_load] *= scale
    new_qd[positive_load] *= scale

    case["bus"][:, PD] = new_pd
    case["bus"][:, QD] = new_qd
    case["bus"][:, VM] = vm0
    case["bus"][:, VA] = va0

    case["gen"][:, PG] = pg0

    bus_mapping = {
        int(bus_id): row
        for row, bus_id in enumerate(
            case["bus"][:, BUS_I]
        )
    }

    for g in case["gen"]:
        row = bus_mapping[
            int(g[GEN_BUS])
        ]
        g[VG] = vm0[row]

    original_positive_demand = float(
        np.maximum(pd0, 0).sum()
    )

    curtailed_mw = (
        original_positive_demand
        * (1.0 - scale)
    )

    return (
        case,
        original_positive_demand,
        curtailed_mw,
        file
    )


def evaluate(n, profile, scale):
    case, demand, shed_mw, file = (
        load_case(n, profile, scale)
    )

    rates = case["branch"][:, RATE_A]

    meaningful_ratings = (
        (rates > 0)
        & (rates < 9000)
    )

    online = (
        case["gen"][:, GEN_STATUS] > 0
    )

    result = {
        "case_size": n,
        "profile": profile,
        "load_scale": float(scale),
        "load_curtailment_fraction":
            float(1.0 - scale),
        "positive_demand_base_MW": demand,
        "curtailed_MW": shed_mw,
        "net_load_after_MW": float(
            case["bus"][:, PD].sum()
        ),
        "online_Pmax_MW": float(
            case["gen"][online, PMAX].sum()
        ),
        "online_Pmin_MW": float(
            case["gen"][online, PMIN].sum()
        ),
        "rated_branches": int(
            meaningful_ratings.sum()
        ),
        "total_branches": int(
            len(rates)
        ),
        "opf_success": False,
        "constraint_check_pass": False,
        "voltage_violations": np.nan,
        "Pg_violations": np.nan,
        "Qg_violations": np.nan,
        "dispatch_change_L1_MW": np.nan,
        "solver_error": "",
        "input_sha256": digest(file),
    }

    original_pg = (
        case["gen"][:, PG].copy()
    )

    try:
        solved = runopf(
            case,
            ppoption(
                VERBOSE=0,
                OUT_ALL=0
            )
        )

        success = bool(
            solved.get("success", False)
        )

        result["opf_success"] = success

        if not success:
            return result

        bus = solved["bus"]
        gen = solved["gen"]

        vbad = int(np.sum(
            (bus[:, VM] < bus[:, VMIN] - 1e-4)
            | (bus[:, VM] > bus[:, VMAX] + 1e-4)
        ))

        pbad = int(np.sum(
            (gen[:, PG] < gen[:, PMIN] - 1e-4)
            | (gen[:, PG] > gen[:, PMAX] + 1e-4)
        ))

        qbad = int(np.sum(
            (gen[:, QG] < gen[:, QMIN] - 1e-4)
            | (gen[:, QG] > gen[:, QMAX] + 1e-4)
        ))

        result.update({
            "voltage_violations": vbad,
            "Pg_violations": pbad,
            "Qg_violations": qbad,
            "dispatch_change_L1_MW": float(
                np.abs(
                    gen[:, PG] - original_pg
                ).sum()
            ),
            "constraint_check_pass": bool(
                vbad == 0
                and pbad == 0
                and qbad == 0
            )
        })

    except Exception as exc:
        result["solver_error"] = (
            type(exc).__name__
            + ": "
            + str(exc)[:400]
        )

    return result


def run_rescue():
    all_rows = []
    summaries = []

    for n, profile in DIFFICULT:
        print(
            f"\n{'='*60}"
            f"\nIEEE{n} / {profile}"
            f"\n{'='*60}",
            flush=True
        )

        trials = []

        for scale in SCALES:
            row = evaluate(
                n, profile, scale
            )

            trials.append(row)

            print(
                f"scale={scale:.2f} "
                f"OPF={row['opf_success']} "
                f"check={row['constraint_check_pass']} "
                f"shed={row['curtailed_MW']:.2f} MW",
                flush=True
            )

            # Examine highest feasible point first.
            if row["constraint_check_pass"]:
                break

        feasible = [
            r for r in trials
            if r["constraint_check_pass"]
        ]

        if feasible:
            best = max(
                feasible,
                key=lambda x: x["load_scale"]
            )

            lower = best["load_scale"]

            # Only refine if an infeasible tested
            # point exists above the feasible point.
            higher_failed = sorted([
                r["load_scale"]
                for r in trials
                if (
                    not r["constraint_check_pass"]
                    and r["load_scale"] > lower
                )
            ])

            if higher_failed:
                upper = higher_failed[0]

                for _ in range(7):
                    middle = (
                        lower + upper
                    ) / 2

                    row = evaluate(
                        n, profile, middle
                    )

                    trials.append(row)

                    print(
                        f"refine={middle:.5f} "
                        f"OPF={row['opf_success']}",
                        flush=True
                    )

                    if row["constraint_check_pass"]:
                        lower = middle
                        best = row
                    else:
                        upper = middle

            summaries.append({
                "case_size": n,
                "profile": profile,
                "feasible_point_found": True,
                "highest_tested_feasible_load_scale":
                    best["load_scale"],
                "load_curtailment_fraction":
                    best["load_curtailment_fraction"],
                "curtailed_MW":
                    best["curtailed_MW"],
                "voltage_violations":
                    best["voltage_violations"],
                "Pg_violations":
                    best["Pg_violations"],
                "Qg_violations":
                    best["Qg_violations"],
                "rated_branches":
                    best["rated_branches"],
                "attempts": len(trials),
            })

        else:
            summaries.append({
                "case_size": n,
                "profile": profile,
                "feasible_point_found": False,
                "highest_tested_feasible_load_scale":
                    np.nan,
                "load_curtailment_fraction":
                    np.nan,
                "curtailed_MW": np.nan,
                "voltage_violations": np.nan,
                "Pg_violations": np.nan,
                "Qg_violations": np.nan,
                "rated_branches": int(
                    trials[0]["rated_branches"]
                ),
                "attempts": len(trials),
            })

        all_rows.extend(trials)

        pd.DataFrame(all_rows).to_csv(
            OUT / "46_opf_recovery_trials.csv",
            index=False
        )

        pd.DataFrame(summaries).to_csv(
            OUT / "46_opf_recovery_summary.csv",
            index=False
        )

    return pd.DataFrame(summaries)


def verify_existing():
    if not FULL45.exists():
        raise FileNotFoundError(
            FULL45
        )

    df = pd.read_csv(FULL45)

    if len(df) != 12:
        raise RuntimeError(
            "Exp45 is incomplete"
        )

    expected = {
        (n, p)
        for n in [14, 57, 118, 300]
        for p in [
            "Morning",
            "Solar-Peak",
            "Evening-Peak"
        ]
    }

    found = set(
        zip(
            df["case_size"],
            df["profile"]
        )
    )

    if found != expected:
        raise RuntimeError(
            "Exp45 profile coverage mismatch"
        )

    ok = df["opf_success"].astype(
        bool
    )

    print("\nEXP45 ORIGINAL OPF")
    print(
        "Successful:",
        int(ok.sum()),
        "/12"
    )

    print("Unsuccessful:")
    print(
        df.loc[
            ~ok,
            ["case_size", "profile"]
        ].to_string(index=False)
    )

    return df


def build_package():
    archive = (
        EXP /
        "TNSE_FINAL_EXPERIMENTS_CLOSEOUT.zip"
    )

    selected = []

    patterns = [
        "02_safeguard_full*_parameter_aggregate.csv",
        "03_ablation_full*_aggregate.csv",
        "03_ablation_full*_per_run.csv",
        "08_observation_full*_aggregate.csv",
        "09_observation_audit*_aggregate.csv",
        "18_objective_weights_full/aggregate.csv",
        "19_dynamic_risk_full/aggregate.csv",
        "23_calibration_full/aggregate.csv",
        "26_multiscale_full/aggregate.csv",
        "34_redispatched_physical_v2_full/physical_per_run.csv",
        "35_37_final_evidence/*.csv",
        "38_same_state_full/aggregate.csv",
        "38_same_state_full/paired_statistics.csv",
        "39_mechanism_full/condition_summary.csv",
        "39_mechanism_full/clean_paired_statistics.csv",
        "40_paired_observation_full/paired_stress_statistics.csv",
        "42_final_guard_audit/*.csv",
        "44_terminal_gate/*.csv",
        "45_acopf_corrective_dispatch/full_summary.csv",
        "46_final_closeout/*.csv",
    ]

    for pattern in patterns:
        selected.extend(
            p for p in R.glob(pattern)
            if p.is_file()
        )

    for name in [
        "44_final_gate_terminal.py",
        "45_acopf_feasibility.py",
        "46_final_closeout.py",
    ]:
        p = EXP / name
        if p.exists():
            selected.append(p)

    fig_dir = (
        EXP / "figures/final_revision"
    )

    selected.extend(
        fig_dir.glob("*.pdf")
    )

    old_archive = (
        EXP /
        "TNSE_FINAL_EVIDENCE_PACKAGE.zip"
    )

    if old_archive.exists():
        selected.append(old_archive)

    selected = sorted(
        set(selected),
        key=str
    )

    manifest = {
        "purpose": (
            "Final TNSE experiment evidence "
            "and corrective OPF recovery analysis"
        ),
        "source_files": {
            str(p.relative_to(ROOT)): digest(p)
            for p in selected
        },
        "limitations": [
            "OPF solver failure does not prove mathematical infeasibility.",
            "Load shedding creates a different corrected operating point.",
            "No trustworthy rated line thermal-limit coverage was verified.",
            "Strict original closed-loop MPC implementation remains to be reconciled.",
            "Original EP and manuscript weights require alignment.",
            "No result may be reported as statistically significant unless supported."
        ]
    }

    manifest_path = (
        OUT / "46_package_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    selected.append(manifest_path)

    with zipfile.ZipFile(
        archive,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:
        for p in selected:
            z.write(
                p,
                arcname=str(
                    p.relative_to(ROOT)
                )
            )

    print("\nFINAL PACKAGE CREATED")
    print("Files:", len(selected))
    print("Path:", archive)
    print(
        "Size:",
        round(
            archive.stat().st_size
            / (1024*1024), 2
        ),
        "MB"
    )


def main():
    verify_existing()

    result = run_rescue()

    if len(result) != 2:
        raise RuntimeError(
            "Expected two difficult profiles"
        )

    print("\n" + "="*65)
    print("EXP46 CORRECTIVE OPF RECOVERY RESULTS")
    print("="*65)

    print(
        result.to_string(index=False),
        flush=True
    )

    build_package()

    print("\nEXP46 FINAL CLOSEOUT COMPLETED")


if __name__ == "__main__":
    main()
