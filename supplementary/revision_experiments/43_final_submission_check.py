from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "revision_experiments"
R = EXP / "results"
OUT = R / "final_submission_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Name, result pattern, required row count
SPECS = [
    ("03 Full ablation",
     "03_ablation_full*_per_run.csv", 3000),
    ("08 Observation robustness",
     "08_observation_full*_per_run.csv", 3600),
    ("09 Observation audit",
     "09_observation_audit*_per_run.csv", 3600),
    ("14 Independent replication",
     "14_replication_test/per_run.csv", 900),
    ("23 Independent calibration",
     "23_calibration_full/per_decision.csv", 320),
    ("26 Multiscale defense",
     "26_multiscale_full/per_run.csv", 960),
    ("34 Physical replay",
     "34_redispatched_physical_v2_full/physical_per_run.csv",
     5760),
    ("35 Physical statistics",
     "35_37_final_evidence/35_physical_paired_statistics.csv",
     12),
    ("36 Guard calibration",
     "35_37_final_evidence/36_guard_calibration_summary.csv",
     2),
    ("37 Serial timing",
     "35_37_final_evidence/37_serial_timing.csv", 32),
    ("38 Same-state validation",
     "38_same_state_full/per_run.csv", 2880),
    ("39 Mechanism study",
     "39_mechanism_full/per_decision.csv", 9600),
    ("40 Paired observation study",
     "40_paired_observation_full/per_action_paired.csv",
     9600),
    ("42 Guard trade-off",
     "42_final_guard_audit/42_guard_tradeoff_by_condition.csv",
     24),
    ("42 Tail risk",
     "42_final_guard_audit/42_tail_risk_and_regret.csv",
     12),
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def find(pattern):
    paths = sorted(
        p for p in R.glob(pattern) if p.is_file()
    )
    return paths[-1] if paths else None

inventory = []
included = set()
problems = []

print("=" * 72)
print("TNSE FINAL EXPERIMENT ACCEPTANCE")
print("=" * 72)

for name, pattern, expected in SPECS:
    path = find(pattern)

    if path is None:
        print(f"[MISSING] {name}")
        problems.append(name)
        continue

    df = pd.read_csv(path)
    passed = len(df) == expected

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{name}: {len(df)}/{expected}"
    )

    if not passed:
        problems.append(name)

    inventory.append({
        "experiment": name,
        "rows": len(df),
        "expected_rows": expected,
        "passed": passed,
        "file": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
    })

    included.add(path)

# Inspect actual parameter-study files.
print("\n" + "=" * 72)
print("PARAMETER SENSITIVITY INVENTORY")
print("=" * 72)

parameter_rows = []

for prefix, title in [
    ("02", "Safe-Guard parameter studies"),
    ("18", "Objective-weight sensitivity"),
    ("19", "Dynamic-risk-weight sensitivity"),
]:
    candidates = sorted(
        p for p in R.rglob("*.csv")
        if p.is_file()
        and (
            p.name.startswith(prefix + "_")
            or any(
                part.startswith(prefix + "_")
                for part in p.relative_to(R).parts
            )
        )
    )

    print(f"\n{title}: {len(candidates)} CSV files")

    if not candidates:
        parameter_rows.append({
            "category": title,
            "file": "",
            "rows": 0,
            "parameter": "",
            "tested_values": "NOT VERIFIED",
        })

    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print("[READ ERROR]", path, exc)
            continue

        included.add(path)
        print(
            " ", path.relative_to(R),
            "rows=", len(df)
        )

        found = False

        for col in df.columns:
            low = col.lower()

            relevant = (
                any(token in low for token in [
                    "gamma", "kappa",
                    "switch_margin", "confidence_k",
                    "lambda", "rho",
                    "weight"
                ])
            )

            if not relevant:
                continue

            vals = df[col].dropna().unique().tolist()

            if len(vals) > 24:
                desc = f"{len(vals)} distinct values"
            else:
                desc = str(vals)

            print("    ", col, ":", desc)
            found = True

            parameter_rows.append({
                "category": title,
                "file": str(path.relative_to(ROOT)),
                "rows": len(df),
                "parameter": col,
                "tested_values": desc,
            })

        if not found:
            parameter_rows.append({
                "category": title,
                "file": str(path.relative_to(ROOT)),
                "rows": len(df),
                "parameter": "(no explicit parameter column)",
                "tested_values": "Check config/source",
            })

# Inspect original full ablation variants.
print("\n" + "=" * 72)
print("ORIGINAL FULL ABLATION DEFINITIONS")
print("=" * 72)

ablation = find("03_ablation_full*_per_run.csv")

if ablation:
    df = pd.read_csv(ablation)

    for col in [
        "variant", "method", "ablation", "strategy"
    ]:
        if col in df.columns:
            print(col, "=", sorted(
                df[col].dropna().astype(str).unique().tolist()
            ))

    print("columns =", list(df.columns))

config_files = sorted(R.glob(
    "03_ablation_full*_config.json"
))

for config in config_files[-1:]:
    included.add(config)
    print("\nOriginal ablation config:")
    print(
        config.read_text(
            encoding="utf-8",
            errors="replace"
        )[:4000]
    )

# Validate key statistical outputs.
print("\n" + "=" * 72)
print("MULTIPLE-TEST-CORRECTION SUMMARY")
print("=" * 72)

statistic_specs = [
    (
        "Exp35",
        "35_37_final_evidence/35_physical_paired_statistics.csv",
        "holm_p_12_tests"
    ),
    (
        "Exp38",
        "38_same_state_full/paired_statistics.csv",
        "holm_adjusted_p"
    ),
    (
        "Exp39",
        "39_mechanism_full/clean_paired_statistics.csv",
        "holm_p_16"
    ),
    (
        "Exp40",
        "40_paired_observation_full/paired_stress_statistics.csv",
        "holm_p_20"
    ),
    (
        "Exp42",
        "42_final_guard_audit/42_guard_tradeoff_by_condition.csv",
        "holm_p_24"
    ),
]

stat_summary = []

for name, relative, pcol in statistic_specs:
    path = R / relative

    if not path.is_file():
        problems.append(name + " statistics")
        continue

    df = pd.read_csv(path)
    included.add(path)

    if pcol not in df.columns:
        problems.append(name + " adjusted p-values")
        continue

    significant = int(
        (df[pcol] < 0.05).sum()
    )

    print(
        f"{name}: {significant}/{len(df)} "
        "comparisons significant after correction"
    )

    stat_summary.append({
        "experiment": name,
        "comparisons": len(df),
        "significant_after_correction": significant,
    })

# Check existing figure exports.
print("\n" + "=" * 72)
print("PUBLICATION FIGURES")
print("=" * 72)

FIG = EXP / "figures/final_revision"

figures = sorted(
    p for p in FIG.glob("*")
    if p.is_file()
    and p.suffix.lower() in (".pdf", ".svg", ".png")
)

for p in figures:
    included.add(p)

print("Exported image files:", len(figures))
print("Distinct figures:", len({
    p.stem for p in figures
}))

if len({p.stem for p in figures}) < 6:
    problems.append("Six required figure families")

pd.DataFrame(inventory).to_csv(
    OUT / "final_experiment_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(parameter_rows).to_csv(
    OUT / "final_parameter_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(stat_summary).to_csv(
    OUT / "final_statistics_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

warnings = [
    "Exp39 NO_CF is not a strictly isolated "
    "single-module ablation.",
    "Exp38-40 implement a new post-observation "
    "same-state protocol; do not equate this "
    "with the original controller.",
    "Original manuscript risk weights must "
    "be reconciled with executed code.",
    "EP definition requires conditional vs "
    "unconditional PAS consistency.",
    "AC PF convergence does not imply "
    "OPF feasibility or dynamic stability.",
    "Exp37 is computation timing, not "
    "measured end-to-end network latency.",
    "Report non-significant findings and "
    "guard performance costs transparently.",
]

manifest = {
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "status": (
        "FILE_CHECK_PASSED" if not problems
        else "REQUIRES_REVIEW"
    ),
    "problems": problems,
    "inventory": inventory,
    "statistical_summary": stat_summary,
    "methodological_warnings": warnings,
}

manifest_path = OUT / "final_acceptance_manifest.json"

manifest_path.write_text(
    json.dumps(
        manifest, ensure_ascii=False, indent=2
    ),
    encoding="utf-8"
)

included.add(manifest_path)

for name in [
    "final_experiment_inventory.csv",
    "final_parameter_inventory.csv",
    "final_statistics_inventory.csv",
]:
    included.add(OUT / name)

# Include experimental driver code for traceability.
for name in [
    "03_ablation.py",
    "08_observation_robustness.py",
    "18_objective_weight_sensitivity.py",
    "19_dynamic_risk_weights.py",
    "26_multiscale_defense.py",
    "38_same_state_validation.py",
    "39_final_mechanism_suite.py",
    "40_paired_observation_replay.py",
    "41_publication_figures.py",
    "42_final_guard_audit.py",
]:
    path = EXP / name
    if path.is_file():
        included.add(path)

archive = EXP / "TNSE_FINAL_EVIDENCE_PACKAGE.zip"

with zipfile.ZipFile(
    archive, "w", compression=zipfile.ZIP_DEFLATED
) as z:
    for path in sorted(included):
        if path.is_file():
            z.write(
                path,
                arcname=str(path.relative_to(ROOT))
            )

print("\n" + "=" * 72)
print("FINAL RESULT")
print("=" * 72)
print("Required output problems:", len(problems))
print("Figure families:", len({p.stem for p in figures}))
print("Package:", archive)
print("Package size:",
      round(archive.stat().st_size / 1024 / 1024, 2),
      "MB")
print("Manifest:", manifest_path)

if problems:
    print("\nITEMS REQUIRING REVIEW:")
    for item in problems:
        print(" -", item)
else:
    print("\nALL TRACKED EXPERIMENT FILE CHECKS PASSED")

print("\nFINAL TNSE EVIDENCE AUDIT COMPLETED")
