#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import ast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
R = ROOT / "revision_experiments/results"
OUT = R / "53_final_replication_audit"
OUT.mkdir(parents=True, exist_ok=True)

MODES = ["oracle", "partial80", "partial50", "blind"]
KEYS = ["cohort", "pattern", "budget", "onset", "seed"]
METHODS = ["CDP", "FULL", "STATE_LOCAL", "STATIC_EXTRA"]
COMPARISONS = [
    ("CDP", "STATE_LOCAL"),
    ("FULL", "STATE_LOCAL"),
    ("STATIC_EXTRA", "STATE_LOCAL")
]

C = {
    "blue": "#315C83",
    "teal": "#16817B",
    "orange": "#CB9049",
    "red": "#BF6062",
    "gray": "#93A4B0",
    "ink": "#263B4D"
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.edgecolor": "#B8C6CE",
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})


def export(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            bbox_inches="tight",
            pad_inches=.18,
            dpi=250
        )
    plt.close(fig)


def read_cohort(cohort, mode):
    if cohort == "Exp51":
        path = (
            R / "51_observation_access_v1" /
            f"full_{mode}" / "outcomes.csv"
        )
    else:
        path = (
            R / "52_reliability_gate" /
            f"test_{mode}" / "outcomes.csv"
        )

    if not path.exists():
        raise FileNotFoundError(path)

    d = pd.read_csv(path)

    required = set(KEYS[1:] + ["method", "PAS"])
    missing = required - set(d.columns)

    if missing:
        raise ValueError(
            f"{path}: missing {missing}"
        )

    d = d[d.method.isin(METHODS)].copy()
    d["cohort"] = cohort
    d["mode"] = mode

    if d.duplicated(KEYS + ["mode", "method"]).any():
        raise ValueError(
            f"Duplicate scenario-method rows: {path}"
        )

    if not np.isfinite(
        pd.to_numeric(d.PAS, errors="coerce")
    ).all():
        raise ValueError(
            f"Nonfinite PAS: {path}"
        )

    return d


def build_data():
    pieces = [
        read_cohort(cohort, mode)
        for cohort in ("Exp51", "Exp52")
        for mode in MODES
    ]

    df = pd.concat(
        pieces,
        ignore_index=True
    )

    for (cohort, mode), block in df.groupby(
        ["cohort", "mode"]
    ):
        wide = block.pivot(
            index=KEYS,
            columns="method",
            values="PAS"
        )

        if (
            set(wide.columns) != set(METHODS)
            or wide.isna().any().any()
        ):
            raise ValueError(
                f"Incomplete paired matrix: {cohort}/{mode}"
            )

        print(
            f"[INPUT] {cohort}/{mode}: "
            f"{len(wide)} complete paired scenarios",
            flush=True
        )

    return df


def bootstrap_effect(data, a, b, seed=202653):
    """
    Stratified paired bootstrap by cohort,
    pattern, budget and onset.

    Positive effect = PAS(a) - PAS(b).
    """

    w = data.pivot(
        index=KEYS,
        columns="method",
        values="PAS"
    )

    effect = (
        w[a] - w[b]
    ).rename("gain").reset_index()

    rng = np.random.default_rng(seed)
    B = 2500
    n = len(effect)

    samples = np.zeros(B)

    for _, group in effect.groupby(
        ["cohort", "pattern", "budget", "onset"],
        sort=True
    ):
        x = group.gain.to_numpy(float)

        ids = rng.integers(
            0, len(x),
            size=(B, len(x))
        )

        samples += (
            x[ids].sum(axis=1) / n
        )

    x = effect.gain.to_numpy(float)

    return {
        "n": len(x),
        "effect": float(x.mean()),
        "ci_low": float(np.quantile(samples, .025)),
        "ci_high": float(np.quantile(samples, .975)),
        "win_rate": float((x > 1e-12).mean()),
        "loss_rate": float((x < -1e-12).mean()),
        "tie_rate": float((abs(x) <= 1e-12).mean())
    }


def analyze(df):
    rows = []

    for mode in MODES:
        for cohort in ("Exp51", "Exp52", "Pooled"):
            if cohort == "Pooled":
                subset = df[df["mode"] == mode]
            else:
                subset = df[
                    (df["mode"] == mode) &
                    (df["cohort"] == cohort)
                ]

            for a, b in COMPARISONS:
                s = bootstrap_effect(subset, a, b)

                rows.append({
                    "mode": mode,
                    "cohort": cohort,
                    "comparison": f"{b} vs {a}",
                    **s
                })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "REPLICATION_PAIRED_CI.csv",
        index=False
    )

    performance = (
        df.groupby(["cohort", "mode", "method"])
        .agg(
            n=("PAS", "size"),
            mean_PAS=("PAS", "mean"),
            P95_PAS=("PAS", lambda x: x.quantile(.95))
        )
        .reset_index()
    )

    performance.to_csv(
        OUT / "COHORT_METHOD_PERFORMANCE.csv",
        index=False
    )

    return result, performance


def audit_gate():
    base = R / "52_reliability_gate"
    lock = json.loads(
        (base / "LOCKED_GATE.json").read_text()
    )

    threshold = float(lock["threshold"])

    rows = []

    for mode in MODES:
        p = base / f"test_{mode}" / "outcomes.csv"
        d = pd.read_csv(p)

        w = d.pivot(
            index=["pattern", "budget", "onset", "seed"],
            columns="method",
            values="PAS"
        )

        maxdiff = float(
            abs(
                w["QUALITY_GATE"] -
                w["STATE_LOCAL"]
            ).max()
        )

        rows.append({
            "mode": mode,
            "locked_threshold": threshold,
            "max_abs_gate_minus_local_PAS": maxdiff
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(
        OUT / "GATE_IDENTITY_AUDIT.csv",
        index=False
    )

    return threshold, audit


def source_audit():
    targets = [
        (
            ROOT / "tnse_exp51.py",
            ["observation", "observed_choose"]
        ),
        (
            ROOT / "tnse_exp50_v2.py",
            ["score_candidate", "candidate_wrapper"]
        ),
        (
            ROOT / "revision_experiments" /
            "47_final_closed_loop_validation.py",
            ["run_one", "predict", "choose"]
        )
    ]

    output = []

    for path, names in targets:
        if not path.exists():
            output.append(f"MISSING SOURCE: {path}")
            continue

        src = path.read_text()
        lines = src.splitlines()

        output.append(
            f"\nFILE: {path.relative_to(ROOT)}"
        )
        output.append(
            "SHA256: " +
            hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        )

        tree = ast.parse(src)

        for node in tree.body:
            if (
                isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and node.name in names
            ):
                output.append(
                    f"\nFUNCTION: {node.name}"
                )
                start = node.lineno - 1
                end = min(node.end_lineno, start + 65)

                for i in range(start, end):
                    output.append(
                        f"{i+1:04d}: {lines[i]}"
                    )

    (OUT / "SOURCE_AUDIT_EXCERPTS.txt").write_text(
        "\n".join(output),
        encoding="utf-8"
    )


def figure_forest(stats):
    sub = stats[
        stats.comparison == "STATE_LOCAL vs FULL"
    ].copy()

    fig, ax = plt.subplots(figsize=(10.8, 6.4))

    colors = {
        "Exp51": C["blue"],
        "Exp52": C["orange"],
        "Pooled": C["teal"]
    }

    labels = []

    for i, mode in enumerate(MODES):
        for j, cohort in enumerate(
            ("Exp51", "Exp52", "Pooled")
        ):
            r = sub[
                (sub["mode"] == mode) &
                (sub["cohort"] == cohort)
            ].iloc[0]

            y = i * 3.8 + j
            x = float(r.effect)

            ax.errorbar(
                x, y,
                xerr=[
                    [max(0, x-r.ci_low)],
                    [max(0, r.ci_high-x)]
                ],
                fmt="o",
                color=colors[cohort],
                ecolor=colors[cohort],
                capsize=3,
                markersize=5,
                lw=1.6
            )

            labels.append(
                (y, f"{mode} | {cohort}  (n={r.n})")
            )

    ax.set_yticks(
        [x[0] for x in labels],
        [x[1] for x in labels]
    )
    ax.invert_yaxis()
    ax.axvline(
        0, ls="--",
        color=C["red"], lw=1
    )

    ax.set_xlabel(
        "PAS(FULL) − PAS(STATE_LOCAL) | positive favors STATE_LOCAL"
    )
    ax.set_title(
        "Cross-cohort replication of state-aware benefit",
        loc="left",
        pad=16,
        fontweight="bold"
    )
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)

    fig.subplots_adjust(
        left=.28, right=.96,
        top=.90, bottom=.12
    )

    export(
        fig,
        "FigA_cross_cohort_replication"
    )


def figure_observation(perf):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.8, 4.6)
    )

    xs = np.arange(len(MODES))

    for ax, cohort, letter in zip(
        axes,
        ("Exp51", "Exp52"),
        ("a", "b")
    ):
        for method, color in [
            ("CDP", C["gray"]),
            ("FULL", C["blue"]),
            ("STATE_LOCAL", C["teal"])
        ]:
            v = (
                perf[
                    (perf.cohort == cohort) &
                    (perf.method == method)
                ]
                .set_index("mode")
                .reindex(MODES)
            )

            ax.plot(
                xs,
                v.mean_PAS,
                "-o",
                lw=1.9,
                markersize=5,
                color=color,
                label=method
            )

        ax.set_xticks(
            xs,
            ["100%", "80%", "50%", "0%"]
        )
        ax.set_xlabel("Simulated infection-state recall")
        ax.set_ylabel("Mean PAS (lower is better)")
        ax.set_title(
            f"({letter}) {cohort} independent seed family",
            loc="left", pad=13
        )
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(.5, .01),
        ncol=3,
        frameon=False,
        fontsize=9
    )

    fig.subplots_adjust(
        left=.08, right=.98,
        bottom=.27, top=.86,
        wspace=.25
    )

    export(
        fig,
        "FigB_observation_quality_replication"
    )


def figure_partial50(df):
    fig, ax = plt.subplots(
        figsize=(9.4, 4.5)
    )

    for cohort, color in [
        ("Exp51", C["blue"]),
        ("Exp52", C["orange"])
    ]:
        d = df[
            (df.cohort == cohort) &
            (df["mode"] == "partial50")
        ]

        w = d.pivot(
            index=KEYS,
            columns="method",
            values="PAS"
        )

        gains = np.sort(
            (
                w["FULL"] -
                w["STATE_LOCAL"]
            ).to_numpy(float)
        )

        ax.step(
            gains,
            np.arange(1, len(gains)+1)/len(gains),
            where="post",
            lw=2,
            color=color,
            label=f"{cohort}: n={len(gains)}"
        )

    ax.axvline(
        0,
        ls="--",
        color=C["red"],
        linewidth=1
    )

    ax.set_xlabel(
        "Paired PAS(FULL) − PAS(STATE_LOCAL)"
    )
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title(
        "50% recall: why independent cohorts differ",
        loc="left",
        pad=16,
        fontweight="bold"
    )
    ax.legend(
        frameon=False,
        loc="lower right"
    )
    ax.grid(alpha=.14)

    fig.tight_layout(pad=1.7)

    export(
        fig,
        "FigC_partial50_paired_difference_ECDF"
    )


def write_report(stats, perf, threshold, gate):
    report = [
        "TNSE Exp53 - Independent Cohort Audit",
        "=" * 56,
        "",
        "This is a reanalysis of Exp51 and Exp52.",
        "Not a newly executed simulator experiment.",
        "It is exploratory because model development",
        "continued between the two test cohorts.",
        "",
        f"Exp52 frozen gate threshold: {threshold}",
        "",
        "GATE IDENTITY CHECK:",
        gate.to_string(index=False),
        "",
        "PRIMARY COMPARISON: STATE_LOCAL vs FULL",
        stats[
            stats.comparison == "STATE_LOCAL vs FULL"
        ].to_string(index=False),
        "",
        "NEXT VALIDATION PRIORITIES:",
        "1. Verify true state and observed state separation.",
        "2. Confirm candidate/Guard fair comparison.",
        "3. If adaptive gating is claimed, demonstrate",
        "   differing gate decisions in held-out conditions.",
        "4. Confirm across network topologies and physical limits.",
        "5. Do not silently select the better test cohort.",
    ]

    (OUT / "AUDIT_REPORT.txt").write_text(
        "\n".join(report),
        encoding="utf-8"
    )


def pack():
    archive = OUT / "EXP53_REANALYSIS_AND_FIGURES.zip"

    with ZipFile(
        archive,
        "w",
        ZIP_DEFLATED,
        compresslevel=6
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, arcname=p.name)

        z.write(
            Path(__file__),
            arcname="SOURCE/tnse_exp53_audit.py"
        )

    with ZipFile(archive) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(
                f"ZIP integrity error: {bad}"
            )

    print("[ZIP VERIFIED]", archive)


def main():
    df = build_data()

    stats, perf = analyze(df)
    threshold, gate = audit_gate()

    source_audit()

    figure_forest(stats)
    figure_observation(perf)
    figure_partial50(df)

    write_report(
        stats, perf, threshold, gate
    )
    pack()

    print("\n========== AUDIT COMPLETE ==========")
    print("Output:", OUT)
    print("\nExp52 threshold:", threshold)

    print(
        "\nSTATE_LOCAL vs FULL:",
        stats[
            stats.comparison == "STATE_LOCAL vs FULL"
        ][[
            "mode", "cohort", "n",
            "effect", "ci_low", "ci_high"
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
