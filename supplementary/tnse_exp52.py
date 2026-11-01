#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TNSE Exp52
Reliability-Aware Adaptive Defense

Phases:
  pilot     Interface and policy identity checks
  validate  Tune reliability threshold, validation only
  test      New-seed frozen holdout and full report

No old checkpoint is modified.
No synthetic values replace experimental outputs.
"""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tnse_exp49 as e49
import tnse_exp50_v2 as e50
import tnse_exp51 as e51

ref = e51.ref

OUT = (
    ROOT / "revision_experiments/results/"
    "52_reliability_gate"
)
OUT.mkdir(parents=True, exist_ok=True)

e49.SAVED = OUT

ORIGINAL_OBSERVED_CHOOSE = e51.observed_choose

METHODS = [
    "CDP",
    "FULL",
    "STATE_LOCAL",
    "STATIC_EXTRA",
    "QUALITY_GATE"
]

MODES = [
    "oracle",
    "partial80",
    "partial50",
    "blind"
]

DEPLOY_MODES = [
    "partial80",
    "partial50"
]

THRESHOLDS = [
    0.00,
    0.55,
    0.70,
    0.85,
    1.01
]

CURRENT_THRESHOLD = 0.70

VAL_FIRST = 11000
TEST_FIRST = 12000

KEYS = [
    "pattern",
    "budget",
    "onset",
    "seed"
]

COLORS = {
    "CDP": "#93A4B0",
    "FULL": "#315D83",
    "STATE_LOCAL": "#C48A3A",
    "STATIC_EXTRA": "#B86C70",
    "QUALITY_GATE": "#16837D"
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.edgecolor": "#BAC7CF",
    "savefig.facecolor": "white"
})


# ==================================================
# 1. RELIABILITY-AWARE POLICY
# ==================================================

def selected_component(mode, threshold):

    # The reliability is a predefined,
    # externally calibrated sensor property.
    # It is NOT the hidden infection state.
    recall = float(
        e51.MODES[mode][0]
    )

    return (
        "STATE_LOCAL"
        if recall >= threshold
        else "FULL"
    )


def quality_choose(
    method, state, pattern,
    budget, info, seed,
    train, guard, adj, weight
):

    policy = method

    if method == "QUALITY_GATE":

        policy = selected_component(
            e51.MODE,
            CURRENT_THRESHOLD
        )

    return ORIGINAL_OBSERVED_CHOOSE(
        policy,
        state,
        pattern,
        budget,
        info,
        seed,
        train,
        guard,
        adj,
        weight
    )


# Register new policy in the shared simulator.
ref.choose = quality_choose


# ==================================================
# 2. FILE INTEGRITY AND PREDECLARED PROTOCOL
# ==================================================

def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def fingerprint():

    paths = [
        "tnse_exp49.py",
        "tnse_exp50_v2.py",
        "tnse_exp51.py",
        "tnse_exp52.py",
        "revision_experiments/"
        "47_final_closed_loop_validation.py",
        "src/drad_mpc.py",
        "src/seid_malware.py",
        "src/dynamic_risk_defense.py"
    ]

    return {
        p: sha(ROOT / p)
        for p in paths
    }


def frozen(path, obj):
    content = json.dumps(
        obj,
        indent=2,
        sort_keys=True
    )

    if path.exists():

        previous = json.loads(
            path.read_text()
        )

        if previous != obj:
            raise RuntimeError(
                "Frozen protocol mismatch: "
                + str(path)
            )

    else:
        path.write_text(content)


def setup():

    global CURRENT_THRESHOLD

    lock50 = (
        ROOT /
        "revision_experiments/results/"
        "50_state_local_v2/"
        "LOCKED_ALPHA.json"
    )

    if not lock50.exists():
        raise FileNotFoundError(
            "Exp50 locked alpha missing"
        )

    conf = json.loads(
        lock50.read_text()
    )

    e50.ALPHA = float(conf["alpha"])

    if ref.run_one.__globals__.get(
        "choose"
    ) is not quality_choose:

        raise RuntimeError(
            "Quality-gate hook is not connected"
        )

    ref.os.chdir(ROOT)

    graph, adj, weights = (
        e50.build_environment()
    )

    print(
        "[LOCKED ALPHA]",
        e50.ALPHA,
        flush=True
    )

    return graph, adj, weights


def scenarios(first, n):

    return e50.make_cases(
        first,
        n
    )


# ==================================================
# 3. RUN A REAL SIMULATION BLOCK
# ==================================================

def run_block(
    phase, mode,
    cases, threshold,
    methods, env, hashes
):

    global CURRENT_THRESHOLD

    CURRENT_THRESHOLD = float(
        threshold
    )

    e51.MODE = mode
    e49.SAVED = OUT

    # The label includes phase + observation mode.
    # Different phases use different seed ranges.
    label = f"{phase}_{mode}"

    print(
        "[RUN]",
        label,
        "cases=", len(cases),
        "gate=", CURRENT_THRESHOLD,
        flush=True
    )

    graph, adj, weights = env

    df, events = e49.run_block(
        label,
        cases,
        0.0,
        0.0,
        list(methods),
        graph,
        adj,
        weights,
        hashes
    )

    if df.duplicated(
        KEYS + ["method"]
    ).any():
        raise RuntimeError(
            "Duplicate paired records"
        )

    return df, events


def wide_pas(df):

    w = df.pivot(
        index=KEYS,
        columns="method",
        values="PAS"
    )

    if w.isna().any().any():
        raise RuntimeError(
            "Missing PAS values"
        )

    return w


# ==================================================
# 4. CHECK POLICY IDENTITY
# ==================================================

def identity_check(
    mode, threshold,
    outcomes, events
):

    expected = selected_component(
        mode,
        threshold
    )

    w = wide_pas(outcomes)

    if not {
        "QUALITY_GATE",
        expected
    }.issubset(w.columns):

        raise RuntimeError(
            "Missing identity-control method"
        )

    maxdiff = float(
        np.max(
            np.abs(
                w["QUALITY_GATE"]
                - w[expected]
            )
        )
    )

    if maxdiff > 1e-10:
        raise RuntimeError(
            "Gate and selected component "
            f"have inconsistent PAS: {maxdiff}"
        )

    required = set(
        KEYS + [
            "stage",
            "method",
            "protected_nodes"
        ]
    )

    if not required.issubset(
        events.columns
    ):
        raise RuntimeError(
            "Action audit columns missing"
        )

    a = events.pivot_table(
        index=KEYS+["stage"],
        columns="method",
        values="protected_nodes",
        aggfunc="first"
    )

    if (
        a["QUALITY_GATE"].isna().any()
        or a[expected].isna().any()
    ):
        raise RuntimeError(
            "Missing action audit rows"
        )

    mismatch = int(
        (
            a["QUALITY_GATE"].astype(str)
            != a[expected].astype(str)
        ).sum()
    )

    if mismatch != 0:
        raise RuntimeError(
            "Gate and selected component "
            f"actions differ: {mismatch}"
        )

    print(
        "[IDENTITY PASS]",
        mode,
        "gate equals",
        expected,
        flush=True
    )


# ==================================================
# 5. VALIDATION OBJECTIVE
# ==================================================

def cvar95(x):

    x = np.asarray(
        x, dtype=float
    )

    n = max(
        1,
        int(np.ceil(.05*len(x)))
    )

    return float(
        np.sort(x)[-n:].mean()
    )


def objective(x):

    x = np.asarray(
        x, dtype=float
    )

    return float(
        x.mean()
        + .20*cvar95(x)
    )


def validate(n_val, n_test, env, hashes):

    files = {}
    cases = scenarios(
        VAL_FIRST,
        n_val
    )

    # Only deployable sensor conditions
    # participate in threshold selection.
    for mode in DEPLOY_MODES:

        df, events = run_block(
            "validation",
            mode,
            cases,
            .70,
            [
                "CDP",
                "FULL",
                "STATE_LOCAL",
                "STATIC_EXTRA"
            ],
            env,
            hashes
        )

        files[mode] = wide_pas(df)

    records = []

    for threshold in THRESHOLDS:

        values = []
        activations = 0

        for mode in DEPLOY_MODES:

            policy = selected_component(
                mode,
                threshold
            )

            x = files[mode][
                policy
            ].to_numpy(float)

            values.append(x)

            if policy == "STATE_LOCAL":
                activations += 1

        pooled = np.concatenate(
            values
        )

        records.append({
            "threshold": threshold,
            "mean_PAS": float(
                pooled.mean()
            ),
            "CVaR95_PAS": cvar95(pooled),
            "objective": objective(pooled),
            "number_dynamic_modes":
                activations
        })

    table = pd.DataFrame(
        records
    ).sort_values(
        [
            "objective",
            "threshold"
        ],
        ascending=[
            True,
            False
        ]
    )

    table.to_csv(
        OUT/"validation_thresholds.csv",
        index=False
    )

    winner = table.iloc[0]

    cfg = {
        "threshold": float(
            winner.threshold
        ),
        "alpha": float(
            e50.ALPHA
        ),
        "n_val": n_val,
        "n_test": n_test,
        "val_seed_first": VAL_FIRST,
        "test_seed_first": TEST_FIRST,
        "methods": METHODS,
        "modes": MODES,
        "deploy_modes": DEPLOY_MODES,
        "criterion": (
            "pooled validation mean PAS "
            "+ 0.20 CVaR95"
        ),
        "source_hashes": hashes
    }

    frozen(
        OUT/"LOCKED_GATE.json",
        cfg
    )

    print(
        "\nVALIDATION RANKING\n",
        table.to_string(index=False),
        flush=True
    )

    print(
        "\nFROZEN THRESHOLD:",
        cfg["threshold"],
        flush=True
    )


# ==================================================
# 6. PAIRWISE INFERENCE
# ==================================================

def infer(delta, strata, rng):

    x = np.asarray(
        delta, dtype=float
    )

    B = 2500
    N = len(x)

    bootstrap = np.zeros(B)

    # Resample complete pairs within
    # predefined scenario strata.
    for group in strata:

        z = x[np.asarray(
            group, dtype=int
        )]

        m = len(z)

        ids = rng.integers(
            0, m,
            size=(B, m)
        )

        bootstrap += (
            z[ids].sum(axis=1)
            / N
        )

    lo, hi = np.quantile(
        bootstrap,
        [.025, .975]
    )

    # Paired sign-flip randomization
    # tests a mean-effect null.
    permutations = 5000

    obs = abs(x.mean())

    extreme = 0

    for start in range(
        0, permutations, 250
    ):

        b = min(
            250,
            permutations-start
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(b, N)
        )

        sim = np.abs(
            (signs*x).mean(axis=1)
        )

        extreme += int(
            (sim >= obs-1e-15).sum()
        )

    p = (
        extreme+1
    ) / (
        permutations+1
    )

    return float(lo), float(hi), float(p)


def assess(
    df, mode, contrasts, rng
):

    w = wide_pas(df)

    strata_columns = [
        "pattern",
        "budget",
        "onset"
    ]

    index_df = w.index.to_frame(
        index=False
    )

    strata = [
        group.index.to_numpy()
        for _, group in index_df.groupby(
            strata_columns,
            sort=True
        )
    ]

    records = []

    for A, B in contrasts:

        # Positive = B has lower PAS than A.
        delta = (
            w[A]-w[B]
        ).to_numpy(float)

        lo, hi, p = infer(
            delta,
            strata,
            rng
        )

        records.append({
            "mode": mode,
            "comparison": f"{B} vs {A}",
            "n": len(delta),
            "mean_gain": float(
                delta.mean()
            ),
            "CI_low": lo,
            "CI_high": hi,
            "p_randomization": p,
            "better_fraction": float(
                (delta > 1e-12).mean()
            ),
            "worse_fraction": float(
                (delta < -1e-12).mean()
            ),
            "tied_fraction": float(
                (abs(delta) <= 1e-12).mean()
            )
        })

    return pd.DataFrame(
        records
    )


def test_and_analyze(
    n_test, env, hashes, lock
):

    cases = scenarios(
        TEST_FIRST,
        n_test
    )

    threshold = lock["threshold"]

    all_dfs = {}
    all_events = {}
    all_stats = []

    contrasts = [
        ("CDP", "QUALITY_GATE"),
        ("FULL", "QUALITY_GATE"),
        ("STATIC_EXTRA", "STATE_LOCAL"),
        ("STATE_LOCAL", "QUALITY_GATE")
    ]

    rng = np.random.default_rng(
        525252
    )

    for mode in MODES:

        df, ev = run_block(
            "test",
            mode,
            cases,
            threshold,
            METHODS,
            env,
            hashes
        )

        identity_check(
            mode,
            threshold,
            df,
            ev
        )

        all_dfs[mode] = df
        all_events[mode] = ev

        all_stats.append(
            assess(
                df, mode,
                contrasts, rng
            )
        )

    # Check CDP is unchanged when only
    # the observation reporting changes.
    reference = wide_pas(
        all_dfs["oracle"]
    )["CDP"]

    for mode in MODES[1:]:

        candidate = wide_pas(
            all_dfs[mode]
        )["CDP"]

        diff = np.max(
            abs(reference-candidate)
        )

        if diff > 1e-10:
            raise RuntimeError(
                f"CDP observation invariance "
                f"failed in {mode}: {diff}"
            )

    print(
        "[INVARIANCE PASS]"
        " CDP external evaluation",
        flush=True
    )

    # Pooled deployable conditions.
    pooled = []

    for mode in DEPLOY_MODES:

        x = all_dfs[mode].copy()
        x["mode"] = mode

        pooled.append(x)

    combined = pd.concat(
        pooled,
        ignore_index=True
    )

    w = combined.pivot(
        index=["mode"]+KEYS,
        columns="method",
        values="PAS"
    )

    strat = (
        w.index.to_frame(
            index=False
        )
    )

    strata = [
        g.index.to_numpy()
        for _, g in strat.groupby(
            [
                "mode",
                "pattern",
                "budget",
                "onset"
            ],
            sort=True
        )
    ]

    pooled_rows = []

    for A, B in [
        ("CDP", "QUALITY_GATE"),
        ("FULL", "QUALITY_GATE")
    ]:

        delta = (
            w[A]-w[B]
        ).to_numpy(float)

        lo, hi, p = infer(
            delta,
            strata,
            rng
        )

        pooled_rows.append({
            "mode": "pooled_deployable",
            "comparison": f"{B} vs {A}",
            "n": len(delta),
            "mean_gain": float(
                delta.mean()
            ),
            "CI_low": lo,
            "CI_high": hi,
            "p_randomization": p,
            "better_fraction": float(
                (delta > 1e-12).mean()
            ),
            "worse_fraction": float(
                (delta < -1e-12).mean()
            ),
            "tied_fraction": float(
                (abs(delta) <= 1e-12).mean()
            )
        })

    all_stats.append(
        pd.DataFrame(pooled_rows)
    )

    result = pd.concat(
        all_stats,
        ignore_index=True
    )

    # Holm adjustment across every reported
    # confirmatory and secondary contrast.
    p = result.p_randomization.to_numpy()

    order = np.argsort(p)
    m = len(p)

    adj = np.maximum.accumulate(
        p[order] *
        (m-np.arange(m))
    )

    corrected = np.empty(m)

    corrected[order] = np.minimum(
        adj, 1.0
    )

    result["holm_p_all"] = corrected

    result.to_csv(
        OUT/"ALL_PAIRED_CONTRASTS.csv",
        index=False
    )

    # Every mode, every method.
    performance = []

    for mode, df in all_dfs.items():

        for method, g in df.groupby(
            "method"
        ):

            x = g.PAS.to_numpy(float)

            performance.append({
                "mode": mode,
                "method": method,
                "n": len(x),
                "mean_PAS": float(
                    x.mean()
                ),
                "P95_PAS": float(
                    np.quantile(x, .95)
                ),
                "CVaR95_PAS": cvar95(x),
                "mean_switches": float(
                    g.switch_count.mean()
                )
            })

    perf = pd.DataFrame(
        performance
    )

    perf.to_csv(
        OUT/"ALL_METHOD_PERFORMANCE.csv",
        index=False
    )

    print(
        "\n========== PAIRED INFERENCE ==========\n",
        result.to_string(index=False),
        flush=True
    )

    print(
        "\n========== METHOD PERFORMANCE ==========\n",
        perf.to_string(index=False),
        flush=True
    )

    return result, perf


# ==================================================
# 7. CAMERA-READY MULTIPANEL FIGURES
# ==================================================

def export(fig, name):

    figdir = OUT/"figures"
    figdir.mkdir(exist_ok=True)

    for ext in ["pdf", "svg", "png"]:

        fig.savefig(
            figdir/f"{name}.{ext}",
            dpi=260,
            bbox_inches="tight",
            pad_inches=.17
        )

    plt.close(fig)


def make_figures(stats, perf):

    # ---------- Figure A ----------
    fig, axs = plt.subplots(
        1, 3,
        figsize=(15, 4.8)
    )

    x = np.arange(len(MODES))

    ax = axs[0]

    for comparison, color, shift in [
        (
            "QUALITY_GATE vs FULL",
            COLORS["QUALITY_GATE"],
            -.09
        ),
        (
            "QUALITY_GATE vs CDP",
            COLORS["CDP"],
            .09
        )
    ]:

        sub = (
            stats[
                (stats.comparison == comparison)
                &
                (stats["mode"].isin(MODES))
            ]
            .set_index("mode")
            .reindex(MODES)
        )

        for i, row in enumerate(
            sub.itertuples()
        ):

            value = row.mean_gain

            ax.errorbar(
                i+shift,
                value,
                yerr=[
                    [
                        max(
                            0,
                            value-row.CI_low
                        )
                    ],
                    [
                        max(
                            0,
                            row.CI_high-value
                        )
                    ]
                ],
                color=color,
                fmt="o",
                ms=5,
                capsize=3,
                lw=1.5
            )

        ax.plot(
            [], [],
            marker="o",
            linestyle="none",
            color=color,
            label=comparison
        )

    ax.axhline(
        0,
        color="#9EADB8",
        ls="--",
        lw=1
    )

    ax.set_xticks(
        x, MODES,
        rotation=20
    )
    ax.set_ylabel(
        "Mean paired PAS reduction"
    )
    ax.set_title(
        "(a) Observation-dependent benefit",
        loc="left",
        pad=15
    )
    ax.grid(
        axis="y", alpha=.15
    )

    # ---------- Figure B ----------
    ax = axs[1]

    for method in [
        "CDP",
        "FULL",
        "STATE_LOCAL",
        "QUALITY_GATE"
    ]:

        sub = (
            perf[
                perf.method == method
            ]
            .set_index("mode")
            .reindex(MODES)
        )

        ax.plot(
            x,
            sub.mean_PAS,
            "-o",
            color=COLORS[method],
            lw=1.7,
            ms=4,
            label=method
        )

    ax.set_xticks(
        x, MODES,
        rotation=20
    )
    ax.set_ylabel(
        "Mean PAS (lower better)"
    )
    ax.set_title(
        "(b) Aggregate defensive impact",
        loc="left",
        pad=15
    )
    ax.grid(
        axis="y", alpha=.15
    )

    # ---------- Figure C ----------
    ax = axs[2]

    for method in [
        "CDP",
        "FULL",
        "STATE_LOCAL",
        "QUALITY_GATE"
    ]:

        sub = (
            perf[
                perf.method == method
            ]
            .set_index("mode")
            .reindex(MODES)
        )

        ax.plot(
            x,
            sub.CVaR95_PAS,
            "-s",
            color=COLORS[method],
            lw=1.65,
            ms=4
        )

    ax.set_xticks(
        x, MODES,
        rotation=20
    )
    ax.set_ylabel(
        "CVaR95 PAS (lower better)"
    )
    ax.set_title(
        "(c) Worst-tail impact",
        loc="left",
        pad=15
    )
    ax.grid(
        axis="y", alpha=.15
    )

    handles, labels = (
        axs[1].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(.5, .025),
        frameon=False,
        ncol=4,
        fontsize=8
    )

    fig.subplots_adjust(
        left=.065,
        right=.985,
        top=.86,
        bottom=.27,
        wspace=.37
    )

    export(
        fig,
        "EXP52_reliability_performance"
    )

    # ---------- Figure B: heatmap ----------
    pivot = (
        stats[
            stats["mode"].isin(MODES)
        ]
        .pivot(
            index="mode",
            columns="comparison",
            values="mean_gain"
        )
        .reindex(MODES)
    )

    values = pivot.to_numpy(
        dtype=float
    )

    maxabs = max(
        abs(values).max(),
        .001
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 4.7)
    )

    im = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-maxabs,
        vmax=maxabs,
        aspect="auto"
    )

    ax.set_yticks(
        np.arange(len(pivot.index)),
        pivot.index
    )

    ax.set_xticks(
        np.arange(len(pivot.columns)),
        [
            x.replace(
                " vs ",
                "\nvs "
            )
            for x in pivot.columns
        ]
    )

    ax.tick_params(
        axis="x",
        labelsize=8
    )

    for i in range(values.shape[0]):

        for j in range(values.shape[1]):

            value = values[i, j]

            ax.text(
                j,
                i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if abs(value) > .65*maxabs
                    else "#203D50"
                )
            )

    ax.set_title(
        "Paired PAS effects across observation regimes",
        loc="left",
        pad=17
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=.028,
        pad=.025,
        label="Positive favors first-named policy"
    )

    fig.subplots_adjust(
        left=.15,
        right=.94,
        top=.87,
        bottom=.24
    )

    export(
        fig,
        "EXP52_paired_effect_heatmap"
    )


# ==================================================
# 8. PACKAGE ALL EXP52 EVIDENCE
# ==================================================

def package():

    target = OUT/"TNSE_EXP52_COMPLETE_EVIDENCE.zip"

    with ZipFile(
        target,
        "w",
        ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True
    ) as archive:

        for f in sorted(
            OUT.rglob("*")
        ):

            if (
                not f.is_file()
                or f == target
                or f.name.endswith(
                    ".json.part"
                )
            ):
                continue

            archive.write(
                f,
                arcname=str(
                    f.relative_to(ROOT)
                )
            )

        archive.write(
            Path(__file__),
            arcname="SOURCE/tnse_exp52.py"
        )

    with ZipFile(
        target
    ) as archive:

        bad = archive.testzip()

        if bad is not None:
            raise RuntimeError(
                "Corrupt zip member: "
                + bad
            )

    digest = sha(target)

    (
        OUT/"TNSE_EXP52_COMPLETE_EVIDENCE.sha256"
    ).write_text(
        digest
        + "  "
        + target.name
        + "\n"
    )

    print(
        "\nPACKAGE VERIFIED:",
        target,
        "\nSIZE MB:",
        round(
            target.stat().st_size/1024**2,
            2
        ),
        flush=True
    )


# ==================================================
# 9. CLI
# ==================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--phase",
        required=True,
        choices=[
            "pilot",
            "validate",
            "test"
        ]
    )

    ap.add_argument(
        "--n-val",
        type=int,
        default=6
    )

    ap.add_argument(
        "--n-test",
        type=int,
        default=20
    )

    args = ap.parse_args()

    hashes = fingerprint()
    env = setup()

    if args.phase == "pilot":

        pilot_cases = [
            (
                "DDAC", 20, 90, 10201
            ),
            (
                "RC", 40, 210, 10202
            )
        ]

        for mode in MODES:

            df, events = run_block(
                "pilot",
                mode,
                pilot_cases,
                .70,
                METHODS,
                env,
                hashes
            )

            identity_check(
                mode,
                .70,
                df,
                events
            )

        print(
            "\nEXP52 PILOT PASSED",
            flush=True
        )

        return

    if (
        args.n_val < 3
        or args.n_test < 5
    ):
        raise ValueError(
            "Too few seeds per condition"
        )

    if args.phase == "validate":

        validate(
            args.n_val,
            args.n_test,
            env,
            hashes
        )

        return

    lockpath = OUT/"LOCKED_GATE.json"

    if not lockpath.exists():
        raise RuntimeError(
            "Run validation before testing"
        )

    lock = json.loads(
        lockpath.read_text()
    )

    if (
        lock["source_hashes"] != hashes
        or lock["n_val"] != args.n_val
        or lock["n_test"] != args.n_test
        or lock["alpha"] != e50.ALPHA
    ):
        raise RuntimeError(
            "Frozen configuration changed"
        )

    statistics, performance = (
        test_and_analyze(
            args.n_test,
            env,
            hashes,
            lock
        )
    )

    make_figures(
        statistics,
        performance
    )

    package()

    print(
        "\nEXP52 FINISHED SUCCESSFULLY",
        flush=True
    )


if __name__ == "__main__":
    main()
