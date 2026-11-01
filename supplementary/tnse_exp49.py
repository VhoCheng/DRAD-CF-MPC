#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TNSE Exp49: split-confirmation rolling validation.

1. Independent validation and test seeds
2. Gamma/kappa tuning only on validation
3. Locked holdout comparison
4. Component ablations with a common protocol
5. Bootstrap confidence intervals and publication figures

This is a new protocol; do not merge with historical Exp47.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "revision_experiments"
SAVED = EXP / "results" / "49_split_confirm_revision"

SAVED.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "exp47_reference",
    EXP / "47_final_closed_loop_validation.py"
)

ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

METHODS = [
    "CDP",
    "FULL",
    "NO_DYNAMIC",
    "NO_CF_SELECTOR",
    "NO_GUARD",
    "HEURISTIC_NO_GUARD"
]

GAMMAS = [0.00, 0.02, 0.05, 0.10]
KAPPAS = [0.00, 0.50, 1.00]

PATTERNS = ["DDAC", "RC"]
BUDGETS = [20, 40]
ONSETS = [90, 210]

SEED_VAL = 1000
SEED_TEST = 2000

KTRAIN = 8
KGUARD = 24
BOOTS = 1500


# ===================================================
# VALIDATION OBJECTIVE
# ===================================================

def loss_score(x):
    """
    Predefined score: mean PAS + 0.20 * CVaR95.
    Lower is better.
    """

    x = np.asarray(x, float)
    tail_start = max(
        0, int(.95 * len(x)) - 1
    )

    tail = np.sort(x)[tail_start:]

    return float(
        x.mean() + .20 * tail.mean()
    )


# ===================================================
# CORRECTED COMMON POLICY SELECTION CONTRACT
# ===================================================

def choose(
    method, state, pattern,
    budget, info, seed,
    train, guard, adj, weight
):

    dynamic = method != "NO_DYNAMIC"

    candidates = ref.candidate_nodes(
        pattern, budget, state,
        info, seed, dynamic
    )

    if method == "CDP":
        return (
            candidates["CDP"],
            "CDP", "CDP",
            False, len(candidates)
        )

    # Remove CF-based candidate ranking, not
    # the separately retained Guard confirmation.
    if method in [
        "NO_CF_SELECTOR",
        "HEURISTIC_NO_GUARD"
    ]:

        nodes = tuple(sorted(
            ref.mpc.candidate_drad_score_nodes(
                pattern=pattern,
                defense_budget=budget,
                attack_information_ratio=info,
                active_nodes=np.flatnonzero(
                    state == 2
                ).astype(int).tolist(),
                exposed_nodes=np.flatnonzero(
                    state == 1
                ).astype(int).tolist(),
                table_dir="outputs/tables",
                seed=seed
            )
        ))

        if len(nodes) != budget:
            raise ValueError(
                "Heuristic budget mismatch"
            )

        candidates["HEURISTIC"] = nodes

        best = "HEURISTIC"
        required = ["CDP", best]

    else:
        required = list(candidates)
        best = None

    # Candidate selection uses training rollouts.
    pred = {
        k: ref.predict(
            state,
            candidates[k],
            train,
            adj,
            weight,
            budget,
            False  # corrected unconditional PAS
        )
        for k in required
    }

    if best is None:
        best = min(
            pred,
            key=lambda k:
                pred[k]["objective"]
        )

    # Independent no-guard counterpart.
    if method in [
        "NO_GUARD",
        "HEURISTIC_NO_GUARD"
    ]:
        return (
            candidates[best],
            best,
            best,
            best != "CDP",
            len(candidates)
        )

    if best == "CDP":
        return (
            candidates["CDP"],
            "CDP",
            best,
            False,
            len(candidates)
        )

    cdp_pred = pred["CDP"]
    alt_pred = pred[best]

    relative = (
        cdp_pred["objective"]
        - alt_pred["objective"]
    ) / (
        abs(cdp_pred["objective"]) + 1e-9
    )

    # Independent Guard confirmation.
    guard_cdp = ref.predict(
        state,
        candidates["CDP"],
        guard,
        adj,
        weight,
        budget,
        False
    )

    guard_alt = ref.predict(
        state,
        candidates[best],
        guard,
        adj,
        weight,
        budget,
        False
    )

    differences = (
        np.asarray(guard_cdp["samples"])
        - np.asarray(guard_alt["samples"])
    )

    if len(differences) > 1:
        se = float(
            np.std(differences, ddof=1)
            / np.sqrt(len(differences))
        )
    else:
        se = float("inf")

    accept = bool(
        relative >= ref.GAMMA
        and
        differences.mean()
        - ref.KAPPA * se > 0
    )

    return (
        candidates[best]
        if accept else candidates["CDP"],

        best if accept else "CDP",
        best,
        accept,
        len(candidates)
    )


# Override policy selection without editing Exp47.
ref.choose = choose


# ===================================================
# REPRODUCIBLE PROTOCOL AND CHECKPOINTS
# ===================================================

def cases(first, n):

    return [
        (pat, b, onset, s)
        for pat in PATTERNS
        for b in BUDGETS
        for onset in ONSETS
        for s in range(first, first + n)
    ]


def source_hashes():

    files = [
        "src/drad_mpc.py",
        "src/dynamic_risk_defense.py",
        "src/seid_malware.py",
        "revision_experiments/"
        "47_final_closed_loop_validation.py",
        Path(__file__).name
    ]

    return {
        f: hashlib.sha256(
            (ROOT / f).read_bytes()
        ).hexdigest()
        for f in files
    }


def freeze(dest, meta):

    dest.mkdir(
        parents=True,
        exist_ok=True
    )

    f = dest / "protocol.json"

    content = json.dumps(
        meta,
        sort_keys=True,
        indent=2
    )

    if f.exists() and f.read_text() != content:
        raise RuntimeError(
            "Frozen protocol mismatch: "
            + str(f)
        )

    if not f.exists():
        f.write_text(content)


def run_block(
    label, selected_cases,
    gamma, kappa, methods,
    graph, adj, weights, hashes
):

    dest = SAVED / label

    freeze(
        dest,
        {
            "gamma": gamma,
            "kappa": kappa,
            "methods": methods,
            "selected_cases": selected_cases,
            "hashes": hashes,
            "KTRAIN": KTRAIN,
            "KGUARD": KGUARD,
            "contract": (
                "Three-stage rolling with "
                "independent guard confirmation"
            )
        }
    )

    ref.GAMMA = gamma
    ref.KAPPA = kappa
    ref.K_TRAIN = KTRAIN
    ref.K_GUARD = KGUARD
    ref.METHODS = methods

    checkpoints = dest / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    start = time.time()

    for i, case in enumerate(
        selected_cases, 1
    ):

        name = (
            f"{case[0]}_b{case[1]}"
            f"_t{case[2]}_s{case[3]}.json"
        )

        path = checkpoints / name

        if not path.exists():

            results, events = ref.run_one(
                case,
                graph,
                adj,
                weights
            )

            temp = path.with_suffix(
                ".json.part"
            )

            temp.write_text(
                json.dumps({
                    "results": results,
                    "events": events
                })
            )

            temp.replace(path)

        if (
            i % 4 == 0
            or i == len(selected_cases)
        ):
            print(
                f"[{label}] "
                f"{i}/{len(selected_cases)} "
                f"elapsed={time.time()-start:.1f}s",
                flush=True
            )

    all_results = []
    all_events = []

    for case in selected_cases:

        name = (
            f"{case[0]}_b{case[1]}"
            f"_t{case[2]}_s{case[3]}.json"
        )

        obj = json.loads(
            (checkpoints / name).read_text()
        )

        all_results += obj["results"]
        all_events += obj["events"]

    df = pd.DataFrame(all_results)
    ev = pd.DataFrame(all_events)

    if len(df) != len(selected_cases) * len(methods):
        raise RuntimeError(
            "Incomplete experiment results"
        )

    if df.duplicated([
        "pattern", "budget",
        "onset", "seed", "method"
    ]).any():
        raise RuntimeError(
            "Duplicated case-method pairs"
        )

    df.to_csv(
        dest / "outcomes.csv",
        index=False
    )

    ev.to_csv(
        dest / "decisions.csv",
        index=False
    )

    return df, ev


# ===================================================
# PAIRED STATISTICAL EVALUATION
# ===================================================

def paired_report(df, ev, dest):

    keys = [
        "pattern", "budget",
        "onset", "seed"
    ]

    wide = df.pivot(
        index=keys,
        columns="method",
        values="PAS"
    )

    baseline = wide["CDP"].to_numpy()

    rng = np.random.default_rng(20261023)
    rows = []

    for method in METHODS:

        if method == "CDP":
            continue

        value = wide[method].to_numpy()

        # Positive: method has lower PAS than CDP.
        delta = baseline - value

        blocks = [
            g["CDP"].to_numpy()
            - g[method].to_numpy()
            for _, g in wide.groupby(
                level=[
                    "pattern", "budget", "onset"
                ]
            )
        ]

        # Stratified paired bootstrap.
        draws = np.zeros(BOOTS)

        for block in blocks:

            indexes = rng.integers(
                len(block),
                size=(BOOTS, len(block))
            )

            draws += (
                block[indexes].sum(axis=1)
                / len(delta)
            )

        nonzero = delta[
            np.abs(delta) > 1e-12
        ]

        p = (
            1.0 if len(nonzero) < 2
            else float(
                wilcoxon(nonzero).pvalue
            )
        )

        sorted_values = np.sort(value)

        tail = sorted_values[
            max(
                0,
                int(.95 * len(sorted_values)) - 1
            ):
        ]

        rows.append({
            "method": method,
            "n": len(delta),
            "mean_PAS": float(value.mean()),
            "CDP_mean": float(baseline.mean()),
            "mean_CDP_minus_method": float(delta.mean()),
            "CI_low": float(
                np.quantile(draws, .025)
            ),
            "CI_high": float(
                np.quantile(draws, .975)
            ),
            "p_unadjusted": p,
            "win_rate": float(
                (delta > 1e-12).mean()
            ),
            "loss_rate": float(
                (delta < -1e-12).mean()
            ),
            "tie_rate": float(
                (abs(delta) <= 1e-12).mean()
            ),
            "P95_PAS": float(
                np.quantile(value, .95)
            ),
            "CVaR95_PAS": float(
                tail.mean()
            ),
            "mean_switches": float(
                df.loc[
                    df.method == method,
                    "switch_count"
                ].mean()
            )
        })

    summary = pd.DataFrame(rows)

    # Holm correction across five comparisons.
    order = np.argsort(
        summary.p_unadjusted.to_numpy()
    )

    values = (
        summary.p_unadjusted.to_numpy()[order]
    )

    corrected = np.minimum(
        1,
        np.maximum.accumulate(
            values * (
                len(summary)
                - np.arange(len(summary))
            )
        )
    )

    summary.loc[
        order, "holm_p_all_methods"
    ] = corrected

    summary.to_csv(
        dest / "summary_paired.csv",
        index=False
    )

    # Eight predefined subgroup settings.
    subgroup_rows = []

    for setting, group in wide.groupby(
        level=[
            "pattern", "budget", "onset"
        ]
    ):

        for method in [
            "FULL",
            "NO_DYNAMIC",
            "NO_CF_SELECTOR",
            "NO_GUARD"
        ]:

            x = (
                group["CDP"]
                - group[method]
            ).to_numpy()

            indexes = rng.integers(
                len(x),
                size=(BOOTS, len(x))
            )

            boots = x[indexes].mean(axis=1)

            subgroup_rows.append({
                "pattern": setting[0],
                "budget": setting[1],
                "onset": setting[2],
                "method": method,
                "n": len(x),
                "gain": float(x.mean()),
                "lo": float(
                    np.quantile(boots, .025)
                ),
                "hi": float(
                    np.quantile(boots, .975)
                )
            })

    pd.DataFrame(subgroup_rows).to_csv(
        dest / "predefined_subgroups_exploratory.csv",
        index=False
    )

    # Actual first-stage action divergence.
    first = ev[
        ev.stage == 0
    ].pivot(
        index=keys,
        columns="method",
        values="protected_nodes"
    )

    divergence = []

    for method in METHODS:

        if method == "FULL":
            continue

        divergence.append({
            "comparator": method,
            "n": len(first),
            "different_first_action": float(
                (
                    first[method] != first["FULL"]
                ).mean()
            )
        })

    pd.DataFrame(divergence).to_csv(
        dest / "first_action_divergence.csv",
        index=False
    )

    print(
        summary.to_string(index=False),
        flush=True
    )


# ===================================================
# REAL PAPER FIGURES
# ===================================================

def plot_report(dest):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = pd.read_csv(
        dest / "summary_paired.csv"
    )

    sub = pd.read_csv(
        dest / "predefined_subgroups_exploratory.csv"
    )

    plt.rcParams.update({
        "font.size": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none"
    })

    # Main comparison forest plot.
    fig, ax = plt.subplots(
        figsize=(8.8, 4.8),
        layout="constrained"
    )

    for i, r in stats.iterrows():

        x = r.mean_CDP_minus_method

        ax.errorbar(
            x, i,
            xerr=[
                [max(0, x-r.CI_low)],
                [max(0, r.CI_high-x)]
            ],
            fmt="o",
            capsize=3,
            color="#287F87"
        )

    ax.axvline(
        0, color="gray", linestyle="--"
    )

    ax.set_yticks(
        range(len(stats)),
        stats.method
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "CDP PAS minus policy PAS "
        "(positive favors policy)"
    )

    ax.set_title(
        "Independent holdout: paired effects "
        "with 95% stratified CI",
        loc="left"
    )

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(
            dest / f"holdout_forest.{ext}",
            bbox_inches="tight",
            dpi=230
        )

    plt.close(fig)

    # Predefined subgroup plot.
    s = sub[
        sub.method == "FULL"
    ].sort_values(
        ["pattern", "budget", "onset"]
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 5),
        layout="constrained"
    )

    yy = np.arange(len(s))
    x = s.gain.to_numpy()

    ax.errorbar(
        x, yy,
        xerr=[
            np.maximum(0, x-s.lo),
            np.maximum(0, s.hi-x)
        ],
        fmt="o",
        capsize=3,
        color="#287F87"
    )

    ax.axvline(
        0, color="gray", linestyle="--"
    )

    ax.set_yticks(
        yy,
        [
            f"{r.pattern} | B={r.budget} | "
            f"onset={r.onset}"
            for r in s.itertuples()
        ]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "CDP PAS - FULL PAS "
        "(positive favors FULL)"
    )

    ax.set_title(
        "Eight predefined settings: "
        "exploratory subgroup CIs",
        loc="left"
    )

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(
            dest / f"holdout_subgroups.{ext}",
            bbox_inches="tight",
            dpi=230
        )

    plt.close(fig)


# ===================================================
# MAIN
# ===================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=[
            "pilot", "tune", "test"
        ],
        required=True
    )

    parser.add_argument(
        "--n-val",
        type=int,
        default=6
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=25
    )

    args = parser.parse_args()

    if args.n_val < 2 or args.n_test < 2:
        raise ValueError(
            "Need at least two seeds per setting"
        )

    hashes = source_hashes()

    ref.os.chdir(ROOT)

    graph = ref.mpc.build_comm_graph(
        n_nodes=118,
        seed=42
    )

    adj = ref.adjacency(graph)

    weights = {
        p: ref.severity_from_pattern(p, 118)
        for p in PATTERNS
    }

    # ---------------------------------
    # Phase 1: smoke test
    # ---------------------------------

    if args.phase == "pilot":

        testcases = [
            ("DDAC", 20, 90, 9000),
            ("RC", 40, 210, 9001)
        ]

        run_block(
            "pilot",
            testcases,
            .05,
            .5,
            METHODS,
            graph,
            adj,
            weights,
            hashes
        )

        print(
            "Pilot complete. "
            "No performance claim yet."
        )
        return

    validation_cases = cases(
        SEED_VAL,
        args.n_val
    )

    # ---------------------------------
    # Phase 2: validation-only tuning
    # ---------------------------------

    if args.phase == "tune":

        scores = []

        for gamma in GAMMAS:
            for kappa in KAPPAS:

                name = (
                    f"val_gamma_{gamma:.3f}"
                    f"_kappa_{kappa:.2f}"
                )

                df, _ = run_block(
                    name,
                    validation_cases,
                    gamma,
                    kappa,
                    ["CDP", "FULL"],
                    graph,
                    adj,
                    weights,
                    hashes
                )

                x = df.loc[
                    df.method == "FULL",
                    "PAS"
                ].to_numpy()

                tail = np.sort(x)[
                    max(0, int(.95*len(x))-1):
                ]

                scores.append({
                    "gamma": gamma,
                    "kappa": kappa,
                    "val_mean": float(x.mean()),
                    "val_CVaR95": float(
                        tail.mean()
                    ),
                    "score": loss_score(x)
                })

        table = pd.DataFrame(
            scores
        ).sort_values(
            ["score", "gamma", "kappa"]
        )

        table.to_csv(
            SAVED / "tune_validation_all_grid.csv",
            index=False
        )

        winner = table.iloc[0]

        lock = {
            "gamma": float(
                winner.gamma
            ),
            "kappa": float(
                winner.kappa
            ),
            "n_val": args.n_val,
            "n_test": args.n_test,
            "criterion": (
                "validation_mean_PAS "
                "+ 0.20*CVaR95_PAS"
            ),
            "source_hashes": hashes
        }

        path = (
            SAVED / "LOCKED_HYPERPARAMETERS.json"
        )

        if path.exists():
            if json.loads(
                path.read_text()
            ) != lock:
                raise RuntimeError(
                    "Locked selection differs. "
                    "Do not overwrite."
                )

        else:
            path.write_text(
                json.dumps(
                    lock,
                    indent=2,
                    sort_keys=True
                )
            )

        print(
            "LOCKED SELECTED:",
            lock
        )
        return

    # ---------------------------------
    # Phase 3: untouched test seeds
    # ---------------------------------

    lockfile = (
        SAVED / "LOCKED_HYPERPARAMETERS.json"
    )

    if not lockfile.exists():
        raise RuntimeError(
            "Run --phase tune first."
        )

    lock = json.loads(
        lockfile.read_text()
    )

    if (
        lock["source_hashes"] != hashes
        or lock["n_test"] != args.n_test
        or lock["n_val"] != args.n_val
    ):
        raise RuntimeError(
            "Source or test protocol changed "
            "after parameter selection."
        )

    test_cases = cases(
        SEED_TEST,
        args.n_test
    )

    df, ev = run_block(
        "test",
        test_cases,
        lock["gamma"],
        lock["kappa"],
        METHODS,
        graph,
        adj,
        weights,
        hashes
    )

    dest = SAVED / "test"

    paired_report(
        df, ev, dest
    )

    plot_report(dest)

    print(
        "FINAL OUTPUT:",
        dest
    )


if __name__ == "__main__":
    main()
