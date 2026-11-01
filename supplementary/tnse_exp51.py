#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exp51: Observation-limited validation.

Based on Exp49 + Exp50 V2.
- Fresh random seeds
- Frozen alpha from Exp50 validation
- Common observation model for every method
- State-location shuffling negative control
- Paired bootstrap and Holm correction
- Checkpointed simulations
"""

import argparse
import json
import hashlib
import sys
from pathlib import Path

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

ref = e50.ref

OUT = (
    ROOT / "revision_experiments/results/"
    "51_observation_access_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

e49.SAVED = OUT

MODE = "partial80"
SHAM = False
SHAM_SEED = 0

MODES = {
    "oracle": (1.00, 0.00, True),
    "partial80": (0.80, 0.01, False),
    "partial50": (0.50, 0.03, False),
    "blind": (0.00, 0.00, False)
}

METHODS = [
    "CDP",
    "FULL",
    "NO_DYNAMIC",
    "STATIC_EXTRA",
    "STATE_LOCAL",
    "SHUFFLED_LOCAL"
]

SAVED_CHOOSE = e50.choose_wrapper
SAVED_SCORE = e50.score_candidate


# ==================================================
# 1. COMMON OBSERVATION MODEL
# ==================================================

def observation(state, action_seed):

    recall, fp, perfect = MODES[MODE]

    state = np.asarray(
        state,
        dtype=np.int8
    ).reshape(-1)

    if perfect:
        return state.copy()

    rng = np.random.default_rng(
        ref.stable_seed(
            "Exp51-fixed-sensor",
            action_seed
        )
    )

    r1 = rng.random(len(state))
    r2 = rng.random(len(state))

    observed = np.zeros_like(state)

    # Globally detected status stays observable.
    observed[state == 3] = 3

    # Only a fraction of infected nodes
    # can be identified.
    identified = (
        (state == 2) &
        (r1 < recall)
    )

    observed[identified] = 2

    # False positive alarms.
    false_positive = (
        (state != 2) &
        (state != 3) &
        (r2 < fp)
    )

    observed[false_positive] = 2

    # Exposed but undetected states are
    # represented as unknown/susceptible.
    return observed


# ==================================================
# 2. SHUFFLED LOCATION NEGATIVE CONTROL
# ==================================================

def score_with_sham(pattern, state, method):

    if SHAM and method == "STATE_LOCAL":

        rng = np.random.default_rng(
            SHAM_SEED
        )

        perm = rng.permutation(
            len(state)
        )

        shuffled = np.asarray(
            state
        )[perm]

        return SAVED_SCORE(
            pattern,
            shuffled,
            method
        )

    return SAVED_SCORE(
        pattern,
        state,
        method
    )


# ==================================================
# 3. RESTRICT EVERY POLICY TO OBSERVATIONS
# ==================================================

def observed_choose(
    method, state, pattern,
    budget, info, seed,
    train, guard, adj, weight
):

    global SHAM, SHAM_SEED

    obs = observation(
        state,
        seed
    )

    SHAM = (
        method == "SHUFFLED_LOCAL"
    )

    SHAM_SEED = ref.stable_seed(
        "Exp51-shuffle",
        seed
    )

    try:

        actual_method = (
            "STATE_LOCAL"
            if SHAM
            else method
        )

        # This passes observed state into both
        # original candidate construction and
        # counterfactual prediction.
        return SAVED_CHOOSE(
            actual_method,
            obs,
            pattern,
            budget,
            info,
            seed,
            train,
            guard,
            adj,
            weight
        )

    finally:
        SHAM = False


e50.score_candidate = score_with_sham
ref.choose = observed_choose


# ==================================================
# 4. SOURCE HASHES AND LOCKED PROTOCOL
# ==================================================

def fingerprints():

    names = [
        "tnse_exp49.py",
        "tnse_exp50_v2.py",
        "tnse_exp51.py",
        "revision_experiments/"
        "47_final_closed_loop_validation.py",
        "src/drad_mpc.py",
        "src/seid_malware.py",
        "src/dynamic_risk_defense.py"
    ]

    return {
        p: hashlib.sha256(
            (ROOT / p).read_bytes()
        ).hexdigest()
        for p in names
    }


def verify_protocol(n, phase):

    lock_path = (
        ROOT /
        "revision_experiments/results/"
        "50_state_local_v2/"
        "LOCKED_ALPHA.json"
    )

    lock = json.loads(
        lock_path.read_text()
    )

    alpha = float(
        lock["alpha"]
    )

    # Use Exp50 validation-selected alpha.
    # Never retune on Exp51 test outcomes.
    e50.ALPHA = alpha

    config = {
        "seed_first": (
            8900 if phase == "pilot"
            else 9000
        ),
        "seeds_per_condition": n,
        "conditions": (
            "DDAC/RC x 20/40/60 "
            "x onset90/210"
        ),
        "modes": list(MODES),
        "methods": METHODS,
        "alpha_exp50_locked": alpha,
        "train_rollouts": e49.KTRAIN,
        "guard_rollouts": e49.KGUARD,
        "source_sha256": fingerprints()
    }

    filename = (
        "PILOT_PROTOCOL.json"
        if phase == "pilot"
        else "FULL_PROTOCOL.json"
    )

    path = OUT / filename

    body = json.dumps(
        config,
        indent=2,
        sort_keys=True
    )

    if path.exists():
        if path.read_text() != body:
            raise RuntimeError(
                "Existing protocol differs: "
                + str(path)
            )
    else:
        path.write_text(body)

    print(
        "LOCKED ALPHA:",
        alpha,
        flush=True
    )


# ==================================================
# 5. PAIRED STATISTICS
# ==================================================

def summarize(mode, df, events):

    dest = OUT / f"full_{mode}"
    dest.mkdir(exist_ok=True)

    keys = [
        "pattern",
        "budget",
        "onset",
        "seed"
    ]

    if df.duplicated(
        keys + ["method"]
    ).any():
        raise RuntimeError(
            "Duplicated case-method records"
        )

    wide = df.pivot(
        index=keys,
        columns="method",
        values="PAS"
    )

    if (
        set(METHODS) != set(wide.columns)
        or wide.isna().any().any()
    ):
        raise RuntimeError(
            "Incomplete matched outcomes"
        )

    rng = np.random.default_rng(
        51109
    )

    comparisons = [
        ("CDP", "STATE_LOCAL"),
        ("FULL", "STATE_LOCAL"),
        ("STATIC_EXTRA", "STATE_LOCAL"),
        ("SHUFFLED_LOCAL", "STATE_LOCAL"),
        ("NO_DYNAMIC", "FULL")
    ]

    rows = []

    for a, b in comparisons:

        # PAS is a loss.
        # Positive means b improves over a.
        delta = wide[a] - wide[b]

        boot = np.zeros(2000)

        for _, group in delta.groupby(
            level=[
                "pattern",
                "budget",
                "onset"
            ]
        ):

            values = group.to_numpy(
                dtype=float
            )

            n = len(values)

            ids = rng.integers(
                0,
                n,
                size=(2000, n)
            )

            boot += (
                values[ids].mean(axis=1)
                * n / len(delta)
            )

        x = delta.to_numpy(float)

        nonzero = x[
            abs(x) > 1e-12
        ]

        p = (
            1.0
            if len(nonzero) < 2
            else float(
                wilcoxon(nonzero).pvalue
            )
        )

        rows.append({
            "mode": mode,
            "contrast": f"{b} vs {a}",
            "n": len(x),
            "mean_gain": float(x.mean()),
            "CI95_low": float(
                np.quantile(boot, .025)
            ),
            "CI95_high": float(
                np.quantile(boot, .975)
            ),
            "win_frac": float(
                (x > 1e-12).mean()
            ),
            "loss_frac": float(
                (x < -1e-12).mean()
            ),
            "tie_frac": float(
                (abs(x) <= 1e-12).mean()
            ),
            "p_unadjusted": p
        })

    table = pd.DataFrame(rows)

    # Holm correction for five contrasts.
    p = table.p_unadjusted.to_numpy()
    order = np.argsort(p)
    m = len(order)

    sorted_adjusted = (
        p[order] *
        (m - np.arange(m))
    )

    sorted_adjusted = np.minimum(
        1,
        np.maximum.accumulate(
            sorted_adjusted
        )
    )

    adjusted = np.empty(m)
    adjusted[order] = sorted_adjusted

    table["p_holm_within_mode"] = adjusted

    table.to_csv(
        dest / "paired_statistics.csv",
        index=False
    )

    methods = (
        df.groupby("method")
        .agg(
            n=("PAS", "size"),
            mean_PAS=("PAS", "mean"),
            mean_switches=(
                "switch_count",
                "mean"
            )
        )
        .reset_index()
    )

    methods.to_csv(
        dest / "method_summary.csv",
        index=False
    )

    # Strong negative-control invariant:
    # Without any infection observations,
    # dynamic-local extra ranking should
    # reduce to static-extra ranking.
    if mode == "blind":

        difference = (
            wide["STATE_LOCAL"]
            - wide["STATIC_EXTRA"]
        ).abs().max()

        if difference > 1e-12:
            raise RuntimeError(
                "BLIND NEGATIVE CONTROL FAILED: "
                + str(difference)
            )

    print(
        "\n========", mode, "========",
        flush=True
    )

    print(
        table.to_string(index=False),
        flush=True
    )

    print(
        methods.to_string(index=False),
        flush=True
    )

    return table


# ==================================================
# 6. SCIENTIFIC FIGURES
# ==================================================

def plot_summary(table):

    fig, axs = plt.subplots(
        1, 2,
        figsize=(13.8, 5.5),
        layout="constrained"
    )

    modes = list(MODES)

    pivot = table.pivot(
        index="mode",
        columns="contrast",
        values="mean_gain"
    ).reindex(modes)

    values = pivot.to_numpy(float)

    limit = max(
        np.max(np.abs(values)),
        .001
    )

    im = axs[0].imshow(
        values,
        vmin=-limit,
        vmax=limit,
        cmap="RdBu_r",
        aspect="auto"
    )

    axs[0].set_yticks(
        range(len(modes)),
        modes
    )

    labels = [
        x.replace(
            " vs ",
            "\nvs "
        )
        for x in pivot.columns
    ]

    axs[0].set_xticks(
        range(len(labels)),
        labels
    )

    axs[0].tick_params(
        axis="x",
        labelsize=7
    )

    axs[0].set_title(
        "(a) Effect direction by observation quality",
        loc="left",
        pad=16,
        fontsize=10
    )

    for i in range(len(modes)):
        for j in range(len(labels)):

            value = values[i, j]

            axs[0].text(
                j,
                i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if abs(value) > .6*limit
                    else "#233D52"
                )
            )

    fig.colorbar(
        im,
        ax=axs[0],
        fraction=.042,
        pad=.035,
        label="Paired PAS reduction"
    )

    ax = axs[1]

    selected = (
        table[
            table.contrast ==
            "STATE_LOCAL vs STATIC_EXTRA"
        ]
        .set_index("mode")
        .reindex(modes)
    )

    xx = np.arange(len(modes))

    for i, row in enumerate(
        selected.itertuples()
    ):

        value = row.mean_gain

        color = (
            "#167F83"
            if row.CI95_low > 0
            else "#C18A45"
        )

        ax.errorbar(
            i,
            value,
            yerr=[
                [
                    max(
                        0,
                        value-row.CI95_low
                    )
                ],
                [
                    max(
                        0,
                        row.CI95_high-value
                    )
                ]
            ],
            fmt="o",
            markersize=6,
            color=color,
            capsize=4,
            linewidth=1.5
        )

    ax.axhline(
        0,
        color="#7D8F9C",
        linestyle="--",
        linewidth=1
    )

    ax.set_xticks(xx, modes)

    ax.set_ylabel(
        "PAS improvement vs static candidate"
    )

    ax.set_title(
        "(b) State-local contribution under sensor stress",
        loc="left",
        pad=16,
        fontsize=10
    )

    ax.grid(
        axis="y",
        alpha=.18
    )
    ax.set_axisbelow(True)

    for ext in [
        "pdf", "svg", "png"
    ]:

        fig.savefig(
            OUT /
            f"Exp51_observation_results.{ext}",
            dpi=260,
            bbox_inches="tight",
            pad_inches=.2
        )

    plt.close(fig)


# ==================================================
# 7. EXECUTION
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "pilot",
            "full"
        ]
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=None
    )

    args = parser.parse_args()

    n = (
        args.seeds
        if args.seeds is not None
        else (
            2 if args.phase == "pilot"
            else 20
        )
    )

    if not 2 <= n <= 100:
        raise ValueError(
            "Seed count must be 2-100"
        )

    verify_protocol(
        n,
        args.phase
    )

    graph, adj, weights = (
        e50.build_environment()
    )

    # Critical check: policy hook must replace
    # the simulator's original policy function.
    if (
        ref.run_one.__globals__.get(
            "choose"
        )
        is not observed_choose
    ):
        raise RuntimeError(
            "Observation hook not connected!"
        )

    selected_cases = e50.make_cases(
        8900 if args.phase == "pilot"
        else 9000,
        n
    )

    hashes = fingerprints()

    all_results = []

    global MODE

    for mode in MODES:

        MODE = mode

        name = (
            f"{args.phase}_{mode}"
        )

        print(
            "\n[RUN]",
            name,
            len(selected_cases),
            "paired cases",
            flush=True
        )

        outcomes, events = e49.run_block(
            name,
            selected_cases,
            0.0,
            0.0,
            METHODS,
            graph,
            adj,
            weights,
            hashes
        )

        if args.phase == "full":
            result = summarize(
                mode,
                outcomes,
                events
            )

            all_results.append(result)

        elif mode == "blind":
            pivot = outcomes.pivot(
                index=[
                    "pattern",
                    "budget",
                    "onset",
                    "seed"
                ],
                columns="method",
                values="PAS"
            )
            diff = (
                pivot.STATE_LOCAL
                - pivot.STATIC_EXTRA
            ).abs().max()
            if diff > 1e-12:
                raise RuntimeError(
                    "Blind pilot control failed"
                )

    if args.phase == "full":

        all_table = pd.concat(
            all_results,
            ignore_index=True
        )

        all_table.to_csv(
            OUT / "ALL_MODES_PAIRED.csv",
            index=False
        )

        plot_summary(all_table)

    print(
        "\nEXP51 COMPLETED:",
        args.phase,
        "\nOUTPUT:",
        OUT,
        flush=True
    )


if __name__ == "__main__":
    main()
