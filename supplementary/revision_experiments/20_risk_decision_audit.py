from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_experiments/results"
SOURCE = OUT / "19_dynamic_risk_full"

sys.path.insert(0, str(ROOT / "src"))

import seid_malware as seid

VARIANTS = [
    "CDP",
    "Original",
    "Balanced",
    "Cyber-Priority",
    "Physical-Priority",
    "Infection-Priority",
    "Uncertainty-Priority",
]

COMPARISONS = [
    ("Balanced", "Original"),
    ("Cyber-Priority", "Original"),
    ("Physical-Priority", "Original"),
    ("Infection-Priority", "Original"),
    ("Uncertainty-Priority", "Original"),
    ("Balanced", "CDP"),
]

BOOTSTRAPS = 4000
RNG = np.random.default_rng(20261019)


def parse_nodes(value):
    if pd.isna(value) or str(value).strip() == "":
        return tuple()

    return tuple(sorted(
        int(x)
        for x in str(value).split("|")
        if x.strip()
    ))


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)

    corrected = np.maximum.accumulate(
        p[order] * (len(p) - np.arange(len(p)))
    )

    corrected = np.minimum(corrected, 1.0)

    result = np.empty(len(p))
    result[order] = corrected

    return result


def main():
    print("=" * 75)
    print("TNSE EXPERIMENT 20")
    print("RISK WEIGHT / SELECTED NODES / ATTACK ENTRY AUDIT")
    print("=" * 75)

    summary_path = SOURCE / "summary.csv"
    detail_path = SOURCE / "per_run.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    if not detail_path.exists():
        raise FileNotFoundError(detail_path)

    summary = pd.read_csv(summary_path)
    df = pd.read_csv(detail_path)

    if len(summary) != 20 * len(VARIANTS):
        raise RuntimeError(
            f"Unexpected summary rows: {len(summary)}"
        )

    if len(df) != 20 * 8 * len(VARIANTS):
        raise RuntimeError(
            f"Unexpected run rows: {len(df)}"
        )

    if set(df["variant"]) != set(VARIANTS):
        raise RuntimeError("Unexpected variants")

    keys = ["scenario_id", "run_index"]

    if df.duplicated(keys + ["variant"]).any():
        raise RuntimeError("Duplicate run records")

    if summary.duplicated(
        ["scenario_id", "variant"]
    ).any():
        raise RuntimeError("Duplicate summary records")

    counts = df.groupby(keys)["variant"].nunique()

    if not (counts == len(VARIANTS)).all():
        raise RuntimeError("Incomplete paired comparisons")

    seed_counts = df.groupby(keys)["seed"].nunique()

    if not (seed_counts == 1).all():
        raise RuntimeError(
            "Different random seeds within a paired run"
        )

    df["node_tuple"] = df["protected_nodes"].apply(
        parse_nodes
    )

    if not df["node_tuple"].map(len).eq(df["budget"]).all():
        raise RuntimeError(
            "Selected protected-node count differs from budget"
        )

    originals = (
        df[df["variant"] == "Original"][
            keys + [
                "node_tuple",
                "selected_policy",
                "PAS",
            ]
        ]
        .rename(columns={
            "node_tuple": "original_nodes",
            "selected_policy": "original_policy",
            "PAS": "original_PAS",
        })
    )

    df = df.merge(
        originals,
        on=keys,
        validate="many_to_one",
    )

    df["final_nodes_changed"] = (
        df["node_tuple"] != df["original_nodes"]
    )

    df["policy_label_changed"] = (
        df["selected_policy"] != df["original_policy"]
    )

    df["same_label_different_nodes"] = (
        df["final_nodes_changed"]
        & ~df["policy_label_changed"]
    )

    df["node_overlap"] = df.apply(
        lambda row: (
            len(
                set(row["node_tuple"])
                & set(row["original_nodes"])
            )
            / max(1, len(row["original_nodes"]))
        ),
        axis=1,
    )

    print("CHECKING ORIGINAL ATTACK-ENTRY RULE ...")

    graph = nx.barabasi_albert_graph(
        118, 2, seed=42
    )

    # Reconstruct actual attack starting nodes
    # using exactly the source-code selection rule.
    # No new SEID propagation simulations are run.
    def get_initial(row):
        initial_node, _ = (
            seid.choose_initial_infection_node(
                Gc=graph,
                attack_information_ratio=float(row["info"]),
                protected_nodes=list(row["node_tuple"]),
                seed=int(row["seed"]),
            )
        )
        return int(initial_node)

    df["initial_attack_node"] = df.apply(
        get_initial, axis=1
    )

    original_entries = (
        df[df["variant"] == "Original"][
            keys + ["initial_attack_node"]
        ]
        .rename(columns={
            "initial_attack_node": "original_attack_node"
        })
    )

    df = df.merge(
        original_entries,
        on=keys,
        validate="many_to_one",
    )

    df["attack_entry_changed"] = (
        df["initial_attack_node"]
        != df["original_attack_node"]
    )

    # Validate that original reported EP is reconstructed
    # exactly from the stored Monte Carlo outcomes.
    reconstructed = (
        df.groupby(
            ["scenario_id", "variant"],
            as_index=False,
        )
        .agg(
            PAS_check=("PAS", "mean"),
            failure_rate=("failed", "mean"),
        )
    )

    reconstructed["EP_check"] = (
        reconstructed["PAS_check"]
        * (1.0 - reconstructed["failure_rate"])
    )

    checked = summary.merge(
        reconstructed,
        on=["scenario_id", "variant"],
        validate="one_to_one",
    )

    max_error = float(np.max(np.abs(
        checked["EP"] - checked["EP_check"]
    )))

    if max_error > 1e-10:
        raise RuntimeError(
            f"EP reconstruction failed: {max_error}"
        )

    print(
        "EP RECONSTRUCTION PASSED:",
        f"max_error={max_error:.3e}"
    )

    print("COMPLETE DATA VALIDATION PASSED")

    # Real selected-node changes versus Original.
    node_audit = (
        df.groupby("variant", as_index=False)
        .agg(
            final_node_change_rate=(
                "final_nodes_changed", "mean"
            ),
            policy_label_change_rate=(
                "policy_label_changed", "mean"
            ),
            same_label_node_change_rate=(
                "same_label_different_nodes", "mean"
            ),
            attack_entry_change_rate=(
                "attack_entry_changed", "mean"
            ),
            mean_protected_node_overlap=(
                "node_overlap", "mean"
            ),
        )
    )

    # Prepare a paired scenario x MC x variant matrix.
    ids = sorted(df["scenario_id"].unique())

    pas_wide = df.pivot(
        index=keys,
        columns="variant",
        values="PAS",
    )

    failed_wide = df.pivot(
        index=keys,
        columns="variant",
        values="failed",
    )

    pas = np.stack([
        pas_wide.loc[sid, VARIANTS].to_numpy(
            dtype=float
        )
        for sid in ids
    ])

    failed = np.stack([
        failed_wide.loc[sid, VARIANTS].to_numpy(
            dtype=float
        )
        for sid in ids
    ])

    if pas.shape != (20, 8, len(VARIANTS)):
        raise RuntimeError("Invalid paired tensor shape")

    ep = (
        pas.mean(axis=1)
        * (1.0 - failed.mean(axis=1))
    )

    index = {
        name: i
        for i, name in enumerate(VARIANTS)
    }

    meta = (
        summary.drop_duplicates("scenario_id")
        .set_index("scenario_id")
        .loc[ids]
    )

    patterns = meta["pattern"].to_numpy()

    groups = [
        np.where(patterns == p)[0]
        for p in sorted(set(patterns))
    ]

    if len(groups) != 5 or any(
        len(group) != 4 for group in groups
    ):
        raise RuntimeError(
            "Unexpected coupling/scenario distribution"
        )

    # Two-level paired bootstrap:
    # resample scenarios within coupling patterns,
    # then resample paired MC repetitions.
    draw_scenarios = np.concatenate([
        RNG.choice(
            group,
            size=(BOOTSTRAPS, len(group)),
            replace=True,
        )
        for group in groups
    ], axis=1)

    draw_runs = RNG.integers(
        0, 8,
        size=(BOOTSTRAPS, 20, 8),
    )

    boot_pas = pas[
        draw_scenarios[:, :, None],
        draw_runs,
        :
    ].mean(axis=2)

    boot_ps = (
        1.0
        - failed[
            draw_scenarios[:, :, None],
            draw_runs,
            :
        ].mean(axis=2)
    )

    boot_ep = boot_pas * boot_ps

    result_rows = []

    for proposed, baseline in COMPARISONS:
        a = index[proposed]
        b = index[baseline]

        # Positive = proposed has lower EP.
        gains = ep[:, b] - ep[:, a]

        boot_gains = (
            boot_ep[:, :, b]
            - boot_ep[:, :, a]
        ).mean(axis=1)

        lo, hi = np.quantile(
            boot_gains,
            [0.025, 0.975],
        )

        if np.all(np.abs(gains) < 1e-14):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(
                gains,
                alternative="two-sided",
                zero_method="wilcox",
            ).pvalue)

        mean_baseline = float(ep[:, b].mean())
        mean_proposed = float(ep[:, a].mean())

        result_rows.append({
            "proposed": proposed,
            "baseline": baseline,
            "baseline_EP": mean_baseline,
            "proposed_EP": mean_proposed,
            "mean_EP_gain": float(gains.mean()),
            "gain_pct_of_means": (
                100.0
                * gains.mean()
                / mean_baseline
            ),
            "bootstrap_CI_low": float(lo),
            "bootstrap_CI_high": float(hi),
            "wilcoxon_p": p_value,
            "better_scenarios": int(
                (gains > 1e-12).sum()
            ),
            "worse_scenarios": int(
                (gains < -1e-12).sum()
            ),
            "tied_scenarios": int(
                (np.abs(gains) <= 1e-12).sum()
            ),
            "worse_MC_rate": float(
                np.mean(
                    pas[:, :, a]
                    > pas[:, :, b] + 1e-12
                )
            ),
        })

    paired_stats = pd.DataFrame(result_rows)

    paired_stats["holm_adjusted_p"] = (
        holm_adjust(
            paired_stats["wilcoxon_p"].to_numpy()
        )
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    stats_path = (
        OUT / f"20_risk_paired_stats_{stamp}.csv"
    )

    nodes_path = (
        OUT / f"20_risk_decision_audit_{stamp}.csv"
    )

    detail_out = (
        OUT / f"20_risk_per_run_audit_{stamp}.csv"
    )

    paired_stats.to_csv(
        stats_path,
        index=False,
    )

    node_audit.to_csv(
        nodes_path,
        index=False,
    )

    # Convert node tuples back to readable strings.
    df["node_tuple"] = df["node_tuple"].map(str)
    df["original_nodes"] = (
        df["original_nodes"].map(str)
    )

    df.to_csv(
        detail_out,
        index=False,
    )

    print("\n" + "=" * 75)
    print("RISK WEIGHT PAIRED STATISTICAL RESULTS")
    print("=" * 75)
    print(paired_stats.to_string(index=False))

    print("\n" + "=" * 75)
    print("ACTUAL DECISION AND ATTACK ENTRY AUDIT")
    print("=" * 75)
    print(node_audit.to_string(index=False))

    print("\n" + "=" * 75)
    print("BALANCED SELECTED POLICY DISTRIBUTION")
    print("=" * 75)

    print(
        df[df["variant"] == "Balanced"][
            "selected_policy"
        ].value_counts().to_string()
    )

    print("\nSaved paired statistics:", stats_path)
    print("Saved node-level audit:", nodes_path)
    print("Saved per-run audit:", detail_out)

    print("EXPERIMENT 20 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
