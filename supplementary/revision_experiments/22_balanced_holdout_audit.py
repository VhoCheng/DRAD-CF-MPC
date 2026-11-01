from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import seid_malware as seid

OUT = ROOT / "revision_experiments/results"
SOURCE = OUT / "21_balanced_holdout_full"

VARIANTS = ["CDP", "Original", "Balanced"]

COMPARISONS = [
    ("Balanced", "Original"),
    ("Balanced", "CDP"),
]

B = 4000
RNG = np.random.default_rng(20261019)


def nodes_from_string(value):
    if pd.isna(value):
        return tuple()

    value = str(value).strip()

    if not value:
        return tuple()

    return tuple(sorted(
        int(item)
        for item in value.split("|")
        if item.strip()
    ))


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    n = len(p)

    ranked = p[order] * (
        n - np.arange(n)
    )

    ranked = np.maximum.accumulate(ranked)
    ranked = np.minimum(ranked, 1.0)

    result = np.empty(n, dtype=float)
    result[order] = ranked
    return result


def main():
    print("=" * 75)
    print("TNSE EXPERIMENT 22")
    print("FROZEN BALANCED HOLDOUT STATISTICAL AUDIT")
    print("=" * 75)

    summary_path = SOURCE / "summary.csv"
    details_path = SOURCE / "per_run.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    if not details_path.exists():
        raise FileNotFoundError(details_path)

    summary = pd.read_csv(summary_path)
    details = pd.read_csv(details_path)

    if len(summary) != 120:
        raise RuntimeError(
            f"Expected 120 summary rows, got {len(summary)}"
        )

    if len(details) != 1440:
        raise RuntimeError(
            f"Expected 1440 run rows, got {len(details)}"
        )

    if set(summary["variant"]) != set(VARIANTS):
        raise RuntimeError("Unexpected methods")

    if summary["scenario_id"].nunique() != 40:
        raise RuntimeError("Expected 40 scenarios")

    if summary.duplicated(
        ["scenario_id", "variant"]
    ).any():
        raise RuntimeError("Duplicate summary records")

    run_key = ["scenario_id", "run_index"]

    if details.duplicated(
        run_key + ["variant"]
    ).any():
        raise RuntimeError("Duplicate run records")

    paired_counts = (
        details.groupby(run_key)["variant"].nunique()
    )

    if not (paired_counts == 3).all():
        raise RuntimeError(
            "Missing method in paired observations"
        )

    run_counts = details.groupby(
        ["scenario_id", "variant"]
    )["run_index"].nunique()

    if not (run_counts == 12).all():
        raise RuntimeError(
            "Expected 12 Monte Carlo runs per setting"
        )

    seed_counts = (
        details.groupby(run_key)["seed"].nunique()
    )

    if not (seed_counts == 1).all():
        raise RuntimeError(
            "Paired methods use different random seeds"
        )

    ids = sorted(
        summary["scenario_id"].unique()
    )

    # Reconstruct original code's EP exactly:
    # EP = mean(PAS) * (1 - mean(attack_failed)).
    pas_wide = details.pivot(
        index=run_key,
        columns="variant",
        values="PAS",
    )

    failed_wide = details.pivot(
        index=run_key,
        columns="variant",
        values="failed",
    )

    if (
        pas_wide.isna().any().any()
        or failed_wide.isna().any().any()
    ):
        raise RuntimeError(
            "Missing PAS or attack-failure observations"
        )

    pas = np.stack([
        pas_wide.loc[
            [(sid, r) for r in range(12)],
            VARIANTS
        ].to_numpy(dtype=float)
        for sid in ids
    ])

    failed = np.stack([
        failed_wide.loc[
            [(sid, r) for r in range(12)],
            VARIANTS
        ].to_numpy(dtype=float)
        for sid in ids
    ])

    assert pas.shape == (40, 12, 3)
    assert failed.shape == (40, 12, 3)

    ep = (
        pas.mean(axis=1)
        * (1.0 - failed.mean(axis=1))
    )

    reference_ep = (
        summary.pivot(
            index="scenario_id",
            columns="variant",
            values="EP",
        )
        .loc[ids, VARIANTS]
        .to_numpy(dtype=float)
    )

    max_error = float(np.max(
        np.abs(ep - reference_ep)
    ))

    if max_error > 1e-10:
        raise RuntimeError(
            f"EP RECONSTRUCTION FAILED: {max_error}"
        )

    print(
        "EP RECONSTRUCTION PASSED:",
        f"max_error={max_error:.3e}",
        flush=True,
    )

    print(
        "DATA VALIDATION PASSED:",
        "40 scenarios, 12 MC, 3 methods",
        flush=True,
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
        np.where(patterns == name)[0]
        for name in sorted(set(patterns))
    ]

    if (
        len(groups) != 5
        or any(len(group) != 8 for group in groups)
    ):
        raise RuntimeError(
            "Expected five couplings and eight settings each"
        )

    # Two-level, paired, stratified bootstrap.
    # Resample tested scenarios within each coupling;
    # resample Monte Carlo runs jointly across methods.
    sampled_scenarios = np.concatenate([
        RNG.choice(
            group,
            size=(B, len(group)),
            replace=True,
        )
        for group in groups
    ], axis=1)

    sampled_runs = RNG.integers(
        0,
        12,
        size=(B, 40, 12),
    )

    bootstrap_pas = pas[
        sampled_scenarios[:, :, None],
        sampled_runs,
        :
    ].mean(axis=2)

    bootstrap_ps = (
        1.0
        - failed[
            sampled_scenarios[:, :, None],
            sampled_runs,
            :
        ].mean(axis=2)
    )

    bootstrap_ep = (
        bootstrap_pas * bootstrap_ps
    )

    result_rows = []

    for proposed, baseline in COMPARISONS:
        a = index[proposed]
        b = index[baseline]

        # Positive means proposed improves EP.
        paired_gain = ep[:, b] - ep[:, a]

        bootstrap_baseline = (
            bootstrap_ep[:, :, b].mean(axis=1)
        )

        bootstrap_proposed = (
            bootstrap_ep[:, :, a].mean(axis=1)
        )

        bootstrap_gain = (
            bootstrap_baseline
            - bootstrap_proposed
        )

        ci = np.quantile(
            bootstrap_gain,
            [0.025, 0.975],
        )

        bootstrap_pct = np.where(
            bootstrap_baseline > 1e-12,
            100.0
            * bootstrap_gain
            / bootstrap_baseline,
            np.nan,
        )

        pct_ci = np.nanquantile(
            bootstrap_pct,
            [0.025, 0.975],
        )

        if np.all(
            np.abs(paired_gain) < 1e-14
        ):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(
                paired_gain,
                alternative="two-sided",
                zero_method="wilcox",
            ).pvalue)

        ep_baseline = float(
            ep[:, b].mean()
        )

        ep_proposed = float(
            ep[:, a].mean()
        )

        valid = ep[:, b] > 1e-12

        scenario_average_gain_pct = (
            float(np.mean(
                100.0
                * paired_gain[valid]
                / ep[valid, b]
            ))
            if np.any(valid)
            else float("nan")
        )

        result_rows.append({
            "comparison": (
                f"{proposed} vs {baseline}"
            ),
            "mean_baseline_EP": ep_baseline,
            "mean_proposed_EP": ep_proposed,
            "mean_EP_gain": float(
                paired_gain.mean()
            ),
            "gain_pct_of_means": (
                100.0
                * paired_gain.mean()
                / ep_baseline
            ),
            "mean_scenario_gain_pct": (
                scenario_average_gain_pct
            ),
            "bootstrap_CI_low": float(ci[0]),
            "bootstrap_CI_high": float(ci[1]),
            "bootstrap_pct_CI_low": float(
                pct_ci[0]
            ),
            "bootstrap_pct_CI_high": float(
                pct_ci[1]
            ),
            "wilcoxon_p": p_value,
            "better_scenarios": int(
                np.sum(paired_gain > 1e-12)
            ),
            "worse_scenarios": int(
                np.sum(paired_gain < -1e-12)
            ),
            "tied_scenarios": int(
                np.sum(
                    np.abs(paired_gain) <= 1e-12
                )
            ),
            "worse_MC_PAS_rate": float(
                np.mean(
                    pas[:, :, a]
                    > pas[:, :, b] + 1e-12
                )
            ),
            "mean_unconditional_PAS_gain": float(
                pas[:, :, b].mean()
                - pas[:, :, a].mean()
            ),
        })

    results = pd.DataFrame(result_rows)

    results["holm_adjusted_p"] = (
        holm_adjust(
            results["wilcoxon_p"].to_numpy()
        )
    )

    # Compare the actual protected-node sets.
    # The same policy label may protect different nodes.
    details["node_set"] = (
        details["protected_nodes"].apply(
            nodes_from_string
        )
    )

    expected_budget = (
        details["budget"].to_numpy(dtype=int)
    )

    actual_budget = (
        details["node_set"]
        .map(len)
        .to_numpy(dtype=int)
    )

    if not np.array_equal(
        expected_budget, actual_budget
    ):
        raise RuntimeError(
            "Protected-node count differs from budget"
        )

    nodes_wide = details.pivot(
        index=run_key,
        columns="variant",
        values="node_set",
    ).sort_index()

    policy_wide = details.pivot(
        index=run_key,
        columns="variant",
        values="selected_policy",
    ).sort_index()

    # Reconstruct the actual original attack-entry rule.
    graph = nx.barabasi_albert_graph(
        n=118,
        m=2,
        seed=42,
    )

    def attack_entry(row):
        node, _ = (
            seid.choose_initial_infection_node(
                Gc=graph,
                attack_information_ratio=float(
                    row["info"]
                ),
                protected_nodes=list(
                    row["node_set"]
                ),
                seed=int(row["seed"]),
            )
        )
        return int(node)

    details["attack_entry"] = details.apply(
        attack_entry,
        axis=1,
    )

    entry_wide = details.pivot(
        index=run_key,
        columns="variant",
        values="attack_entry",
    ).sort_index()

    decision_rows = []

    for baseline in ["Original", "CDP"]:
        left_nodes = list(nodes_wide["Balanced"])
        right_nodes = list(nodes_wide[baseline])

        node_changed = np.array([
            a != b
            for a, b in zip(
                left_nodes, right_nodes
            )
        ], dtype=bool)

        left_policy = (
            policy_wide["Balanced"].to_numpy()
        )

        right_policy = (
            policy_wide[baseline].to_numpy()
        )

        label_changed = (
            left_policy != right_policy
        )

        entry_changed = (
            entry_wide["Balanced"].to_numpy()
            != entry_wide[baseline].to_numpy()
        )

        overlaps = np.array([
            len(set(a) & set(b))
            / max(1, len(a))
            for a, b in zip(
                left_nodes, right_nodes
            )
        ])

        decision_rows.append({
            "comparison": (
                f"Balanced vs {baseline}"
            ),
            "paired_runs": len(node_changed),
            "policy_label_change_rate": float(
                label_changed.mean()
            ),
            "actual_node_change_rate": float(
                node_changed.mean()
            ),
            "same_label_different_nodes_rate": (
                float(
                    np.mean(
                        node_changed & ~label_changed
                    )
                )
            ),
            "mean_protected_node_overlap": (
                float(overlaps.mean())
            ),
            "attack_entry_change_rate": float(
                entry_changed.mean()
            ),
        })

    decision_df = pd.DataFrame(
        decision_rows
    )

    # Identify scenarios responsible for deterioration.
    scenario_df = meta[
        ["pattern", "info", "budget", "onset"]
    ].copy()

    for name in VARIANTS:
        scenario_df[f"{name}_EP"] = (
            ep[:, index[name]]
        )

    scenario_df["Balanced_gain_vs_CDP"] = (
        scenario_df["CDP_EP"]
        - scenario_df["Balanced_EP"]
    )

    scenario_df["Balanced_gain_vs_Original"] = (
        scenario_df["Original_EP"]
        - scenario_df["Balanced_EP"]
    )

    by_coupling = (
        scenario_df.groupby("pattern")
        .agg(
            CDP_EP=("CDP_EP", "mean"),
            Original_EP=("Original_EP", "mean"),
            Balanced_EP=("Balanced_EP", "mean"),
            Balanced_gain_vs_CDP=(
                "Balanced_gain_vs_CDP", "mean"
            ),
            Balanced_gain_vs_Original=(
                "Balanced_gain_vs_Original", "mean"
            ),
        )
        .reset_index()
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    prefix = f"22_balanced_audit_{stamp}"

    stats_path = (
        OUT / f"{prefix}_paired_stats.csv"
    )

    decision_path = (
        OUT / f"{prefix}_decision_audit.csv"
    )

    scenario_path = (
        OUT / f"{prefix}_by_scenario.csv"
    )

    coupling_path = (
        OUT / f"{prefix}_by_coupling.csv"
    )

    results.to_csv(
        stats_path, index=False
    )

    decision_df.to_csv(
        decision_path, index=False
    )

    scenario_df.to_csv(
        scenario_path
    )

    by_coupling.to_csv(
        coupling_path, index=False
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 22 - PAIRED STATISTICAL RESULTS")
    print("=" * 75)
    print(results.to_string(index=False))

    print("\n" + "=" * 75)
    print("EXPERIMENT 22 - ACTUAL DECISION AUDIT")
    print("=" * 75)
    print(decision_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("EXPERIMENT 22 - RESULTS BY COUPLING")
    print("=" * 75)
    print(by_coupling.to_string(index=False))

    print("\n" + "=" * 75)
    print("FIVE WORST BALANCED SCENARIOS VS CDP")
    print("=" * 75)

    print(
        scenario_df.sort_values(
            "Balanced_gain_vs_CDP"
        ).head(5).to_string()
    )

    print("\nPositive EP gain = lower attack payoff.")
    print(
        "Bootstrap uncertainty is conditional "
        "on the tested scenario design."
    )

    print("\nSaved:", stats_path)
    print("Saved:", decision_path)
    print("Saved:", scenario_path)
    print("Saved:", coupling_path)

    print(
        "EXPERIMENT 22 COMPLETED SUCCESSFULLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
