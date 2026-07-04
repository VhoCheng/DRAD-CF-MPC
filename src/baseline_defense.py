import os
import numpy as np
import pandas as pd


def select_protected_nodes(
    strategy: str,
    defense_budget: int,
    pattern: str = "RC",
    seed: int = 42,
    table_dir: str = "outputs/tables",
):
    """
    Select protected communication nodes according to baseline defense strategies.

    Strategies:
        NP: no protection
        RP: random protection
        CDP: communication degree-based protection
        PDP: power degree-based protection
        CP: capacity-based protection
    """
    strategy = strategy.upper()

    comm_features = pd.read_csv(os.path.join(table_dir, "comm_features.csv"))
    power_features = pd.read_csv(os.path.join(table_dir, "power_features.csv"))
    coupling = pd.read_csv(os.path.join(table_dir, f"coupling_{pattern}.csv"))

    if strategy == "NP":
        return []

    defense_budget = int(defense_budget)
    defense_budget = max(0, min(defense_budget, len(comm_features)))

    if defense_budget == 0:
        return []

    if strategy == "RP":
        rng = np.random.default_rng(seed)
        nodes = comm_features["comm_node"].values
        return sorted(rng.choice(nodes, size=defense_budget, replace=False).astype(int).tolist())

    if strategy == "CDP":
        selected = (
            comm_features.sort_values("comm_degree", ascending=False)
            .head(defense_budget)["comm_node"]
            .astype(int)
            .tolist()
        )
        return sorted(selected)

    # For PDP and CP, first map communication nodes to their coupled power nodes.
    merged = coupling.merge(
        power_features,
        on="power_node",
        how="left"
    )

    if strategy == "PDP":
        selected = (
            merged.sort_values("power_degree", ascending=False)
            .head(defense_budget)["comm_node"]
            .astype(int)
            .tolist()
        )
        return sorted(selected)

    if strategy == "CP":
        selected = (
            merged.sort_values("capacity_proxy", ascending=False)
            .head(defense_budget)["comm_node"]
            .astype(int)
            .tolist()
        )
        return sorted(selected)

    raise ValueError(f"Unknown strategy: {strategy}")


def build_baseline_defense_table(
    strategies=None,
    patterns=None,
    defense_budgets=None,
    seed: int = 42,
    out_dir: str = "outputs/tables",
):
    """
    Generate a table listing protected nodes under each strategy, pattern, and budget.
    """
    os.makedirs(out_dir, exist_ok=True)

    if strategies is None:
        strategies = ["NP", "RP", "CDP", "PDP", "CP"]

    if patterns is None:
        patterns = ["DDAC", "DDDC", "DCAC", "DCDC", "RC"]

    if defense_budgets is None:
        defense_budgets = [0, 20, 40, 60, 80, 100]

    rows = []

    for pattern in patterns:
        for strategy in strategies:
            for budget in defense_budgets:
                protected_nodes = select_protected_nodes(
                    strategy=strategy,
                    defense_budget=budget,
                    pattern=pattern,
                    seed=seed + budget,
                    table_dir=out_dir,
                )

                rows.append({
                    "pattern": pattern,
                    "strategy": strategy,
                    "defense_budget": budget,
                    "n_protected": len(protected_nodes),
                    "protected_nodes": ",".join(map(str, protected_nodes)),
                })

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "baseline_defense_nodes.csv")
    df.to_csv(out_path, index=False)

    print("\nBaseline defense table generated.")
    print(df.head(10))
    print(f"\nSaved to: {out_path}")

    return df


if __name__ == "__main__":
    build_baseline_defense_table()