import os
import sys
import pandas as pd
from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from run_experiments_core import run_single_setting


def run_one_task(task):
    (
        pattern,
        strategy,
        info_ratio,
        budget,
        onset_time,
        n_runs,
        lookahead_runs,
    ) = task

    actual_budget = 0 if strategy.upper() == "NP" else budget

    row = run_single_setting(
        pattern=pattern,
        strategy=strategy,
        attack_information_ratio=info_ratio,
        defense_budget=actual_budget,
        onset_time=onset_time,
        n_runs=n_runs,
        lookahead_runs=lookahead_runs,
    )

    return row


def run_experiments_parallel(
    patterns=None,
    strategies=None,
    attack_information_ratios=None,
    defense_budgets=None,
    onset_times=None,
    n_runs=20,
    lookahead_runs=20,
    n_jobs=20,
    out_dir="outputs/tables",
):
    os.makedirs(out_dir, exist_ok=True)

    if patterns is None:
        patterns = ["DDAC", "DDDC", "DCAC", "DCDC", "RC"]

    if strategies is None:
        strategies = ["NP", "RP", "CDP", "PDP", "CP", "DRAD", "DRAD-CF-MPC"]

    if attack_information_ratios is None:
        attack_information_ratios = [0.25, 0.50, 1.00]

    if defense_budgets is None:
        defense_budgets = [0, 20, 40, 60, 80, 100]

    if onset_times is None:
        onset_times = list(range(0, 301, 30))

    tasks = list(
        product(
            patterns,
            strategies,
            attack_information_ratios,
            defense_budgets,
            onset_times,
            [n_runs],
            [lookahead_runs],
        )
    )

    print("=" * 80)
    print("Parallel experiment configuration")
    print("=" * 80)
    print(f"Total tasks: {len(tasks)}")
    print(f"n_runs per task: {n_runs}")
    print(f"lookahead_runs per MPC task: {lookahead_runs}")
    print(f"n_jobs: {n_jobs}")
    print("=" * 80)

    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_one_task)(task) for task in tqdm(tasks, desc="Parallel experiments")
    )

    df = pd.DataFrame(rows)

    raw_out = os.path.join(out_dir, "experiment_results_raw_parallel.csv")
    df.to_csv(raw_out, index=False)

    group_cols = [
        "pattern",
        "strategy",
        "attack_information_ratio",
        "defense_budget",
    ]

    idx = df.groupby(group_cols)["EP"].idxmax()
    opt_df = df.loc[idx].copy()

    opt_df = opt_df.rename(
        columns={
            "onset_time": "T_opt",
            "EP": "EP_max",
            "PAS": "PAS_at_T_opt",
            "PS": "PS_at_T_opt",
        }
    )

    opt_out = os.path.join(out_dir, "experiment_results_optimal_parallel.csv")
    opt_df.to_csv(opt_out, index=False)

    summary_df = (
        opt_df.groupby("strategy")
        .agg(
            mean_EP_max=("EP_max", "mean"),
            mean_PAS_at_T_opt=("PAS_at_T_opt", "mean"),
            mean_PS_at_T_opt=("PS_at_T_opt", "mean"),
            mean_RT=("RT", "mean"),
            mean_RI=("RI", "mean"),
            mean_RE=("RE", "mean"),
        )
        .reset_index()
        .sort_values("mean_EP_max", ascending=True)
    )

    summary_out = os.path.join(out_dir, "experiment_strategy_summary_parallel.csv")
    summary_df.to_csv(summary_out, index=False)

    print("\nParallel experiments finished.")
    print(f"Raw results saved to: {raw_out}")
    print(f"Optimal results saved to: {opt_out}")
    print(f"Strategy summary saved to: {summary_out}")

    print("\nStrategy-level summary, lower EP/PAS/RT is better; higher RI/RE is better:")
    print(summary_df)

    if "selected_policy_mode" in opt_df.columns:
        mpc_df = opt_df[opt_df["strategy"].str.contains("MPC", case=False, na=False)].copy()
        if len(mpc_df) > 0:
            policy_summary = (
                mpc_df["selected_policy_mode"]
                .value_counts()
                .reset_index()
            )
            policy_summary.columns = ["selected_policy", "count"]

            policy_out = os.path.join(out_dir, "mpc_policy_switching_summary_parallel.csv")
            policy_summary.to_csv(policy_out, index=False)

            print("\nMPC selected policy distribution:")
            print(policy_summary)

    if "safe_guard_mode" in opt_df.columns:
        guard_df = opt_df[opt_df["strategy"].str.contains("MPC", case=False, na=False)].copy()
        if len(guard_df) > 0:
            guard_summary = (
                guard_df["safe_guard_mode"]
                .value_counts()
                .reset_index()
            )
            guard_summary.columns = ["safe_guard_mode", "count"]

            guard_out = os.path.join(out_dir, "mpc_safe_guard_summary_parallel.csv")
            guard_summary.to_csv(guard_out, index=False)

            print("\nMPC safe guard distribution:")
            print(guard_summary)

    return df, opt_df, summary_df


if __name__ == "__main__":
    # 服务器 25 核，先用 20 核比较稳，不要直接 25 核拉满。
    # 快速验证可以先 n_runs=5, lookahead_runs=5。
    # 正式实验建议 n_runs=100, lookahead_runs=30。
    run_experiments_parallel(
        patterns=["DDAC", "DDDC", "DCAC", "DCDC", "RC"],
        strategies=["NP", "RP", "CDP", "PDP", "CP", "DRAD", "DRAD-CF-MPC"],
        attack_information_ratios=[0.25, 0.50, 1.00],
        defense_budgets=[0, 20, 40, 60, 80, 100],
        onset_times=list(range(0, 301, 30)),
        n_runs=50,
        lookahead_runs=20,
        n_jobs=20
    )