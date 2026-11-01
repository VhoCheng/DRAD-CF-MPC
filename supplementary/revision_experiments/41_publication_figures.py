from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "revision_experiments/results"
OUT = ROOT / "revision_experiments/figures/final_revision"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.facecolor": "white",
})

COLORS = {
    "CDP": "#8796A6",
    "FULL": "#24557A",
    "NO_GUARD": "#C4774B",
    "NO_DYNAMIC": "#68A59A",
    "NO_CF": "#9B86AE",
    "Legacy-K5": "#C4774B",
    "Confirm15-Robust": "#24557A",
}

def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0.12
        )
    plt.close(fig)
    print("[SAVED]", name, flush=True)

def grouped_bars(frame, methods, ylabel, filename,
                 multiplier=1.0):
    sizes = [14, 57, 118, 300]
    x = np.arange(len(sizes))
    width = 0.78 / len(methods)

    fig, ax = plt.subplots(figsize=(8.0, 3.8))

    for i, method in enumerate(methods):
        vals = []
        for size in sizes:
            row = frame[
                (frame["case_size"] == size)
                & (frame["method"] == method)
            ]
            if len(row) != 1:
                raise RuntimeError(
                    f"Missing data: {size}/{method}"
                )
            vals.append(
                float(row["value"].iloc[0]) * multiplier
            )

        ax.bar(
            x - 0.39 + (i + 0.5) * width,
            vals,
            width=width * 0.92,
            color=COLORS[method],
            label=method,
            edgecolor="white",
            linewidth=0.45
        )

    ax.set_xticks(x, [
        f"IEEE {n}" for n in sizes
    ])
    ax.set_ylabel(ylabel)
    ax.grid(
        axis="y",
        alpha=0.16,
        linewidth=0.6
    )
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        ncol=min(len(methods), 5),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17)
    )
    ax.set_ylim(bottom=0)
    save(fig, filename)

def plot_ablations():
    path = R / (
        "39_mechanism_full/condition_summary.csv"
    )
    df = pd.read_csv(path)
    clean = df[
        df["condition"] == "clean"
    ][[
        "case_size", "method", "mean_PAS"
    ]].rename(columns={"mean_PAS": "value"})

    grouped_bars(
        clean,
        ["CDP", "NO_CF", "NO_DYNAMIC",
         "FULL", "NO_GUARD"],
        "Mean affected-service proxy (PAS)",
        "Fig_Ablation_39"
    )

def plot_ablations_ci():
    path = R / (
        "39_mechanism_full/"
        "clean_paired_statistics.csv"
    )
    df = pd.read_csv(path)

    labels = [
        f"IEEE {int(row.case_size)}  "
        f"{row.comparison}"
        for row in df.itertuples()
    ]

    y = np.arange(len(df))
    x = df["absolute_gain"].to_numpy()
    lo = df["CI_low"].to_numpy()
    hi = df["CI_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.0, 6.7))

    ax.axvline(
        0, color="#555555",
        linewidth=0.85, linestyle="--"
    )

    for i, row in df.reset_index().iterrows():
        significant = (
            float(row["holm_p_16"]) < 0.05
        )
        ax.errorbar(
            x[i], y[i],
            xerr=[
                [max(0, x[i] - lo[i])],
                [max(0, hi[i] - x[i])]
            ],
            fmt="o",
            markersize=4.5,
            color=(
                "#24557A" if significant
                else "#8796A6"
            ),
            capsize=2,
            linewidth=1.0
        )

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Absolute PAS improvement (positive is better)"
    )
    ax.grid(
        axis="x", alpha=0.16, linewidth=0.6
    )
    ax.set_axisbelow(True)
    save(fig, "Fig_Ablation_Effects_39")

def plot_physical():
    path = R / (
        "35_37_final_evidence/"
        "35_physical_overview.csv"
    )
    df = pd.read_csv(path)

    frame = df[[
        "case_size", "variant", "mean_unserved"
    ]].rename(columns={
        "variant": "method",
        "mean_unserved": "value"
    })

    grouped_bars(
        frame,
        ["CDP", "Legacy-K5",
         "Confirm15-Robust"],
        "Modeled unavailable load (%)",
        "Fig_Physical_Impact_34",
        multiplier=100.0
    )

def plot_timing():
    path = R / (
        "35_37_final_evidence/"
        "37_serial_timing_summary.csv"
    )
    df = pd.read_csv(path).sort_values(
        "case_size"
    )

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.4, 3.7))

    ax.bar(
        x - 0.18,
        df["mean_stage1"],
        width=0.34,
        color="#24557A",
        label="Stage 1 (K5)"
    )
    ax.bar(
        x + 0.18,
        df["mean_stage2"],
        width=0.34,
        color="#68A59A",
        label="Stage 2 (K15)"
    )

    ax.set_xticks(
        x,
        [f"IEEE {int(n)}"
         for n in df["case_size"]]
    )
    ax.set_ylabel("Mean computation time (s)")
    ax.grid(
        axis="y", alpha=0.16, linewidth=0.6
    )
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig, "Fig_Computational_Cost_37")

def plot_calibration():
    path = R / (
        "35_37_final_evidence/"
        "36_guard_calibration_summary.csv"
    )
    df = pd.read_csv(path)

    methods = ["Original", "Balanced"]
    x = np.arange(2)
    width = 0.32

    def get_value(method, key):
        rows = df[df["method"] == method]
        if len(rows) != 1:
            raise RuntimeError("Missing calibration")
        return float(rows.iloc[0][key])

    predicted = [
        get_value(m, "K5_predicted_gain")
        for m in methods
    ]
    independent = [
        get_value(m, "K20_independent_gain")
        for m in methods
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axhline(0, color="#444444", lw=0.8)

    ax.bar(
        x - width/2, predicted, width,
        color="#24557A",
        label="K5 predicted gain"
    )
    ax.bar(
        x + width/2, independent, width,
        color="#C4774B",
        label="Independent K20 gain"
    )

    ax.set_xticks(x, methods)
    ax.set_ylabel("Mean predicted PAS improvement")
    ax.grid(
        axis="y", alpha=0.16, lw=0.6
    )
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig, "Fig_Counterfactual_Calibration_36")

def plot_paired_stress_if_ready():
    path = R / (
        "40_paired_observation_full/"
        "paired_stress_statistics.csv"
    )

    if not path.exists():
        print(
            "[PENDING] Exp40 paired observation results"
        )
        return

    df = pd.read_csv(path)
    if len(df) != 20:
        raise RuntimeError(
            "Exp40 stress statistics incomplete"
        )

    conditions = [
        "delay_10", "delay_30", "loss_20",
        "false_positive_05",
        "combined_delay30_loss20"
    ]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(13, 3.9),
        sharey=True
    )

    for ax, size in zip(
        axes, [14, 57, 118, 300]
    ):
        subset = df[
            df["case_size"] == size
        ].set_index("condition")

        ax.axvline(
            0, ls="--",
            color="#777777", lw=0.8
        )

        for j, condition in enumerate(conditions):
            row = subset.loc[condition]

            x = float(
                row["delta_PAS_stress_minus_clean"]
            )
            low = float(row["CI_low"])
            high = float(row["CI_high"])

            ax.errorbar(
                x, j,
                xerr=[
                    [max(0, x-low)],
                    [max(0, high-x)]
                ],
                fmt="o",
                color="#24557A",
                capsize=2,
                markersize=4,
                linewidth=1
            )

        ax.set_yticks(
            range(5),
            conditions
        )
        ax.invert_yaxis()
        ax.set_xlabel(
            f"IEEE {size}\nDelta PAS"
        )
        ax.grid(
            axis="x", alpha=0.16, lw=0.6
        )
        ax.set_axisbelow(True)

    save(fig, "Fig_Paired_Observation_40")

if __name__ == "__main__":
    plot_ablations()
    plot_ablations_ci()
    plot_physical()
    plot_timing()
    plot_calibration()
    plot_paired_stress_if_ready()

    print(
        "\nPUBLICATION FIGURE EXPORT COMPLETED"
    )
    print("Output directory:", OUT)
