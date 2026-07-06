from __future__ import annotations

import compileall
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


EXPECTED_MAIN = {
    "DRAD-CF-MPC": (0.289, 0.306, 6.638, 0.827, 0.444),
    "CDP": (0.298, 0.315, 6.765, 0.822, 0.439),
    "DRAD": (0.348, 0.369, 7.657, 0.796, 0.402),
    "CP": (0.465, 0.499, 9.697, 0.716, 0.319),
    "PDP": (0.465, 0.503, 9.770, 0.714, 0.316),
    "RP": (0.567, 0.611, 11.529, 0.647, 0.244),
    "NP": (0.786, 0.848, 17.968, 0.449, 0.152),
}


def check_main_table() -> None:
    path = ROOT / "outputs" / "tables" / "experiment_strategy_summary_parallel.csv"
    df = pd.read_csv(path)
    required = {
        "strategy",
        "mean_EP_max",
        "mean_PAS_at_T_opt",
        "mean_RT",
        "mean_RI",
        "mean_RE",
    }
    missing = required - set(df.columns)
    if missing:
        raise AssertionError(f"Missing expected columns in {path}: {sorted(missing)}")

    table = df.set_index("strategy")
    for strategy, expected in EXPECTED_MAIN.items():
        actual = (
            round(float(table.loc[strategy, "mean_EP_max"]), 3),
            round(float(table.loc[strategy, "mean_PAS_at_T_opt"]), 3),
            round(float(table.loc[strategy, "mean_RT"]), 3),
            round(float(table.loc[strategy, "mean_RI"]), 3),
            round(float(table.loc[strategy, "mean_RE"]), 3),
        )
        if actual != expected:
            raise AssertionError(f"{strategy}: expected {expected}, got {actual}")


def check_no_old_terms() -> None:
    banned = ("PLL", "load loss", "lost load", "evaluate_pll")
    checked_suffixes = {".py", ".tex", ".csv", ".md"}
    allow = {"quick_check.py"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in checked_suffixes:
            continue
        if path.name in allow:
            continue
        text = path.read_text(errors="ignore")
        lowered = text.lower()
        for term in banned:
            haystack = text if term.isupper() else lowered
            needle = term if term.isupper() else term.lower()
            if needle in haystack:
                raise AssertionError(f"Old term {term!r} found in {path}")


def check_python_syntax() -> None:
    ok = compileall.compile_dir(ROOT / "src", quiet=1)
    if not ok:
        raise AssertionError("Python syntax check failed")


def main() -> None:
    check_main_table()
    check_no_old_terms()
    check_python_syntax()
    print("Quick check passed: main table, PAS terminology, and Python syntax are OK.")


if __name__ == "__main__":
    main()
