from pathlib import Path
import sys
import time
import json
import platform

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_experiments_core import run_single_setting


def main():

    print("=" * 70)
    print("TNSE MAJOR REVISION")
    print("EXPERIMENT 00: ORIGINAL BASELINE SMOKE TEST")
    print("=" * 70)

    config = {
        "pattern": "DDAC",
        "attack_information_ratio": 0.50,
        "defense_budget": 60,
        "onset_time": 150,
        "n_runs": 2,
        "lookahead_runs": 2,
        "infection_rate": 0.01,
        "activation_rate": 0.15,
        "detection_rate": 1e-5,
        "seed": 42,
    }

    output_dir = ROOT / "revision_experiments/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for strategy in ["CDP", "DRAD-CF-MPC"]:

        print(f"\nRunning {strategy} ...", flush=True)

        start = time.perf_counter()

        result = run_single_setting(
            strategy=strategy,
            **config
        )

        elapsed = time.perf_counter() - start

        result["wall_time_sec"] = elapsed

        for name in ["EP", "PAS", "RT", "RI", "RE"]:
            value = float(result[name])

            if not np.isfinite(value):
                raise RuntimeError(
                    f"Invalid {name} for {strategy}: {value}"
                )

        rows.append(result)

        print(f"EP      : {result['EP']:.6f}")
        print(f"PAS     : {result['PAS']:.6f}")
        print(f"RT      : {result['RT']:.6f}")
        print(f"RI      : {result['RI']:.6f}")
        print(f"RE      : {result['RE']:.6f}")

        print(
            "Policy  :",
            result["selected_policy_mode"]
        )

        print(
            "Guard   :",
            result["safe_guard_mode"]
        )

        print(f"Runtime : {elapsed:.3f} s", flush=True)

    df = pd.DataFrame(rows)

    output = output_dir / "00_smoke_original.csv"

    if output.exists():
        raise FileExistsError(
            f"{output} already exists; refusing to overwrite."
        )

    df.to_csv(output, index=False)

    metadata = {
        "purpose": "Original implementation smoke test",
        "status": "smoke_test_not_statistical_evidence",
        "configuration": config,
        "python": platform.python_version(),
        "strategies": ["CDP", "DRAD-CF-MPC"],
    }

    meta_path = output_dir / "00_smoke_original.json"

    if meta_path.exists():
        raise FileExistsError(
            f"{meta_path} already exists; refusing to overwrite."
        )

    meta_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        df[
            [
                "strategy",
                "EP",
                "PAS",
                "RT",
                "RI",
                "RE",
                "selected_policy_mode",
                "safe_guard_mode",
                "wall_time_sec"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved:", output)
    print("Metadata saved:", meta_path)
    print("EXPERIMENT 00 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
