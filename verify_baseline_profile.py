"""Bounded execution check of DRAD profile propagation; does not change released results."""
from pathlib import Path
import importlib.util
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import dynamic_risk_defense
import drad_mpc
from experiment_profiles import DEFAULT_RISK_PROFILE, get_risk_weights


def load_driver():
    spec = importlib.util.spec_from_file_location("baseline_parallel", ROOT / "src/05_run_experiments_parallel.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def main():
    import os
    os.chdir(ROOT)
    driver = load_driver()
    assert DEFAULT_RISK_PROFILE == "distributed-feature"
    for profile in ("distributed-feature", "degree-prioritized"):
        expected = get_risk_weights(profile)
        with patch.object(dynamic_risk_defense, "compute_dynamic_risk_score", wraps=dynamic_risk_defense.compute_dynamic_risk_score) as direct_score:
            row = driver.run_one_task(("RC", "DRAD", 0.5, 20, 120, 1, 1, profile))
            assert direct_score.call_args.kwargs["weights"] == expected
            assert row["risk_profile"] == profile
            assert all(row["drad_weight_" + key] == value for key, value in expected.items())
        with patch.object(drad_mpc, "compute_dynamic_risk_score", wraps=drad_mpc.compute_dynamic_risk_score) as candidate_score:
            row = driver.run_one_task(("RC", "DRAD-CF-MPC", 0.5, 20, 120, 1, 1, profile))
            # First call scores DRAD_SCORE; the separate rank-ensemble follows.
            assert candidate_score.call_args_list[0].kwargs["weights"] == expected
            assert row["risk_profile"] == profile
            assert all(row["drad_weight_" + key] == value for key, value in expected.items())
        print(f"PASS: {profile} reaches standalone DRAD and DRAD_SCORE through the parallel task runner")
    with patch.object(drad_mpc, "compute_dynamic_risk_score", wraps=drad_mpc.compute_dynamic_risk_score) as score:
        drad_mpc.candidate_drad_score_nodes("RC", 20)
        assert score.call_args.kwargs["weights"] == get_risk_weights("degree-prioritized")
    print("PASS: unconfigured low-level candidate retains its prior default")
    try:
        get_risk_weights("invalid-profile")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown profiles must fail")
    print("Bounded profile checks passed. Released result CSV files were not written.")


if __name__ == "__main__":
    main()
