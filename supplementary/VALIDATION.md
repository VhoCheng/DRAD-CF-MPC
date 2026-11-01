# Release verification

This source package was checked on 2026-10-27. Imported files match the supplied archive manifests, and every Python file parses. The two principal source protocols match all seven frozen source hashes checked for the policy-library rolling and local-information studies.

The restored local-information CSV contains 57,600 method outcomes, 9,600 matched contexts and 15 graph instances. Recalculation gives CF_LOCAL mean PAS 0.257409383710835 and baseline-relative deterioration fraction 0.13552083333333334; direct LOCAL_RANK gives 0.2535060219506552 and 0.2275. These are checks of supplied results, not new simulator outcomes.

The actual CLI/import checks for the rolling and local-information drivers and the risk-analysis utility passed under Python 3.13.5 in the available scientific environment. Full Monte Carlo experiments, ACOPF studies and end-to-end timing were not rerun for this release. A follow-up check installed PYPOWER 5.1.21 in an isolated temporary environment and successfully invoked the physical-sensitivity driver with `--help`. Install `requirements-physical.txt` before running electrical code. These import/CLI checks do not constitute a rerun of ACOPF experiments.

The supplied environment snapshot is incomplete as a historical lockfile. This package preserves actual source/configuration/data provenance and distinguishes syntax/import checks from complete result reproduction.
