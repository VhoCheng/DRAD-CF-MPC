# Final physical-science claim boundaries

- IEEE 14: 20/20 active branches with usable PGLib-v23.07 RATE_A; coverage 100.00%.
- IEEE 57: 80/80 active branches with usable PGLib-v23.07 RATE_A; coverage 100.00%.
- IEEE 118: 186/186 active branches with usable PGLib-v23.07 RATE_A; coverage 100.00%.
- IEEE 300: 403/411 active branches with usable PGLib-v23.07 RATE_A; coverage 98.05%.

**IEEE 300 full branch-thermal-limit verification remains blocked if coverage < 100%.**
Do not replace zeros or 9900 with fabricated utility thermal limits.
AC OPF solver success does not establish closed-loop CPPS safety; failure does not prove infeasibility.
PGLib thermal ratings are reference benchmark data, not site-measured limits.
Exp58 physical load interruption is an assumed cyber-to-power sensitivity mapping, not field-observed curtailment.

## Original pandapower IEEE118 vs PGLib IEEE118 comparison
```json
{
  "availability": "FAILED_OR_UNAVAILABLE",
  "exception": "ModuleNotFoundError(\"No module named 'pandapower'\")",
  "action": "Do not assert exact physical-model equivalence."
}
```

## All Exp57 statuses (not just feasible ones)
```
 expected_case            kind                          classified  n
            14 N1_high_loading        SOLVER_FAILURE_OR_INFEASIBLE  1
            14 N1_high_loading                     SUCCESS_CHECKED  7
            14       N1_random                     SUCCESS_CHECKED  4
            14            load                     SUCCESS_CHECKED  6
            57 N1_high_loading        SOLVER_FAILURE_OR_INFEASIBLE  2
            57 N1_high_loading                     SUCCESS_CHECKED  6
            57       N1_random                     SUCCESS_CHECKED  4
            57            load        SOLVER_FAILURE_OR_INFEASIBLE  1
            57            load                     SUCCESS_CHECKED  5
           118 N1_high_loading                     SUCCESS_CHECKED  8
           118       N1_random                     SUCCESS_CHECKED  4
           118            load                     SUCCESS_CHECKED  6
           300 N1_high_loading                              ISLAND  2
           300 N1_high_loading        SOLVER_FAILURE_OR_INFEASIBLE  4
           300 N1_high_loading SUCCESS_WITH_UNVERIFIED_CONSTRAINTS  2
           300       N1_random                              ISLAND  3
           300       N1_random SUCCESS_WITH_UNVERIFIED_CONSTRAINTS  1
           300            load        SOLVER_FAILURE_OR_INFEASIBLE  3
           300            load SUCCESS_WITH_UNVERIFIED_CONSTRAINTS  3
```