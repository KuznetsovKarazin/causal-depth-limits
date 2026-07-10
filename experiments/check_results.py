"""Validate saved experiment artifacts against the paper's headline claims.

Unlike the fast unit tests (which use a loose 10% tolerance so they run in
seconds), this script checks the *saved* results that actually back the numbers
in the paper. It exits non-zero if any claim regresses, so CI catches silent
drift between the text and the data.

    python experiments/check_results.py
"""
import sys
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "results/data"
failures = []


def check(name, condition, detail):
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not condition:
        failures.append(name)


# --- Claim: optimal controller matches the Riccati floor to <1% (Block C) ---
p = DATA / "c_delay.csv"
if p.exists():
    d = pd.read_csv(p)
    worst = d["rel_pct"].max()
    check("kalman_riccati_<1pct", worst < 1.0,
          f"max |kalman-riccati|/riccati = {worst:.2f}% (claim: < 1%)")
    # PI must be flagged unstable somewhere at large tau, not silently 0
    if "pi_unstable" in d.columns:
        check("pi_instability_encoded", d["pi_unstable"].any(),
              f"{int(d['pi_unstable'].sum())} PI-unstable rows encoded as NaN")
else:
    check("c_delay.csv present", False, "missing — run `make phase2`")

# --- Claim: PIRES memory ladder is monotone at large N (Block A map) ---
floors = {}
for dlvl in [0, 1, 3, 5, 7]:
    f = DATA / f"a_map_d{dlvl}.csv"
    if f.exists():
        m = pd.read_csv(f)
        floors[dlvl] = m[m.N == m.N.max()]["mse"].mean()
if len(floors) == 5:
    seq = [floors[k] for k in [0, 1, 3, 5, 7]]
    mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    check("pires_ladder_monotone", mono,
          "floors " + " > ".join(f"{v:.2f}" for v in seq))
else:
    check("map floors present", False, "missing a_map_d*.csv — run `make phase2`")

# --- Claim: equal-budget points are exact (N*d == B) ---
b = DATA / "a_equal_budget.csv"
if b.exists():
    eb = pd.read_csv(b)
    exact = (eb["Nd"] == eb["budget"]).all()
    check("equal_budget_exact", exact, "every N*d equals its budget B")

if failures:
    print(f"\n{len(failures)} check(s) FAILED: {', '.join(failures)}")
    sys.exit(1)
print("\nAll result checks passed.")
