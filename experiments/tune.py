"""Fair-comparison gain tuning.

Each controller class gets a grid search over its gains, minimizing MSE on the
base task at N_tune agents (training seeds are disjoint from evaluation seeds
via a different rng_seed offset). Tuned gains are frozen and reused for every
N in the scaling experiments, so no controller is strawmanned.
"""
import sys, itertools, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
import numpy as np
from cdl.sim import simulate, NoiseCfg, CtrlCfg

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
task, tun = cfg["task"], cfg["tuning"]

noise = NoiseCfg(freqs=tuple(task["drift_freqs"]), amps=tuple(task["drift_amps"]),
                 sigma_process=task["sigma_process"], sigma_meas=task["sigma_meas"])

RES_FREQS = tuple(task["drift_freqs"])  # internal model matched to drift band

best = {}
for kind, grid in tun["grids"].items():
    keys, vals = list(grid.keys()), list(grid.values())
    best_mse, best_kw = np.inf, None
    for combo in itertools.product(*vals):
        kw = dict(zip(keys, combo))
        if kind == "RES":
            kw["res_freqs"] = RES_FREQS
        r = simulate(N=tun["N_tune"], T=tun["T"], seeds=tun["seeds"],
                     delay=task["delay"], noise=noise,
                     ctrl=CtrlCfg(kind=kind, **kw), rng_seed=999)
        m = float(r["mse"].mean()) if not r["unstable"] else np.inf
        if m < best_mse:
            best_mse, best_kw = m, kw
    best[kind] = {"gains": {k: v for k, v in best_kw.items() if k != "res_freqs"},
                  "tune_mse": best_mse}
    print(f"{kind:8s} -> {best[kind]['gains']}  (tune MSE {best_mse:.3f})")

out = ROOT / "configs/tuned_gains.json"
json.dump(best, open(out, "w"), indent=2)
print("saved", out)
