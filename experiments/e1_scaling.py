"""E1-E3: scaling MSE(N) for flat vs deep controllers on the base task.

E1: flat memoryless P swarm -> causal floor C > 0 (confirms the paper's Fig.1).
E2: hierarchical CASCADE     -> floor greatly reduced (confirms hierarchy helps).
E3: flat PI / flat RES swarm -> floor reduced/eliminated WITHOUT any hierarchy
    (refutes the theorem as stated: the resource is per-agent internal memory,
    not architectural hierarchy).

Outputs: results/data/e1_scaling.csv, results/data/e1_fits.json,
         figures fig1 (scaling), fig2 (PSD), fig3 (time series).
"""
import sys, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cdl.sim import simulate, NoiseCfg, CtrlCfg, internal_state_dim
from cdl.metrics import fit_floor, band_power

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
gains = json.load(open(ROOT / "configs/tuned_gains.json"))
task, sc = cfg["task"], cfg["scaling"]

noise = NoiseCfg(freqs=tuple(task["drift_freqs"]), amps=tuple(task["drift_amps"]),
                 sigma_process=task["sigma_process"], sigma_meas=task["sigma_meas"])

LABELS = {"P": "Flat P (memoryless, d=0)",
          "PI": "Flat PI (d=1)",
          "PID": "Flat PID (d=2)",
          "CASCADE": "Hierarchical cascade (deep, d=2)",
          "RES": "Flat + internal model (d=2)"}
COLORS = {"P": "tab:red", "PI": "tab:blue", "PID": "tab:cyan",
          "CASCADE": "tab:green", "RES": "tab:purple"}

def make_ctrl(kind):
    kw = dict(gains[kind]["gains"])
    if kind == "RES":
        kw["res_freqs"] = tuple(task["drift_freqs"])
    return CtrlCfg(kind=kind, **kw)

rows, fits, trajs = [], {}, {}
t0 = time.time()
for kind in sc["controllers"]:
    ctrl = make_ctrl(kind)
    mse_matrix = []
    for N in sc["N_grid"]:
        want_traj = (N == 1000)
        r = simulate(N=N, T=sc["T"], seeds=sc["seeds"], delay=task["delay"],
                     noise=noise, ctrl=ctrl, burn_in=sc["burn_in"],
                     rng_seed=7, return_traj=want_traj)
        mse_matrix.append(r["mse"])
        if want_traj:
            trajs[kind] = r["traj"]
        for s, m in enumerate(r["mse"]):
            rows.append({"controller": kind, "N": N, "seed": s, "mse": m,
                         "state_dim": r["state_dim"]})
        print(f"{kind:8s} N={N:5d} MSE={r['mse'].mean():.4f}  [{time.time()-t0:.0f}s]")
    mse_matrix = np.array(mse_matrix)
    fits[kind] = fit_floor(sc["N_grid"], mse_matrix)
    f = fits[kind]
    print(f"  fit: MSE = {f['A']:.2f}/N^{f['alpha']:.2f} + {f['C']:.4f}  "
          f"C 95% CI [{f['C_ci'][0]:.4f}, {f['C_ci'][1]:.4f}]")

df = pd.DataFrame(rows)
(ROOT / "results/data").mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / "results/data/e1_scaling.csv", index=False)
np.savez_compressed(ROOT / "results/data/e1_trajs.npz", **{k: v[:2] for k, v in trajs.items()})
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
               for kk, vv in v.items()} for k, v in fits.items()},
          open(ROOT / "results/data/e1_fits.json", "w"), indent=2)

# ---------------- Figure 1: scaling curves with floor fits ----------------
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555", "axes.linewidth": 0.9,
    "font.size": 10.5, "axes.titlesize": 11.5, "legend.frameon": False,
})
# vivid, colour-blind-friendly palette keyed by controller kind
_PAL = {"P": "#9c9a92", "PI": "#378ADD", "PID": "#185FA5",
        "CASCADE": "#D85A30", "RES": "#1D9E75", "PIRES": "#1D9E75"}
fig, ax = plt.subplots(figsize=(7.5, 5.2))
Ngrid = np.array(sc["N_grid"], float)
Nfine = np.logspace(0, np.log10(Ngrid[-1]), 200)
for kind in sc["controllers"]:
    col = _PAL.get(kind, COLORS[kind])
    sub = df[df.controller == kind].groupby("N")["mse"]
    mean, sem = sub.mean(), sub.sem()
    f = fits[kind]
    if f["C_ci"][0] > 1e-6:
        ax.axhline(f["C"], color=col, ls=":", alpha=0.30, lw=1)
    ax.plot(Nfine, f["A"] / Nfine ** f["alpha"] + f["C"], "-",
            color=col, alpha=0.25, lw=6, solid_capstyle="round", zorder=1)
    ax.errorbar(mean.index, mean.values, yerr=1.96 * sem.values, fmt="o-",
                color=col, label=LABELS[kind], ms=6, lw=2.0, capsize=2,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of agents  $N$"); ax.set_ylabel("MSE")
ax.set_title("Scaling of flat vs. deep controllers: the floor is set by\n"
             "internal-model content, not by hierarchy", fontsize=11.5)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.18, which="both", lw=0.7)
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig1_scaling.png", dpi=190)

# ---------------- Figure 2: PSD decomposition at N=1000 ----------------
fig, ax = plt.subplots(figsize=(7.5, 5.0))
drift_f = task["drift_freqs"][0] / (2 * np.pi)  # cycles/step
for kind in sc["controllers"]:
    _, (f, pxx) = band_power(trajs[kind], bands=[(1e-4, 0.5)])
    ax.loglog(f[1:], pxx[1:], color=COLORS[kind], label=LABELS[kind], lw=1.2)
ax.axvline(drift_f, color="k", ls="--", alpha=0.5, lw=1)
ax.text(drift_f * 1.1, ax.get_ylim()[0] * 3, "drift band", fontsize=8)
ax.set_xlabel("Frequency (cycles/step)"); ax.set_ylabel("PSD of residual error")
ax.set_title("Residual error spectrum at $N=1000$: the causal floor is an\n"
             "incompressible low-frequency component")
ax.legend(fontsize=8.5); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig2_psd.png", dpi=180)

# ---------------- Figure 3: time series ----------------
fig, ax = plt.subplots(figsize=(8.5, 4.2))
tt = np.arange(4000, 8000)
for kind in ["P", "PI", "CASCADE", "RES"]:
    ax.plot(tt, trajs[kind][0, tt[0]:tt[-1] + 1], color=COLORS[kind],
            label=LABELS[kind], lw=0.9, alpha=0.85)
ax.set_xlabel("time step"); ax.set_ylabel("error $x(t)$")
ax.set_title("Residual error traces, $N=1000$ (single seed)")
ax.legend(fontsize=8.5, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig3_timeseries.png", dpi=180)
print("done", round(time.time() - t0), "s")
