"""E4: internal memory vs. number of disturbance bands (refined theorem).

Disturbance has M=2 sinusoidal bands (slow + mid) plus per-agent white noise.
Controllers differ only in which bands their internal model covers:
    P        : d=0, covers nothing
    RES-slow : d=2, resonator at the slow band only
    RES-mid  : d=2, resonator at the mid band only
    RES-both : d=4, resonators at both bands

Prediction of the refined theorem ("width cannot buy depth"): the residual
power in an UNCOVERED band is independent of N — no swarm size compensates
for a missing internal model — while power in covered bands and in the white
band keeps falling.
"""
import sys, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cdl.sim import simulate, NoiseCfg, CtrlCfg
from cdl.metrics import band_power

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
e4 = cfg["e4_memory_vs_bands"]
task = cfg["task"]

f1, f2 = e4["drift_freqs"]  # rad/step
noise = NoiseCfg(freqs=(f1, f2), amps=tuple(e4["drift_amps"]),
                 sigma_process=task["sigma_process"], sigma_meas=e4["sigma_meas"])

def band_around(w, rel=0.5):  # narrow band around a sinusoid, cycles/step
    c = w / (2 * np.pi)
    return (c * (1 - rel), c * (1 + rel))

BANDS = {"slow": band_around(f1), "mid": band_around(f2), "white": (0.05, 0.5)}

CONTROLLERS = {
    "P (d=0, covers none)":        CtrlCfg(kind="P",   kp=0.18),
    "RES-slow (d=2, covers slow)": CtrlCfg(kind="RES", kp=0.18, res_freqs=(f1,), res_gain=0.002),
    "RES-mid (d=2, covers mid)":   CtrlCfg(kind="RES", kp=0.18, res_freqs=(f2,), res_gain=0.002),
    "RES-both (d=4, covers both)": CtrlCfg(kind="RES", kp=0.18, res_freqs=(f1, f2), res_gain=0.002),
}

rows = []
t0 = time.time()
for name, ctrl in CONTROLLERS.items():
    for N in e4["N_grid"]:
        r = simulate(N=N, T=e4["T"], seeds=e4["seeds"], delay=task["delay"],
                     noise=noise, ctrl=ctrl, rng_seed=11, return_traj=True)
        bp, _ = band_power(r["traj"], bands=list(BANDS.values()))
        for bname, key in zip(BANDS.keys(), BANDS.values()):
            for s, p in enumerate(bp[key]):
                rows.append({"controller": name, "N": N, "band": bname,
                             "seed": s, "power": p})
        print(f"{name:30s} N={N:5d} MSE={r['mse'].mean():.4f} [{time.time()-t0:.0f}s]")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/data/e4_bands.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(CONTROLLERS)))
for ax, bname in zip(axes, BANDS):
    for c, (name, _) in zip(colors, CONTROLLERS.items()):
        sub = df[(df.controller == name) & (df.band == bname)].groupby("N")["power"]
        ax.errorbar(sub.mean().index, sub.mean().values,
                    yerr=1.96 * sub.sem().values, fmt="o-", color=c,
                    label=name, ms=4, lw=1.3, capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{bname} band"); ax.set_xlabel("$N$"); ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("Residual power in band")
axes[1].legend(fontsize=7.5)
fig.suptitle("E4: residual power per spectral band — uncovered bands do not improve with $N$", y=1.02)
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig4_memory_vs_bands.png", dpi=180, bbox_inches="tight")
print("done", round(time.time() - t0), "s")
