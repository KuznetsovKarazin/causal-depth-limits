"""E5: role of latency — the true hard constraint.

Sweep the observation delay tau for the tuned flat PI swarm at fixed N.
(a) MSE grows with delay: latency, not agent count, bounds achievable control.
(b) Bode waterbed: aggressive low-frequency suppression pumps error power
    into mid frequencies; visible as a hump in the residual PSD that grows
    with delay.
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
gains = json.load(open(ROOT / "configs/tuned_gains.json"))
task, e5 = cfg["task"], cfg["e5_delay"]

noise = NoiseCfg(freqs=tuple(task["drift_freqs"]), amps=tuple(task["drift_amps"]),
                 sigma_process=task["sigma_process"], sigma_meas=task["sigma_meas"])
ctrl = CtrlCfg(kind="PI", **gains["PI"]["gains"])

rows, psds = [], {}
t0 = time.time()
for tau in e5["delays"]:
    r = simulate(N=e5["N"], T=e5["T"], seeds=e5["seeds"], delay=tau,
                 noise=noise, ctrl=ctrl, rng_seed=13, return_traj=True)
    if r["unstable"]:
        rows.append({"delay": tau, "seed": -1, "mse": np.nan, "unstable": True})
        print(f"tau={tau:3d}  UNSTABLE (exceeds delay-limited stability margin)")
        continue
    _, (f, pxx) = band_power(r["traj"], bands=[(1e-4, 0.5)])
    psds[tau] = (f, pxx)
    for s, m in enumerate(r["mse"]):
        rows.append({"delay": tau, "seed": s, "mse": m, "unstable": False})
    print(f"tau={tau:3d}  MSE={r['mse'].mean():.4f} [{time.time()-t0:.0f}s]")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results/data/e5_delay.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
colors = plt.cm.plasma(np.linspace(0, 0.85, len(e5["delays"])))
for c, tau in zip(colors, e5["delays"]):
    if tau not in psds:
        continue
    f, pxx = psds[tau]
    axes[0].loglog(f[1:], pxx[1:], color=c, lw=1.2, label=f"$\\tau={tau}$")
axes[0].set_xlabel("Frequency (cycles/step)")
axes[0].set_ylabel("PSD of residual error")
axes[0].set_title("Waterbed effect: delay pumps error\ninto mid frequencies")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, which="both")

stable = df[df.unstable == False]
g = stable.groupby("delay")["mse"]
axes[1].errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.sem().values,
                 fmt="o-", color="tab:blue", capsize=3)
axes[1].set_xlabel("Observation delay $\\tau$ (steps)")
axes[1].set_ylabel("MSE"); axes[1].set_title(f"MSE vs. delay (flat PI, $N={e5['N']}$)")
axes[1].grid(alpha=0.3)
unst = sorted(df[df.unstable == True].delay.unique())
if len(unst):
    axes[1].axvline(min(unst), color="tab:red", ls="--", lw=1)
    axes[1].text(min(unst) * 0.98, axes[1].get_ylim()[1] * 0.6,
                 f"unstable\n($\\tau\\geq{min(unst)}$)", color="tab:red",
                 ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig5_delay_waterbed.png", dpi=180)
print("done", round(time.time() - t0), "s")
