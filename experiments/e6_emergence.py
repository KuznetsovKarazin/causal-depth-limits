"""E6: does temporal hierarchy EMERGE in flat learned swarms?

Each of N agents is a learnable leaky-integrator controller:
    z_i[t+1] = rho_i * z_i[t] + y_i[t]
    u_i      = -(a_i^2) * y_i - (b_i^2) * z_i
with per-agent parameters theta_i = (a_i, b_i, c_i), rho_i = sigmoid(c_i).
The collective control is the mean over agents, so a HETEROGENEOUS population
implements a higher-order filter (a sum of first-order sections) even though
no hierarchy is designed in.

Conditions (trained with simple antithetic evolution strategies):
  het-M2 : heterogeneous params, disturbance with M=2 bands
  hom-M2 : parameters SHARED across agents (flat in the strictest sense), M=2
  het-M1 : heterogeneous params, M=1 band (control condition)

Exploratory question: does heterogeneity create virtual temporal depth?
Observed: (i) het-M2 beats hom-M2 and is more stable; (ii) learned time constants
1/(1-rho_i) in het-M2 spread across separated scales. A slow specialist also
emerges in het-M1 (single band), so the robust effect is heterogeneity-vs-
homogeneity, not a sharp M2-only claim. Emergent differentiation = proto-
temporal-depth without design (exploratory).
"""
import sys, json, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))["e6_emergence"]
N = cfg["N_agents"]
DELAY = cfg["delay"]
ES = cfg["es"]


def rollout(theta, freqs, amps, sigma_meas, T, burn_in, rng, shared=False):
    """theta: (P, N, 3) or (P, 1, 3) if shared. Returns fitness (P,) = -MSE."""
    P = theta.shape[0]
    a2 = theta[..., 0] ** 2                      # (P, N) or (P, 1)
    b2 = theta[..., 1] ** 2
    rho = 1.0 / (1.0 + np.exp(-theta[..., 2]))
    if shared:
        a2 = np.broadcast_to(a2, (P, N)); b2 = np.broadcast_to(b2, (P, N))
        rho = np.broadcast_to(rho, (P, N))
    x = np.zeros(P)
    hist = np.zeros((DELAY + 1, P))
    z = np.zeros((P, N))
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    sq, n = np.zeros(P), 0
    clipped = np.zeros(P)
    for t in range(T):
        y = hist[t % (DELAY + 1)][:, None] + sigma_meas * rng.standard_normal((P, N))
        z = rho * z + y
        u = (-(a2 * y) - (b2 * z)).mean(axis=1)
        d = sum(A * np.sin(w * t + ph) for A, w, ph in zip(amps, freqs, phases))
        x = x + u + d + 0.05 * rng.standard_normal(P)
        clipped += (np.abs(x) >= 99.9)
        x = np.clip(x, -100.0, 100.0)   # soft saturation keeps ES gradient informative
        hist[t % (DELAY + 1)] = x
        if t >= burn_in:
            sq += x * x; n += 1
    return -(sq / n + 1e3 * clipped / T)   # explicit instability penalty


def train(freqs, amps, shared, seed):
    rng = np.random.default_rng(seed)
    n_units = 1 if shared else N
    theta = np.tile(np.array([0.3, 0.05, 0.0]), (n_units, 1))       # homogeneous init
    theta += 0.01 * rng.standard_normal(theta.shape)
    hist = []
    P = ES["popsize"]
    for g in range(ES["generations"]):
        eps = rng.standard_normal((P // 2, n_units, 3))
        eps = np.concatenate([eps, -eps])                            # antithetic
        cand = theta[None] + ES["sigma"] * eps
        fit = rollout(cand, freqs, amps, cfg["sigma_meas"], ES["T"],
                      ES["burn_in"], np.random.default_rng(10_000 + g), shared)
        adv = (fit - fit.mean()) / (fit.std() + 1e-8)
        theta = theta + ES["lr"] / (P * ES["sigma"]) * np.einsum("p,pij->ij", adv, eps)
        hist.append(-fit.max())
    return theta, hist


def evaluate(theta, freqs, amps, shared, n_eval=6):
    fits = [rollout(theta[None], freqs, amps, cfg["sigma_meas"], 8000, 1000,
                    np.random.default_rng(50_000 + k), shared)[0] for k in range(n_eval)]
    return -np.mean(fits), np.std(fits) / np.sqrt(n_eval)


import argparse
t0 = time.time()
M2 = (tuple(cfg["drift_freqs"]), tuple(cfg["drift_amps"]))
M1 = ((cfg["drift_freqs"][0],), (sum(cfg["drift_amps"]),))
CONDS = {"het-M2": (M2, False), "hom-M2": (M2, True), "het-M1": (M1, False)}
PART = ROOT / "results/data/e6_parts"
PART.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--cond", choices=list(CONDS) + ["assemble"], default=None)
ap.add_argument("--seed", type=int, default=0)
args, _ = ap.parse_known_args()

if args.cond and args.cond != "assemble":
    (freqs, amps), shared = CONDS[args.cond]
    th, hist = train(freqs, amps, shared, args.seed)
    mse, sem = evaluate(th, freqs, amps, shared)
    json.dump({"theta": th.tolist(), "curve": hist, "eval_mse": mse},
              open(PART / f"{args.cond}_s{args.seed}.json", "w"))
    print(f"{args.cond} seed{args.seed}: eval MSE={mse:.4f} [{time.time()-t0:.0f}s]")
    sys.exit(0)

# assemble mode (default): gather parts, write combined results, plot
results = {}
for name in CONDS:
    thetas, curves, mses = [], [], []
    for seed in range(5):
        f = PART / f"{name}_s{seed}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        thetas.append(np.array(d["theta"])); curves.append(d["curve"]); mses.append(d["eval_mse"])
    if len(mses) < 3:
        raise SystemExit(f"condition '{name}' has only {len(mses)} seed(s); "
                         f"run e.g.: python experiments/e6_emergence.py --cond {name} --seed 0 "
                         f"(need >=3, ideally 5)")
    if len(mses) < 5:
        print(f"[warn] {name}: assembling from {len(mses)} seeds (<5); "
              f"medians are less stable.")
    results[name] = {"thetas": [t.tolist() for t in thetas],
                     "curves": curves, "eval_mse": mses}
json.dump(results, open(ROOT / "results/data/e6_emergence.json", "w"))

# ---------------- figure ----------------
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555", "axes.linewidth": 0.9,
    "font.size": 10.5, "legend.frameon": False,
})
_C = {"het-M2": "#185FA5", "hom-M2": "#EF9F27", "het-M1": "#1D9E75"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for name, c in [("het-M2", _C["het-M2"]), ("hom-M2", _C["hom-M2"]), ("het-M1", _C["het-M1"])]:
    for cv in results[name]["curves"]:
        axes[0].plot(cv, color=c, alpha=0.45, lw=1.3)
    axes[0].plot([], [], color=c, lw=2.5, label=f"{name} (median MSE {np.median(results[name]['eval_mse']):.2f})")
axes[0].set_yscale("log"); axes[0].set_xlabel("ES generation"); axes[0].set_ylabel("best MSE")
axes[0].set_title("Training curves"); axes[0].legend(fontsize=8.5); axes[0].grid(alpha=0.18)

def timescales(name):
    ts = []
    for th in results[name]["thetas"]:
        th = np.array(th)
        rho = 1 / (1 + np.exp(-th[:, 2]))
        ts.append(1.0 / np.maximum(1 - rho, 1e-4))
    return np.concatenate(ts)

for i, (name, c) in enumerate([("het-M2", _C["het-M2"]), ("het-M1", _C["het-M1"])]):
    ts = timescales(name)
    axes[1].scatter(np.full_like(ts, i) + 0.08 * np.random.randn(len(ts)),
                    ts, color=c, alpha=0.75, s=34, edgecolor="white", linewidth=0.5)
axes[1].set_yscale("log"); axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["het-M2", "het-M1"])
for w, ls in zip(cfg["drift_freqs"], ["--", ":"]):
    axes[1].axhline(1.0 / w, color="#333", ls=ls, alpha=0.5, lw=1)
axes[1].set_ylabel("learned integrator timescale $1/(1-\\rho_i)$")
axes[1].set_title("Learned per-agent timescales\n(dashed lines: disturbance timescales $1/\\omega_j$)")
axes[1].grid(alpha=0.18)

x = np.arange(3)
keys = ["het-M2", "hom-M2", "het-M1"]
meds = [np.median(results[k]["eval_mse"]) for k in keys]
axes[2].bar(x, meds, width=0.55, color=[_C[k] for k in keys],
            edgecolor="white", linewidth=0.8, zorder=2)
for i, k in enumerate(keys):
    vals = np.array(results[k]["eval_mse"])
    stable = vals[vals < 1e3]
    axes[2].scatter(np.full(len(stable), i) + 0.06 * np.random.randn(len(stable)),
                    stable, color="#26215C", s=18, zorder=3, edgecolor="white", linewidth=0.4)
    n_unst = int((vals >= 1e3).sum())
    if n_unst:
        axes[2].text(i, meds[i] * 1.1, f"{n_unst}/5 runs\nunstable",
                     ha="center", fontsize=8, color="#D85A30", fontweight="bold")
axes[2].set_yscale("log")
axes[2].set_xticks(x); axes[2].set_xticklabels(keys)
axes[2].set_ylabel("eval MSE (median bar, dots = stable runs)")
axes[2].set_title("Heterogeneity creates virtual temporal depth\n(exploratory)")
axes[2].grid(alpha=0.18, axis="y", which="both")
fig.tight_layout()
fig.savefig(ROOT / "results/figures/fig6_emergence.png", dpi=190)
print("done", round(time.time() - t0), "s")
