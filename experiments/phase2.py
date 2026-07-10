"""Phase-2 experiments. Run chunked:
    python experiments/phase2.py map --d 0        (per memory level; then 1 3 5 7)
    python experiments/phase2.py map-adaptive
    python experiments/phase2.py robustness
    python experiments/phase2.py nonstationary
    python experiments/phase2.py delay
    python experiments/phase2.py nonlinear
    python experiments/phase2.py figures          (assemble all Phase-2 figures)
"""
import sys, json, pathlib, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cdl.sim import simulate, NoiseCfg, CtrlCfg
from cdl.central import run_kalman, run_adaptive_ar

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "results/data", ROOT / "results/figures"

# ---- Block A task: three separated bands ----
W3 = (0.003, 0.015, 0.06)
NOISE3 = NoiseCfg(freqs=W3, amps=(0.3, 0.3, 0.3), sigma_process=0.05, sigma_meas=20.0)
NGRID = [1, 3, 10, 30, 100, 300, 1000]
# PIRES ladder: the integrator (PI) covers bias / very-low-frequency drift, and
# each added resonator covers one narrowband component. This makes memory depth
# natural and monotone (d = 1 + 2*#resonators for the PIRES rungs).
LADDER = {0: CtrlCfg(kind="P", kp=0.18),
          1: CtrlCfg(kind="PI", kp=0.18, ki=0.004),
          3: CtrlCfg(kind="PIRES", kp=0.18, ki=0.004, res_freqs=W3[:1], res_gain=0.002),
          5: CtrlCfg(kind="PIRES", kp=0.18, ki=0.004, res_freqs=W3[:2], res_gain=0.002),
          7: CtrlCfg(kind="PIRES", kp=0.18, ki=0.004, res_freqs=W3[:3], res_gain=0.002)}
SEEDS, T = 8, 20000

# ---- Block B task: single band, oracle mismatch ----
W0 = 0.01
NOISE1 = NoiseCfg(freqs=(W0,), amps=(0.3,), sigma_process=0.05, sigma_meas=20.0)


def cmd_map(d):
    ctrl = LADDER[d]
    rows = []
    t0 = time.time()
    for N in NGRID:
        r = simulate(N=N, T=T, seeds=SEEDS, delay=5, noise=NOISE3, ctrl=ctrl, rng_seed=21)
        for s, m in enumerate(r["mse"]):
            rows.append({"d": d, "N": N, "seed": s, "mse": m})
        print(f"d={d} N={N:5d} MSE={r['mse'].mean():.4f} [{time.time()-t0:.0f}s]")
    f = DATA / f"a_map_d{d}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)


def cmd_map_adaptive():
    rows = []
    t0 = time.time()
    for N in [10, 30, 100, 300, 1000]:
        r = run_adaptive_ar(N=N, T=T, seeds=6, delay=5, noise=NOISE3,
                            n_tones=3, pll_gain=0.03, rng_seed=23)
        for s, m in enumerate(r["mse"]):
            rows.append({"d": "adaptive(7)", "N": N, "seed": s, "mse": m})
        print(f"adaptive N={N:5d} MSE={r['mse'].mean():.4f} [{time.time()-t0:.0f}s]")
    pd.DataFrame(rows).to_csv(DATA / "a_map_adaptive.csv", index=False)


def cmd_robustness():
    rows = []
    t0 = time.time()
    ratios = [0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3]
    for rho in [0.99, 0.999, 0.9999]:
        for q in ratios:
            ctrl = CtrlCfg(kind="RES", kp=0.12, res_freqs=(W0 * q,),
                           res_gain=0.002, res_damp=rho)
            r = simulate(N=300, T=16000, seeds=6, delay=5, noise=NOISE1,
                         ctrl=ctrl, rng_seed=31)
            for s, m in enumerate(r["mse"]):
                rows.append({"rho": rho, "ratio": q, "seed": s, "mse": m})
        print(f"rho={rho} done [{time.time()-t0:.0f}s]")
    # references: PI and adaptive at the same task
    rpi = simulate(N=300, T=16000, seeds=6, delay=5, noise=NOISE1,
                   ctrl=CtrlCfg(kind="PI", kp=0.08, ki=0.004), rng_seed=31)
    rad = run_adaptive_ar(N=300, T=16000, seeds=6, delay=5, noise=NOISE1, pll_gain=0.03)
    json.dump({"PI": float(rpi["mse"].mean()), "ADAPT": float(rad["mse"].mean())},
              open(DATA / "b_refs.json", "w"))
    pd.DataFrame(rows).to_csv(DATA / "b_mismatch.csv", index=False)


def cmd_nonstationary():
    rows = []
    cases = {
        "chirp": NoiseCfg(freqs=(0.008,), amps=(0.3,), sigma_process=0.05,
                          sigma_meas=20.0, chirp_rate=2e-7),
        "switching": NoiseCfg(freqs=(0.007,), amps=(0.3,), sigma_process=0.05,
                              sigma_meas=20.0, switch_period=5000,
                              switch_freqs=(0.007, 0.014)),
    }
    ctrls = {
        "oracle-RES (rho=.999)": ("sim", CtrlCfg(kind="RES", kp=0.12, res_freqs=(0.008,), res_gain=0.002, res_damp=0.999)),
        "oracle-RES detuned (rho=.99)": ("sim", CtrlCfg(kind="RES", kp=0.12, res_freqs=(0.008,), res_gain=0.002, res_damp=0.99)),
        "PI": ("sim", CtrlCfg(kind="PI", kp=0.08, ki=0.004)),
        "ADAPT (learned)": ("adapt", None),
    }
    for cname, nz in cases.items():
        for kname, (kind, ctrl) in ctrls.items():
            if kind == "sim":
                r = simulate(N=300, T=20000, seeds=6, delay=5, noise=nz, ctrl=ctrl, rng_seed=41)
            else:
                r = run_adaptive_ar(N=300, T=20000, seeds=6, delay=5, noise=nz, pll_gain=0.03, rng_seed=41)
            for s, m in enumerate(r["mse"]):
                rows.append({"case": cname, "controller": kname, "seed": s, "mse": m})
            print(cname, kname, round(float(np.mean(r["mse"])), 4))
    pd.DataFrame(rows).to_csv(DATA / "b_nonstationary.csv", index=False)


def cmd_delay():
    rows = []
    sig_e = 0.3
    # Larger T / seeds / burn-in so the optimal controller's finite-sample MSE
    # matches the semi-analytic Riccati floor to <1% even at large tau and a.
    for a in [0.0, 0.9, 0.99]:
        nz = NoiseCfg(freqs=(), amps=(), sigma_process=sig_e, sigma_meas=20.0, ar_coef=a)
        for tau in [1, 2, 5, 10, 15]:
            rk = run_kalman(N=100, T=60000, seeds=16, delay=tau, noise=nz,
                            burn_in=8000, rng_seed=51)
            rp = simulate(N=100, T=60000, seeds=16, delay=tau, noise=nz,
                          ctrl=CtrlCfg(kind="PI", kp=0.08, ki=0.004),
                          burn_in=8000, rng_seed=51)
            pi_unstable = bool(rp["unstable"])
            pi_mse = float("nan") if pi_unstable else float(rp["mse"].mean())
            rel = abs(rk["mse"].mean() - rk["floor_riccati"]) / rk["floor_riccati"] * 100
            rows.append({"a": a, "tau": tau, "kalman_mse": float(rk["mse"].mean()),
                         "kalman_sem": float(rk["mse"].std() / np.sqrt(16)),
                         "riccati": rk["floor_riccati"],
                         "rel_pct": rel,
                         "pi_mse": pi_mse, "pi_unstable": pi_unstable})
            print(f"a={a} tau={tau}: kalman={rk['mse'].mean():.3f} "
                  f"riccati={rk['floor_riccati']:.3f} rel={rel:.2f}% "
                  f"PI={'UNSTABLE' if pi_unstable else round(pi_mse,3)}")
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "c_delay.csv", index=False)
    print(f"max relative error: {df.rel_pct.max():.2f}%")


def cmd_equal_budget():
    """Strict fixed-total-state-budget comparison: for each budget B, allocate
    it as N*d with EXACT points (many shallow agents vs fewer deep ones)."""
    rows = []
    rung = {1: LADDER[1], 3: LADDER[3], 5: LADDER[5], 7: LADDER[7]}
    budgets = [1050, 3150, 6300, 12600]   # multiples of lcm(1,3,5,7)=105 -> exact N*d
    t0 = time.time()
    for B in budgets:
        for d, ctrl in rung.items():
            N = max(1, round(B / d))
            r = simulate(N=N, T=T, seeds=SEEDS, delay=5, noise=NOISE3, ctrl=ctrl, rng_seed=27)
            for s, m in enumerate(r["mse"]):
                rows.append({"budget": B, "d": d, "N": N, "Nd": N * d, "seed": s, "mse": m})
            print(f"B={B:6d} d={d} N={N:5d} (N*d={N*d}) MSE={r['mse'].mean():.4f} [{time.time()-t0:.0f}s]")
    pd.DataFrame(rows).to_csv(DATA / "a_equal_budget.csv", index=False)


def cmd_nonlinear():
    rows = []
    g = 0.15
    for d, ctrl in [(0, LADDER[0]), (1, LADDER[1]),
                    (2, CtrlCfg(kind="RES", kp=0.12, res_freqs=(W0,), res_gain=0.002))]:
        for N in [10, 300]:
            r = simulate(N=N, T=16000, seeds=6, delay=5, noise=NOISE1,
                         ctrl=ctrl, rng_seed=61, gamma_sin=g)
            for s, m in enumerate(r["mse"]):
                rows.append({"exp": "ladder", "d": d, "N": N, "controller": ctrl.kind,
                             "seed": s, "mse": m})
            print(f"nl ladder d={d} N={N}: {r['mse'].mean():.4f}")
    nz = NoiseCfg(freqs=(0.007,), amps=(0.3,), sigma_process=0.05, sigma_meas=20.0,
                  switch_period=5000, switch_freqs=(0.007, 0.014))
    for kname, kind, ctrl in [
            ("oracle-RES@0.007", "sim", CtrlCfg(kind="RES", kp=0.12, res_freqs=(0.007,), res_gain=0.002)),
            ("PI", "sim", CtrlCfg(kind="PI", kp=0.08, ki=0.004)),
            ("ADAPT", "adapt", None)]:
        if kind == "sim":
            r = simulate(N=300, T=20000, seeds=6, delay=5, noise=nz, ctrl=ctrl,
                         rng_seed=63, gamma_sin=g)
        else:
            r = run_adaptive_ar(N=300, T=20000, seeds=6, delay=5, noise=nz,
                                pll_gain=0.03, rng_seed=63, gamma_sin=g)
        for s, m in enumerate(r["mse"]):
            rows.append({"exp": "switch", "d": -1, "N": 300, "controller": kname,
                         "seed": s, "mse": m})
        print(f"nl switch {kname}: {np.mean(r['mse']):.4f}")
    pd.DataFrame(rows).to_csv(DATA / "e_nonlinear.csv", index=False)


def cmd_figures():
    # ---- MODERN STYLE ----
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "medium",
        "legend.frameon": False,
    })
    PAL = {"oracle": "#185FA5", "oracle2": "#85B7EB", "pi": "#9c9a92",
           "adapt": "#D85A30", "res": "#1D9E75"}
    # ---- figA1: MSE(N, d) map ----
    dfs = [pd.read_csv(DATA / f"a_map_d{d}.csv") for d in LADDER]
    df = pd.concat(dfs)
    ad = pd.read_csv(DATA / "a_map_adaptive.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(LADDER)))
    for c, d in zip(colors, LADDER):
        g = df[df.d == d].groupby("N")["mse"]
        _bands = {0:"no model",1:"integrator",3:"PI+1 res",5:"PI+2 res",7:"PI+3 res"}[d]
        axes[0].errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.sem().values,
                         fmt="o-", color=c, label=f"$d={d}$ ({_bands})",
                         ms=4, lw=1.4, capsize=2)
    ga = ad.groupby("N")["mse"]
    axes[0].errorbar(ga.mean().index, ga.mean().values, yerr=1.96 * ga.sem().values,
                     fmt="s--", color="tab:red", label="adaptive (no-oracle; extra\nestimator memory, off-axis)", ms=5, lw=1.4)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("$N$"); axes[0].set_ylabel("MSE")
    axes[0].set_title("Width x memory map: each memory level has its own floor")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, which="both")

    # heatmap
    piv = df.groupby(["d", "N"])["mse"].mean().unstack()
    im = axes[1].imshow(np.log10(piv.values), aspect="auto", cmap="viridis_r", origin="lower")
    axes[1].set_xticks(range(len(piv.columns))); axes[1].set_xticklabels(piv.columns)
    axes[1].set_yticks(range(len(piv.index))); axes[1].set_yticklabels(piv.index)
    axes[1].set_xlabel("$N$"); axes[1].set_ylabel("per-agent memory $d$")
    axes[1].set_title("$\\log_{10}$ MSE: L-shaped contours = no exchange")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[1].text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color="w" if np.log10(piv.values[i, j]) > 0.3 else "k")
    fig.colorbar(im, ax=axes[1], label="$\\log_{10}$ MSE (color)")
    fig.tight_layout(); fig.savefig(FIGS / "fig8_map.png", dpi=180)

    # ---- figA2: honest equal-budget Pareto (exact N*d points) ----
    dfb = pd.read_csv(DATA / "a_equal_budget.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    budgets = sorted(dfb.budget.unique())
    colors_b = plt.cm.plasma(np.linspace(0.1, 0.85, len(budgets)))
    for c, B in zip(colors_b, budgets):
        g = dfb[dfb.budget == B].groupby("d")["mse"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.sem().values,
                    fmt="o-", color=c, capsize=3,
                    label=f"budget $N\\!\\cdot\\!d={B}$")
        for d in g.mean().index:
            N = int(dfb[(dfb.budget == B) & (dfb.d == d)].N.iloc[0])
            ax.annotate(f"N={N}", (d, g.mean()[d]), fontsize=6.5,
                        xytext=(3, 4), textcoords="offset points")
    ax.set_yscale("log"); ax.set_xlabel("per-agent internal-state dimension $d$")
    ax.set_ylabel("MSE"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, title="strict $N\\cdot d$ = const")
    ax.set_title("Honest fixed total-state budget: past a minimum SNR,\n"
                 "spending states on memory beats spending them on width")
    fig.tight_layout(); fig.savefig(FIGS / "fig9_budget.png", dpi=180)

    # ---- figB1: mismatch x damping ----
    dfm = pd.read_csv(DATA / "b_mismatch.csv")
    refs = json.load(open(DATA / "b_refs.json"))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for rho, c in zip([0.99, 0.999, 0.9999], ["tab:green", "tab:blue", "tab:purple"]):
        g = dfm[dfm.rho == rho].groupby("ratio")["mse"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.sem().values,
                    fmt="o-", color=c, label=f"resonator damping $\\rho={rho}$", capsize=2)
    ax.axhline(refs["PI"], color="gray", ls="--", lw=1, label="PI (no model)")
    ax.axhline(refs["ADAPT"], color="tab:red", ls=":", lw=1.6, label="adaptive (learns $\\omega$)")
    ax.set_yscale("log"); ax.set_xlabel("assumed / true frequency  $\\hat\\omega/\\omega$")
    ax.set_ylabel("MSE"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("Memory robustness: sharper internal models buy depth,\npay in fragility; adaptation avoids oracle-frequency fragility")
    fig.tight_layout(); fig.savefig(FIGS / "fig10_mismatch.png", dpi=180)

    # ---- figB2: nonstationary bars ----
    dfn = pd.read_csv(DATA / "b_nonstationary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, case in zip(axes, ["chirp", "switching"]):
        sub = dfn[dfn.case == case].groupby("controller")["mse"]
        order = ["oracle-RES (rho=.999)", "oracle-RES detuned (rho=.99)", "PI", "ADAPT (learned)"]
        m = sub.mean().reindex(order); e = sub.sem().reindex(order)
        ax.bar(range(len(order)), m.values, yerr=1.96 * e.values, width=0.58,
               color=[PAL["oracle"], PAL["oracle2"], PAL["pi"], PAL["adapt"]],
               edgecolor="white", linewidth=0.8, capsize=3, zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["oracle\n$\\rho$=.999", "oracle detuned\n$\\rho$=.99", "PI", "adaptive"], fontsize=8)
        ax.set_title(case); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("MSE")
    fig.suptitle("Nonstationary disturbances: oracle memory goes stale, learned memory tracks", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "fig11_nonstationary.png", dpi=180, bbox_inches="tight")

    # ---- figC1: delay vs predictability ----
    dfd = pd.read_csv(DATA / "c_delay.csv")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for a, c in zip([0.0, 0.9, 0.99], ["tab:gray", "tab:blue", "tab:purple"]):
        sub = dfd[dfd.a == a].sort_values("tau")
        ax.plot(sub.tau, sub.riccati, "-", color=c, lw=1.6,
                label=f"theory floor, $a={a}$ (Riccati)")
        ax.errorbar(sub.tau, sub.kalman_mse, yerr=1.96 * sub.kalman_sem,
                    fmt="o", color=c, ms=5, capsize=2)
        # PI: plot stable points; mark unstable ones explicitly
        stab = sub[~sub.pi_unstable]
        ax.plot(stab.tau, stab.pi_mse, "x--", color=c, alpha=0.5, lw=1)
        unst = sub[sub.pi_unstable]
        if len(unst):
            ymax = dfd.kalman_mse.max()
            ax.scatter(unst.tau, [ymax * 1.5] * len(unst), marker="v",
                       color=c, s=40, edgecolor="k", zorder=5)
    ax.text(0.98, 0.02, "$\\blacktriangledown$ PI unstable",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("observation delay $\\tau$"); ax.set_ylabel("MSE")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("Theorem check: floor = disturbance innovation over $\\tau{+}1$\n"
                 "+ delayed-sensing estimation uncertainty (opt. sim matches $<1\\%$)")
    fig.tight_layout(); fig.savefig(FIGS / "fig12_delay_prediction.png", dpi=180)

    # ---- figE1: nonlinear ----
    dfe = pd.read_csv(DATA / "e_nonlinear.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lad = dfe[dfe.exp == "ladder"]
    for N, c in zip([10, 300], ["tab:orange", "tab:blue"]):
        g = lad[lad.N == N].groupby("d")["mse"]
        axes[0].errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.sem().values,
                         fmt="o-", color=c, label=f"$N={N}$", capsize=3)
    axes[0].set_yscale("log"); axes[0].set_xlabel("per-agent memory $d$")
    axes[0].set_ylabel("MSE"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Nonlinear plant ($+0.15\\sin x$):\nmemory helps once SNR is sufficient")
    sw = dfe[dfe.exp == "switch"].groupby("controller")["mse"]
    order = ["oracle-RES@0.007", "PI", "ADAPT"]
    m = sw.mean().reindex(order); e = sw.sem().reindex(order)
    axes[1].bar(range(3), m.values, yerr=1.96 * e.values, width=0.55,
                color=[PAL["oracle"], PAL["pi"], PAL["adapt"]],
                edgecolor="white", linewidth=0.8, capsize=3, zorder=3)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["oracle-RES\n(stale)", "PI", "adaptive"], fontsize=9)
    axes[1].set_ylabel("MSE"); axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_title("Nonlinear + regime switching:\ninternal-model memory beats memoryless PI")
    fig.tight_layout(); fig.savefig(FIGS / "fig13_nonlinear.png", dpi=180)
    print("all Phase-2 figures saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--d", type=int, default=0)
    a = ap.parse_args()
    {"map": lambda: cmd_map(a.d), "map-adaptive": cmd_map_adaptive,
     "robustness": cmd_robustness, "nonstationary": cmd_nonstationary,
     "delay": cmd_delay, "nonlinear": cmd_nonlinear,
     "equal-budget": cmd_equal_budget, "figures": cmd_figures}[a.cmd]()
