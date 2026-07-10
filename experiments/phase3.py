"""Phase-3 diagnostic figures (colleague's visualization suggestions).

    python experiments/phase3.py concept       # resource schematic (no data)
    python experiments/phase3.py zones          # N x d map with regime zones
    python experiments/phase3.py psd            # disturbance vs residual spectra
    python experiments/phase3.py robust2d       # mismatch x damping heatmap
    python experiments/phase3.py adaptive-dyn   # online frequency ID + sliding MSE
    python experiments/phase3.py spatial        # coupled 1-D ring: memory vs width
    python experiments/phase3.py all
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from scipy.signal import welch
from cdl.sim import simulate, NoiseCfg, CtrlCfg
from cdl.central import run_kalman, run_adaptive_ar

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "results/data", ROOT / "results/figures"
W3 = (0.003, 0.015, 0.06)
NOISE3 = NoiseCfg(freqs=W3, amps=(0.3, 0.3, 0.3), sigma_process=0.05, sigma_meas=20.0)


# ============================================================= concept diagram
def cmd_concept():
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, Polygon, Circle
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    BLUE, GREEN, CORAL, INK = "#185FA5", "#1D9E75", "#D85A30", "#26215C"
    BLUE_F, GREEN_F, CORAL_F = "#E6F1FB", "#E1F5EE", "#FAECE7"

    ax.text(50, 97, "The scalable-control trilemma", ha="center",
            fontsize=20, fontweight="bold", color=INK)
    ax.text(50, 92.5, "three resources, three distinct jobs — none substitutes for another",
            ha="center", fontsize=11.5, color="#555", style="italic")

    # triangle vertices (Delay lowered so its circle clears the subtitle)
    V = {"W": (18, 30), "M": (82, 30), "D": (50, 74)}
    tri = Polygon([V["W"], V["M"], V["D"]], closed=True, fill=False,
                  edgecolor="#bbb", lw=1.6, ls=(0, (6, 4)), zorder=1)
    ax.add_patch(tri)

    def vertex(cx, cy, color, fill, icon, title, sub):
        ax.add_patch(Circle((cx, cy), 11.5, facecolor=fill, edgecolor=color,
                            lw=3, zorder=3))
        ax.text(cx, cy + 3.6, icon, ha="center", va="center", fontsize=25,
                color=color, fontweight="bold", zorder=4)
        ax.text(cx, cy - 3.4, title, ha="center", va="center", fontsize=13.5,
                color=color, fontweight="bold", zorder=4)
        ax.text(cx, cy - 7.2, sub, ha="center", va="center", fontsize=8.8,
                color="#444", zorder=4)

    vertex(*V["W"], BLUE, BLUE_F, "N", "Width", "averages\nindependent noise")
    vertex(*V["M"], GREEN, GREEN_F, "d", "Memory", "internal model of\nthe disturbance")
    vertex(*V["D"], CORAL, CORAL_F, r"$\tau$", "Delay", "prediction horizon;\ninnovation floor")

    # edge annotations (what each pair trades)
    ax.text(50, 24.5, "width cannot buy spectral coverage", ha="center",
            fontsize=9.5, color="#777", style="italic")
    ax.text(30, 54, "prediction\nneeds memory", ha="center", fontsize=9.5,
            color="#777", style="italic", rotation=55)
    ax.text(70, 54, "delay caps\nany width", ha="center", fontsize=9.5,
            color="#777", style="italic", rotation=-55)

    # center thesis chip
    ax.add_patch(FancyBboxPatch((34, 43), 32, 12, boxstyle="round,pad=0.6,rounding_size=2",
                                facecolor="#FFF7E6", edgecolor=CORAL, lw=1.8, zorder=2))
    ax.text(50, 51.4, "depth = dynamical, not architectural", ha="center",
            fontsize=10.2, fontweight="bold", color=INK, zorder=3)
    ax.text(50, 47.0, "a flat swarm with internal-model memory\nremoves the floor — no hierarchy needed",
            ha="center", fontsize=8.6, color="#555", zorder=3)

    # bottom takeaway bar
    ax.add_patch(FancyBboxPatch((8, 6), 84, 11, boxstyle="round,pad=0.5,rounding_size=2",
                                facecolor="#F1EFE8", edgecolor="#ccc", lw=1, zorder=2))
    ax.text(50, 13.2, "Width reduces averageable uncertainty · memory reduces structured "
            "uncertainty · delay sets the irreducible floor",
            ha="center", fontsize=9.8, color=INK, zorder=3)
    ax.text(50, 8.9, "hierarchy is sufficient, but not necessary — temporal depth can be a "
            "recurrent internal-model state",
            ha="center", fontsize=9.2, color="#666", style="italic", zorder=3)

    fig.savefig(FIGS / "fig14_concept.png", dpi=190, bbox_inches="tight")
    print("saved fig14_concept.png")


# ================================================================ map w/ zones
def cmd_zones():
    df = pd.concat([pd.read_csv(DATA / f"a_map_d{d}.csv") for d in [0, 1, 3, 5, 7]])
    piv = df.groupby(["d", "N"])["mse"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    im = ax.imshow(np.log10(piv.values), aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("population width $N$"); ax.set_ylabel("per-agent memory $d$")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center",
                    fontsize=7, color="w" if np.log10(piv.values[i, j]) > 0.4 else "k")
    fig.colorbar(im, ax=ax, label="$\\log_{10}$ MSE (color)")

    ny, nx = piv.shape
    # zone annotations (schematic regions on the grid)
    ax.text(0.8, ny - 1.3, "variance-limited\n(add width)", fontsize=8.5, color="white",
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#2c7fb8", ec="none", alpha=0.85))
    ax.text(nx - 1.6, 0.35, "memory-limited\n(add internal model)", fontsize=8.5, color="white",
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#31a354", ec="none", alpha=0.85))
    ax.text(nx - 1.5, ny - 1.2, "approaching\ndelay/innovation floor", fontsize=8.5, color="white",
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#d95f0e", ec="none", alpha=0.9))
    ax.set_title("Regime map: width cures variance, memory cures structure,\n"
                 "delay caps the achievable floor")
    fig.tight_layout(); fig.savefig(FIGS / "fig15_zones.png", dpi=180)
    print("saved fig15_zones.png")


# ============================================================= PSD figure
def _disturbance_series(noise, T, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    ph = rng.uniform(0, 2 * np.pi, size=len(noise.freqs))
    freqs = np.asarray(noise.freqs, float); amps = np.asarray(noise.amps)
    d = np.empty(T); pn = 0.0
    for t in range(T):
        ph = ph + freqs
        val = (amps * np.sin(ph)).sum()
        pn = noise.ar_coef * pn + noise.sigma_process * rng.standard_normal()
        d[t] = val + pn
    return d


def cmd_psd():
    T = 60000
    ctrls = [("P (memoryless)", CtrlCfg(kind="P", kp=0.18), "tab:gray"),
             ("PI ($d{=}1$)", CtrlCfg(kind="PI", kp=0.18, ki=0.004), "tab:blue"),
             ("PI+1 res ($d{=}3$)", CtrlCfg(kind="PIRES", kp=0.18, ki=0.004,
                                            res_freqs=W3[:1], res_gain=0.002), "tab:green"),
             ("PI+3 res ($d{=}7$)", CtrlCfg(kind="PIRES", kp=0.18, ki=0.004,
                                            res_freqs=W3, res_gain=0.002), "tab:red")]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    # disturbance reference
    d = _disturbance_series(NOISE3, T)
    fq, pdd = welch(d, fs=1.0, nperseg=8192)
    ax.semilogy(fq, pdd, color="k", lw=2.2, ls=":", label="disturbance (input)")
    for name, c, col in ctrls:
        r = simulate(N=300, T=T, seeds=4, delay=5, noise=NOISE3, ctrl=c,
                     rng_seed=7, return_traj=True)
        # residual PSD averaged over seeds
        ps = []
        for s in range(r["traj"].shape[0]):
            f, p = welch(r["traj"][s], fs=1.0, nperseg=8192)
            ps.append(p)
        ax.semilogy(f, np.mean(ps, axis=0), color=col, lw=1.5, label=name)
    for w in W3:
        ax.axvline(w / (2 * np.pi), color="0.6", lw=0.7, ls="--")
    ax.set_xlim(0, 0.02); ax.set_xlabel("frequency (cycles/step)")
    ax.set_ylabel("power spectral density"); ax.legend(fontsize=8.5)
    ax.set_title("Where the error lives: P averages noise but leaves the\n"
                 "disturbance peaks; each resonator notches out one band")
    fig.tight_layout(); fig.savefig(FIGS / "fig16_psd.png", dpi=180)
    print("saved fig16_psd.png")


# ============================================================= robustness 2D
def cmd_robust2d():
    W0 = 0.01
    noise1 = NoiseCfg(freqs=(W0,), amps=(0.3,), sigma_process=0.05, sigma_meas=20.0)
    ratios = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4])
    damps = np.array([0.99, 0.995, 0.999, 0.9995, 0.9999])
    M = np.zeros((len(damps), len(ratios)))
    for i, rho in enumerate(damps):
        for j, q in enumerate(ratios):
            ctrl = CtrlCfg(kind="RES", kp=0.12, res_freqs=(W0 * q,), res_gain=0.002, res_damp=rho)
            r = simulate(N=300, T=16000, seeds=5, delay=5, noise=noise1, ctrl=ctrl, rng_seed=31)
            M[i, j] = r["mse"].mean()
        print(f"rho={rho} done")
    np.savez(DATA / "robust2d.npz", M=M, ratios=ratios, damps=damps)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    im = ax.imshow(np.log10(M), aspect="auto", origin="lower", cmap="magma_r")
    ax.set_xticks(range(len(ratios))); ax.set_xticklabels([f"{q:.2f}" for q in ratios], fontsize=8)
    ax.set_yticks(range(len(damps))); ax.set_yticklabels([f"{d:.4f}" for d in damps])
    ax.set_xlabel("assumed / true frequency  $\\hat\\omega/\\omega$")
    ax.set_ylabel("resonator damping $\\rho$")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="w" if np.log10(M[i, j]) > -0.2 else "k")
    fig.colorbar(im, ax=ax, label="$\\log_{10}$ MSE (color)")
    ax.axvline(list(ratios).index(1.0), color="cyan", lw=1.4, ls="--")
    ax.set_title("Memory robustness surface: sharp resonators reach the deepest\n"
                 "suppression but only in a narrow band around the true frequency")
    fig.tight_layout(); fig.savefig(FIGS / "fig17_robust2d.png", dpi=180)
    print("saved fig17_robust2d.png")


# ============================================================= adaptive dynamics
def cmd_adaptive_dyn():
    W0 = 0.01
    noise1 = NoiseCfg(freqs=(W0,), amps=(0.3,), sigma_process=0.05, sigma_meas=20.0)
    T = 30000
    r = run_adaptive_ar(N=300, T=T, seeds=1, delay=5, noise=noise1, n_tones=2,
                        pll_gain=0.03, rng_seed=5, log_omega=True, return_traj=True)
    oh = r["omega_hist"]          # (T, n_tones)
    traj = r["traj"][0]
    # sliding-window MSE
    win = 1500
    sq = traj ** 2
    csum = np.cumsum(np.insert(sq, 0, 0))
    smse = (csum[win:] - csum[:-win]) / win
    tt = np.arange(win, T + 1)[:len(smse)]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True)
    axes[0].axhline(W0, color="k", ls="--", lw=1, label="true $\\omega=0.01$")
    prim = oh[:, 0].copy()
    prim[prim == 0] = np.nan
    axes[0].plot(prim, color="tab:red", lw=1.2, label="learned $\\hat\\omega$ (dominant tone)")
    axes[0].set_ylabel("frequency (rad/step)"); axes[0].legend(fontsize=8.5)
    axes[0].set_ylim(0, 0.03)
    axes[0].set_title("Adaptive controller learning the unknown frequency online")
    axes[1].semilogy(tt, smse, color="tab:blue", lw=1.2, label="sliding-window MSE")
    # oracle reference
    ro = simulate(N=300, T=T, seeds=4, delay=5, noise=noise1,
                  ctrl=CtrlCfg(kind="RES", kp=0.12, res_freqs=(W0,), res_gain=0.002), rng_seed=5)
    axes[1].axhline(ro["mse"].mean(), color="tab:green", ls=":", lw=1.6,
                    label=f"oracle RES floor ({ro['mse'].mean():.2f})")
    axes[1].set_ylabel("MSE"); axes[1].set_xlabel("time step $t$"); axes[1].legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIGS / "fig18_adaptive_dyn.png", dpi=180)
    print("saved fig18_adaptive_dyn.png")


# ======================================================== coupled 1-D ring plant
def _spatial_run(M_nodes, N, T, delay, ctrl_kind, res_freq=None,
                 kp=0.15, ki=0.004, res_gain=0.002, res_damp=0.999,
                 w_slow=0.008, A_slow=0.4, sig_local=0.15, sig_meas=20.0,
                 kappa=0.15, seeds=4, burn_in=3000, rng_seed=0):
    """Coupled 1-D ring of M integrator nodes with a SHARED slow disturbance mode
    plus local per-node noise. N agents per node average local noisy delayed
    observations; each node runs a local P / PI / PI+resonator controller."""
    S = seeds
    rng = np.random.default_rng(rng_seed)
    X = np.zeros((S, M_nodes))
    hist = np.zeros((delay + 1, S, M_nodes))
    z = np.zeros((S, M_nodes))                       # integrator state
    rz = np.zeros((S, M_nodes, 2))                   # resonator state
    ph = rng.uniform(0, 2 * np.pi, size=S)
    ct, st = np.cos(res_freq if res_freq else 0.0), np.sin(res_freq if res_freq else 0.0)
    sq, cnt = np.zeros(S), 0
    for t in range(T):
        xdel = hist[t % (delay + 1)]
        y = xdel + (sig_meas / np.sqrt(N)) * rng.standard_normal((S, M_nodes))
        if ctrl_kind == "P":
            u = -kp * y
        elif ctrl_kind == "PI":
            z += y; u = -kp * y - ki * z
        else:  # PIRES on the shared slow mode
            z += y; u = -kp * y - ki * z
            new0 = ct * rz[..., 0] + st * rz[..., 1] + y
            new1 = -st * rz[..., 0] + ct * rz[..., 1]
            rz[..., 0], rz[..., 1] = new0, new1
            u = u - res_gain * new0
        # shared slow mode + local disturbances
        ph = ph + w_slow
        d_shared = (A_slow * np.sin(ph))[:, None]
        d_local = sig_local * rng.standard_normal((S, M_nodes))
        lap = np.roll(X, 1, axis=1) - 2 * X + np.roll(X, -1, axis=1)   # ring Laplacian
        X = X + kappa * lap + u + d_shared + d_local
        X = np.clip(X, -1e6, 1e6)
        hist[t % (delay + 1)] = X
        if t >= burn_in:
            sq += (X ** 2).mean(axis=1); cnt += 1   # per-seed spatial-mean SE
    return sq / max(cnt, 1)                          # shape (S,)


def _grid2d_run(M, N, T, delay, ctrl_kind, res_freq=None, kp=0.15, ki=0.004,
                res_gain=0.002, w_slow=0.008, A_slow=0.4, sig_local=0.15,
                sig_meas=20.0, kappa=0.12, obs_frac=1.0, seeds=6, burn_in=3000,
                rng_seed=0):
    """M x M grid of coupled nodes (2D ring Laplacian), shared slow mode plus
    local noise. PARTIAL OBSERVABILITY: only a fraction obs_frac of nodes carry
    sensors/actuators; the rest are steered only through diffusion from
    neighbours. Returns per-seed grid-mean MSE over ALL nodes."""
    S = seeds
    rng = np.random.default_rng(rng_seed)
    X = np.zeros((S, M, M))
    hist = np.zeros((delay + 1, S, M, M))
    z = np.zeros((S, M, M))
    rz = np.zeros((S, M, M, 2))
    ph = rng.uniform(0, 2 * np.pi, size=S)
    ct, st = np.cos(res_freq if res_freq else 0.0), np.sin(res_freq if res_freq else 0.0)
    mask = (rng.random((M, M)) < obs_frac).astype(float)
    sq, cnt = np.zeros(S), 0
    for t in range(T):
        xdel = hist[t % (delay + 1)]
        y = (xdel + (sig_meas / np.sqrt(N)) * rng.standard_normal((S, M, M))) * mask
        if ctrl_kind == "P":
            u = -kp * y
        elif ctrl_kind == "PI":
            z += y; u = -kp * y - ki * z
        else:
            z += y; u = -kp * y - ki * z
            new0 = ct * rz[..., 0] + st * rz[..., 1] + y
            new1 = -st * rz[..., 0] + ct * rz[..., 1]
            rz[..., 0], rz[..., 1] = new0, new1
            u = u - res_gain * new0
        u = u * mask
        ph = ph + w_slow
        d_shared = (A_slow * np.sin(ph))[:, None, None]
        d_local = sig_local * rng.standard_normal((S, M, M))
        lap = (np.roll(X, 1, 1) + np.roll(X, -1, 1) +
               np.roll(X, 1, 2) + np.roll(X, -1, 2) - 4 * X)
        X = np.clip(X + kappa * lap + u + d_shared + d_local, -1e6, 1e6)
        hist[t % (delay + 1)] = X
        if t >= burn_in:
            sq += (X ** 2).mean(axis=(1, 2)); cnt += 1
    return sq / max(cnt, 1)


def cmd_spatial2d():
    rows = []
    W_SLOW = 0.008
    for obs in [1.0, 0.5]:
        for label, kind, rf in [("P (d=0)", "P", None),
                                 ("PI (d=1)", "PI", None),
                                 ("PI+res (d=3)", "PIRES", W_SLOW)]:
            for N in [10, 100, 300]:
                mse = _grid2d_run(M=6, N=N, T=14000, delay=5, ctrl_kind=kind,
                                  res_freq=rf, w_slow=W_SLOW, obs_frac=obs,
                                  seeds=6, rng_seed=73)
                for s, m in enumerate(mse):
                    rows.append({"obs": obs, "ctrl": label, "N": N, "seed": s, "mse": float(m)})
                print(f"grid2d obs={obs} {label} N={N}: {mse.mean():.4f} +/- {mse.std()/np.sqrt(len(mse)):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "spatial2d.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    palette = {"P (d=0)": "#8c8c8c", "PI (d=1)": "#2c7fb8", "PI+res (d=3)": "#31a354"}
    for ax, obs, ttl in zip(axes, [1.0, 0.5],
                            ["full observability", "partial observability (50% of nodes)"]):
        sub0 = df[df.obs == obs]
        for label, col in palette.items():
            g = sub0[sub0.ctrl == label].groupby("N")["mse"]
            ax.errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.sem().values,
                        fmt="o-", color=col, label=label, lw=2.0, ms=6, capsize=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("agents per node  $N$"); ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=0.2, which="both")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("grid-mean MSE"); axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("2-D grid (6$\\times$6 coupled nodes, shared slow mode): the memory "
                 "ordering survives\ncoupling dimension and partial observability", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "fig20_spatial2d.png", dpi=180, bbox_inches="tight")
    print("saved fig20_spatial2d.png")


def cmd_spatial():
    rows = []
    W_SLOW = 0.008
    for N in [3, 10, 30, 100, 300]:
        for label, kind, rf in [("P (d=0)", "P", None),
                                 ("PI (d=1)", "PI", None),
                                 ("PI+res (d=3)", "PIRES", W_SLOW)]:
            mse_seeds = _spatial_run(M_nodes=12, N=N, T=16000, delay=5, ctrl_kind=kind,
                                     res_freq=rf, w_slow=W_SLOW, seeds=6, rng_seed=71)
            for s, m in enumerate(mse_seeds):
                rows.append({"N": N, "ctrl": label, "seed": s, "mse": float(m)})
            print(f"spatial N={N} {label}: {mse_seeds.mean():.4f} +/- {mse_seeds.std()/np.sqrt(len(mse_seeds)):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "spatial.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    palette = {"P (d=0)": "#8c8c8c", "PI (d=1)": "#2c7fb8", "PI+res (d=3)": "#31a354"}
    for label, col in palette.items():
        g = df[df.ctrl == label].groupby("N")["mse"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=1.96 * g.sem().values,
                    fmt="o-", color=col, label=label, lw=2.0, ms=6, capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("agents per node  $N$"); ax.set_ylabel("spatial-mean MSE")
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.25, which="both")
    ax.set_title("Coupled 1-D ring (12 nodes, shared slow mode): above a minimum SNR,\n"
                 "width saturates while a resonator on the shared mode lowers the floor",
                 fontsize=11)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "fig19_spatial.png", dpi=180)
    print("saved fig19_spatial.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); a = ap.parse_args()
    cmds = {"concept": cmd_concept, "zones": cmd_zones, "psd": cmd_psd,
            "robust2d": cmd_robust2d, "adaptive-dyn": cmd_adaptive_dyn,
            "spatial": cmd_spatial, "spatial2d": cmd_spatial2d}
    if a.cmd == "all":
        for f in cmds.values():
            f()
    else:
        cmds[a.cmd]()
