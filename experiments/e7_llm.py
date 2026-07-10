"""E7 runner: flat vs. deep LLM agents on the drifting-target task.

Usage:
    python experiments/e7_llm.py --backend mock                 # pipeline check
    ANTHROPIC_API_KEY=... python experiments/e7_llm.py --backend anthropic

Conditions swept: flat (window K), flat-ensemble (n votes; the LLM analogue of
larger N), deep (fast + slow strategist loop). The hypothesis mirrors the
control experiments: ensembling reduces per-turn noise but cannot cancel the
persistent drift; the nested slow loop can.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cdl.llm import MockLLM, AnthropicLLM, run_episode

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
ap.add_argument("--episodes", type=int, default=8)
ap.add_argument("--turns", type=int, default=120)
args = ap.parse_args()

llm = MockLLM() if args.backend == "mock" else AnthropicLLM()

CONDS = {
    "flat (K=4)":           dict(deep=False, n_ensemble=1),
    "flat ensemble (n=5)":  dict(deep=False, n_ensemble=5),
    "deep (fast+slow loop)": dict(deep=True,  n_ensemble=1),
}

results = {}
for name, kw in CONDS.items():
    traces = np.array([run_episode(llm, T=args.turns, seed=s, **kw)
                       for s in range(args.episodes)])
    results[name] = traces
    print(f"{name:24s} mean|e| (last 50% of turns) = "
          f"{traces[:, args.turns // 2:].mean():.3f} +- "
          f"{traces[:, args.turns // 2:].mean(axis=1).std():.3f}")

out = {name: tr.tolist() for name, tr in results.items()}
tag = "MOCK" if args.backend == "mock" else llm.name
json.dump({"backend": tag, "traces": out},
          open(ROOT / f"results/data/e7_llm_{tag.lower()}.json", "w"))

fig, ax = plt.subplots(figsize=(8, 4.4))
for name, tr in results.items():
    m = tr.mean(axis=0)
    ax.plot(np.convolve(m, np.ones(7) / 7, mode="valid"), label=name, lw=1.5)
ax.set_xlabel("turn"); ax.set_ylabel("|tracking error| (7-turn moving average)")
ax.set_title(f"E7 [{tag}]: LLM-agent stabilization — ensemble width vs. nested slow loop")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / f"results/figures/fig7_llm_{tag.lower()}.png", dpi=180)
print("saved results for backend:", tag)
