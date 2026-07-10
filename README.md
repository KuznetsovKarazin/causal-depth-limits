# Width, Memory, and Delay: A Resource Accounting for the Limits of Flat Multi-Agent Systems

**Oleksandr Kuznetsov**¹² · **Emanuele Frontoni**³

¹ Department of Theoretical and Applied Sciences, eCampus University, Novedrate (CO), Italy
² SMARTEST Research Center, eCampus University, Novedrate (CO), Italy · Department of Intelligent Software Systems and Technologies, V. N. Karazin Kharkiv National University, Kharkiv, Ukraine
³ Department of Political Sciences, Communication and International Relations, University of Macerata, Macerata, Italy

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/latex/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Zenodo-blue.svg)](#citation)

**Status:** research code + full experimental results, submitted to *IEEE
Transactions on Cybernetics*.
This repository empirically tests the central claim of *"Causal Depth and Limits
of Scalable Multi-Agent Systems"* (Kovalov, 2026, [Zenodo:18253051](https://zenodo.org/records/18253051)): that flat, homogeneous
multi-agent systems hit an N-independent error floor ("causal floor") on tasks
whose disturbances span separated timescales, and that only hierarchical
(nested-loop) architectures can remove it.

**TL;DR of our findings:** the causal floor is real — but it is **not** caused
by the absence of hierarchy. It is caused by insufficient *per-agent internal
memory* (internal-model content). A perfectly flat, homogeneous swarm whose
agents carry a matched internal model **outperforms a designed
two-loop hierarchy at the same per-agent memory (d=2)**. (Accounting note: the
flat swarm then holds 2N internal states in total, whereas the cascade's slow
loop is a single shared state — so the comparison is between *per-agent memory
classes*, not total state counts. The total-budget view, where N·d is held
fixed, is the subject of the Phase-2 fixed-budget experiment below, and it
favours spending states on memory rather than width, once a minimum sensing budget is met.) The theorem as stated in the paper is
refuted by counterexample; a sharper, defensible claim survives:

> **Refined claim.** Swarm size N buys variance reduction (residual error
> ∝ 1/N in the averageable noise component). Per-agent internal state
> dimension d buys spectral coverage of the disturbance. These resources are
> **not interchangeable**: residual power in a disturbance band not covered by
> the collective internal model is independent of N.

---

## 1. Task and model

Discrete-time integrator plant with multi-band disturbance:

```
x[t+1] = x[t] + u[t] + d[t],       d[t] = Σ_j A_j sin(ω_j t + φ_j) + σ_p η[t]
```

N homogeneous agents each observe `y_i[t] = x[t−τ] + v_i[t]` (delay τ,
per-agent iid measurement noise — this is what large N can average away) and
the applied control is the mean of individual controls. Controller classes,
distinguished only by per-agent **internal state dimension d**:

| class | d | structure |
|---|---|---|
| P | 0 | memoryless proportional (the paper's "flat swarm") |
| PI | 1 | + integrator |
| PID | 2 | + integrator + derivative |
| **RES** | 2 | + resonator matched to the drift band (internal-model principle) — *still flat* |
| **CASCADE** | 2 | fast P loop + slow integral loop on a low-passed aggregate — *the paper's "deep" system* |

**Fairness protocol:** every controller's gains are grid-search tuned on the
task (`experiments/tune.py`, training seeds disjoint from evaluation seeds),
then frozen across all N. No controller is strawmanned.

## 2. Results

### E1–E3 — Scaling and the counterexample (`fig1_scaling.png`, `fig2_psd.png`, `fig3_timeseries.png`)

Fits of `MSE(N) = A/N^α + C` over N ∈ [1, 3000], 12 seeds, bootstrap 95% CI on
the floor C:

| controller | floor C | 95% CI | α |
|---|---|---|---|
| Flat P (d=0) | **1.401** | [1.307, 1.488] | 1.00 |
| Flat PI (d=1) | 0.315 | [0.265, 0.366] | 1.00 |
| Flat PID (d=2) | 0.117 | [0.006, 0.225] | 1.00 |
| Hierarchical cascade (d=2) | 0.107 | [0.067, 0.148] | 1.00 |
| **Flat + internal model (d=2)** | **0.058** | [0.028, 0.088] | 1.00 |

* **E1 (confirms the paper):** the flat memoryless swarm has a hard positive
  floor; α = 1.00 is textbook noise averaging.
* **E2 (confirms):** the two-loop hierarchy reduces the floor ~13×.
* **E3 (refutes the theorem as stated):** a *flat homogeneous* swarm with a
  matched internal model (same d=2 as the cascade, zero hierarchy, zero
  communication) achieves a floor **~2× lower than the hierarchy** and ~24×
  lower than flat P. The spectral view (fig2) shows why: the floor is an
  incompressible spike at the drift band, and it is removed by internal-model
  content, not by loop nesting.

### E4 — Width cannot buy spectral coverage (`fig4_memory_vs_bands.png`)

Two-band disturbance (ω = 0.003 and 0.03). Controllers differ only in which
bands their resonators cover. Residual power in a **covered** band and in the
white band falls with N; residual power in an **uncovered** band is flat in N:
total MSE at N=1000 is 2.96 (covers none), 1.55 / 1.53 (covers one), 0.25
(covers both). Increasing N by 100× does not substitute for two missing state
variables. This is the refined theorem, confirmed.

### E5 — Latency is the real hard limit (`fig5_delay_waterbed.png`)

For the tuned flat PI swarm at N=100: MSE grows with observation delay
(0.61 → 3.29 as τ: 1 → 10) and the loop goes **unstable at τ = 20**, matching
the delay-limited stability margin. The residual PSD shows the Bode waterbed:
suppressing low frequencies pumps error power into mid frequencies, worsening
with delay. Note this constraint binds *hierarchies too* — it is a latency
limit, not a flatness limit.

### E6 — Temporal specialization emerges without design (`fig6_emergence.png`)

Agents are learnable leaky integrators with per-agent parameters (gain,
integral weight, time constant), trained by evolution strategies from a
*homogeneous* initialization; the collective control is a plain mean. Five
training runs per condition:

| condition | median eval MSE | unstable runs |
|---|---|---|
| heterogeneous, 2-band disturbance | **4.27** | 0/5 |
| parameters forced shared (strictly flat), 2-band | 53.7 | **2/5** |
| heterogeneous, 1-band (control) | 1.37 | 0/5 |

In **every** heterogeneous run the population spontaneously splits into a
majority of fast agents (time constants 1–8 steps) and a **single emergent
slow specialist** (time constants ~20–1500 steps, stretching toward the drift
timescale). Forcing homogeneity both degrades performance ~12× and
destabilizes 40% of runs. Honest caveat: we predicted the timescale *spread*
would appear only under multi-band disturbance; in fact a slow specialist also
emerges in the 1-band condition (drift tracking alone already rewards it), so
the clean het-vs-hom contrast, not the M1-vs-M2 contrast, is the supported
result.

Interpretation: "depth" need not be designed — given parameter diversity,
flat collectives *grow* their own timescale hierarchy. This further undermines
the architectural reading of the original theorem while supporting the
resource reading (someone must own the slow state).

### E7 — LLM-agent analogue (`fig7_llm_mock.png`, harness)

`src/cdl/llm.py` implements the same task as a language-agent benchmark:
a fast LLM controller with a bounded observation window, optionally supervised
by a slow "strategist" LLM invoked every 20 turns (one added nested loop,
λ: 1→2), versus flat ensembling (the LLM analogue of larger N).

Included results are from the deterministic **mock backend** (pipeline
validation only — in mock mode ensembling is a no-op by construction; do not
interpret mock numbers as LLM findings). To run the real experiment:

```bash
ANTHROPIC_API_KEY=... make e7-real
```

Hypothesis to test with real models: ensemble width reduces per-turn response
noise but cannot cancel persistent drift; the nested slow loop can — mirroring
E1/E2.

## 3. What this means for the original paper

1. **Confirmed:** flat *memoryless* swarms have an N-independent causal floor
   (Theorem 1 holds for d=0 agents); hierarchy is *sufficient* to reduce it.
2. **Refuted:** hierarchy is not *necessary*. The theorem's quantifier over
   "flat systems" silently assumes memoryless agents; flat swarms with
   internal models beat the hierarchy at equal state dimension.
3. **Refined and supported:** the true dichotomy is width (N) vs. per-agent
   internal memory (d). Uncovered spectral bands are invisible to N (E4);
   latency bounds everyone (E5); and the required slow states can *emerge*
   in flat learned collectives (E6).
4. **Suggested rewrite of the theory:** define causal depth λ via the number
   of dynamically separated internal states in the *closed loop* (internal
   model order), not via architectural nesting. Then "Aeff = μ/λ" should be
   replaced by the two-resource picture, and the impossibility result becomes
   a corollary of the internal-model principle + Bode/delay constraints —
   properly attributable to classical control theory.

## 3b. Phase 2 — the resource triangle, trade-off rates, and a delay theorem

Phase 1 established a *dichotomy* (width vs. memory). Phase 2 turns it into a
quantitative **resource accounting**, adds strong optimal baselines to close
the "weak-baseline" objection, and removes the oracle-frequency crutch that a
reviewer would flag. Everything is chunked under `experiments/phase2.py`.

**Two new strong baselines** (`src/cdl/central.py`), both acting on the
aggregate observation `mean_i y_i = x(t−τ) + v/√N` (one scalar state x, N noisy
sensors) so that width enters *only*
as measurement-noise reduction — cleanly separating the width axis:

* `run_kalman` — oracle-model Kalman-predictive **minimum-variance** control
  (Åström): the theoretical optimum. Its converged Riccati covariance yields a
  **semi-analytic floor** the simulation must match. This is our lower-bound
  frontier; every other controller is measured by its distance to it.
* `run_adaptive_ar` — **no oracle**: reconstructs disturbance increments from
  the plant equation, low-passes them, identifies the drift tones online (EW
  Hann periodogram + peak-picking on a decimated series), and cancels them with
  a bank of PLL-locked **learned resonators** plus soft feedback. It knows
  nothing about the frequencies a priori and tracks chirps / regime switches.

### Block A — the width×memory map (`fig8_map.png`, `fig9_budget.png`)

MSE surface over N ∈ {1…1000} × d ∈ {0,1,3,5,7} on a 3-band disturbance
(ω = 0.003, 0.015, 0.06), the PIRES ladder adding an integrator then one
resonator per band. Floors at N=1000:

| d (PIRES ladder) | 0 (P) | 1 (PI) | 3 (PI+1res) | 5 (PI+2res) | 7 (PI+3res) | adaptive (no-oracle) |
|---|---|---|---|---|---|---|
| MSE floor | 4.49 | 2.40 | 2.05 | 1.66 | **0.44** | **0.47** |

The heatmap contours are **L-shaped**: moving along N (adding width) slides you
down a shallow noise ramp to a fixed floor; moving along d (adding a resonator
that covers another band) is the only way to lower the floor itself. Width and
memory are **not interchangeable** — width cannot buy spectral coverage. The
learned adaptive controller (bottom row) starts *worse* than everything at
small N (it cannot identify tones from few noisy agents) and ends near the
oracle d=7 floor at large N: **adaptivity converts width into memory**, but
only once there is enough signal to learn from.

Fixed-budget slices (N·d ≈ const) confirm the total-state accounting: given a
exact total-state budget (N·d = B, budgets multiples of lcm(1,3,5,7)=105), the story is a threshold: at the smallest budget (B=1050) deep and wide roughly tie (d=7 with N=150 → 2.24 vs d=1 with N=1050 → 2.37, SNR-limited), but past a minimum budget (B≥3150) the same states spent on memory win decisively (d=7 → 0.80 at B=3150, 0.29 at B=12600, vs ≈2.3 for d=1 at every budget). This directly answers the "you just gave each agent more memory" objection.

### Block B — robustness of memory, and the cost of the oracle

*Frequency mismatch × resonator damping* (`fig10_mismatch.png`): a sharp
resonator (ρ = 0.9999) reaches the deepest suppression (MSE ≈ 0.22) but only in
a narrow ±10 % window around the true ω; detune it 30 % and it degrades to
≈ 0.64 — worse than a model-free PI (0.49). A softer resonator (ρ = 0.99) is
nearly useless (≈ 1.0 everywhere): it has traded all its depth away for width.
This is a **waterbed inside the notch**: depth of suppression ↔ width of
robustness is itself a conserved trade-off. The learned adaptive controller
(0.42) beats the *strongly* detuned oracles (≥30% error) and the soft resonator, though a *well-centred* sharp resonator still wins where it is accurate (≈0.22–0.33 within ±15% of ω). The value of adaptation is removing the need to know ω in advance and avoiding the fragility of a mistuned fixed model — the honest way to "know the domain".

*Nonstationary disturbances* (`fig11_nonstationary.png`): under a **chirp**,
the adaptive controller wins (0.40 vs 0.41 for a still-well-centred oracle,
0.51 for PI). Under **regime switching**, results are mixed and we report them
honestly: PI (0.59) actually edges out the adaptive controller (0.79), because
every switch triggers a re-identification transient. **Rule of thumb: adapt
only if regimes outlast your identification time; otherwise a robust
memoryless loop is safer.**

### Block C — the delay theorem (`fig12_delay_prediction.png`)

The central quantitative claim. For a linear plant with delay τ and a
disturbance with rational spectrum, the minimum achievable error equals the
**conditional variance of the controlled next state given the delayed, noisy
observation** — the sum of the disturbance innovation accumulated over the
horizon τ+1 (the classical Åström minimum-variance term) *and* the residual
state-estimation uncertainty from observing only the delayed, noise-corrupted
`x[t−τ]`. (With perfect sensing it collapses to the pure (τ+1)-step disturbance
prediction-error variance.) We verify it: across 15 (a, τ) combinations with
AR(1) process noise (a ∈ {0, 0.9, 0.99}, τ ∈ {1…15}), the simulated optimal MSE
matches the semi-analytic Riccati floor to **< 1 %** (maximum 0.89 %, at longer
horizons T and more seeds). PI control, by contrast, diverges past τ ≈ 10 and is
unstable at τ = 15 (recorded as NaN with an `unstable` flag, not as MSE = 0). The floor grows with τ, with the noise memory a, and
with the sensing noise — all raise the variance of what cannot be reconstructed
and predicted across the delay. **Delay sets an absolute floor equal to the
environment's unpredictability over the latency horizon; neither width nor
memory removes the part you cannot forecast.**

### Block E — nonlinearity (`fig13_nonlinear.png`)

Repeating the key experiments on a nonlinear plant (`x ← x + u + d + 0.15 sin x`)
leaves the memory ordering intact **once width is sufficient** (N=300: d=2 1.76 < d=1 2.32 < d=0 3.65; but at N=10 integrative PI memory amplifies noise, 16.4, worse than plain P, 13.6, while the resonator still leads at 6.5), and under nonlinear +
switching disturbance the stale oracle resonator (2.07) and adaptive controller
(2.45) both beat PI (3.75). The conclusions are not artifacts of linearity.

### The resource triangle (what Phase 2 buys the theory)

| Resource | Buys | Does **not** buy | Rate / law |
|---|---|---|---|
| **Width N** | averaging of iid noise | structural / spectral content | error ∝ 1/N down to a floor; saturates at N* |
| **Per-agent memory d** | spectral coverage, internal model of drift | delay compensation by itself | need d ≥ 2·(#bands); floor set by *uncovered* power |
| **Prediction (memory across the delay)** | cancels the *predictable* part of the disturbance over τ (shifts the floor curve down) | the innovation (unpredictable part) is never removed | floor = disturbance innovation over τ+1 **plus** delayed-sensing estimation uncertainty (Prop. 2; verified <1 %, max 0.89 %) |
| **Adaptation** | replaces oracle knowledge of ω | instant tracking (pays a transient) | asymptotically ≈ oracle − small gap; loses under fast switching |
| **Resonator damping** | width of robustness to model error | depth at that width | notch waterbed: depth × bandwidth ≈ const |

### "Golden rules" for practitioners

1. **Count the disturbance modes first.** Each persistent spectral band needs
   its own internal-model state (≈ 2 states/band). No number of agents
   substitutes for a missing mode — width cannot buy depth.
2. **Add width only up to N\*.** Beyond the point where 1/N noise reduction
   reaches the structural floor, extra agents are ballast. N\* is computable
   from σ² and the target floor.
3. **Latency is the hard wall.** The achievable floor equals the environment's
   unpredictability over the observation (feedback) delay. Buy a predictor or cut latency;
   do not buy more agents.
4. **Under model uncertainty, detune.** Trade notch depth for notch width, or
   go adaptive — but only if regimes outlive the identification transient.

## 3c. Phase 3 — diagnostic figures

Six figures that expose the *mechanism* rather than adding more MSE curves
(`experiments/phase3.py`):

* **`fig14_concept.png`** — the resource triangle schematic (width / memory /
  delay and the three architectures), the paper's opening figure.
* **`fig16_psd.png`** — residual-error spectra for P / PI / PI+res: width scales
  the noise floor, each resonator notches out one disturbance band (the
  internal-model principle made visible).
* **`fig15_zones.png`** — the width×memory map annotated into variance-limited,
  memory-limited, and delay/innovation-limited regimes.
* **`fig17_robust2d.png`** — the depth/robustness waterbed as a 2-D surface over
  frequency mismatch × resonator damping.
* **`fig18_adaptive_dyn.png`** (appendix) — the no-oracle controller learning the
  unknown frequency online and its sliding-window MSE approaching the oracle.
* **`fig19_spatial.png`** — a 12-node coupled chain plant with a shared slow
  mode: width saturates, a resonator on the shared mode lowers the floor, so the
  accounting is not an artifact of the scalar setup.
* **`fig20_spatial2d.png`** — a 6×6 coupled grid under full and 50% partial
  observability: the memory ordering survives both the coupling dimension and
  incomplete sensing.

## 4. Reproducing everything

```bash
pip install -r requirements.txt          # or: conda env create -f environment.yml
make all          # tune + E1..E5 + E6 + E7 mock + Phase 2 + Phase 3 + tests
make phase2       # just the Phase 2 blocks (map, robustness, delay, nonlinear, figures)
make phase3       # diagnostic figures (concept, PSD, zones, robustness, spatial 1D/2D)
make smoke        # fast: rebuild figures from saved data + validate headline claims (CI-friendly)
make check        # validate saved results/data/*.csv against the paper's numeric claims
```

Docker: `docker build -t cdl . && docker run --rm -v $PWD/results:/app/results cdl`.
CI (`.github/workflows/ci.yml`) runs the unit tests, a tuning smoke test, and
`make smoke` on every push.

Total runtime ≈ 25–30 min on a laptop CPU (Phase 1 ≈ 15–20, Phase 2 ≈ 8–10).
All randomness is seeded; `results/` regenerates deterministically.
Configuration lives in `configs/base.yaml`; tuned gains in
`configs/tuned_gains.json`.

## 5. Repository layout

```
src/cdl/sim.py        vectorized plant + controller classes (seeds × agents),
                      now with AR(1) process noise, chirp / regime switching,
                      and optional plant nonlinearity
src/cdl/central.py    Phase 2: Kalman-predictive optimal frontier (+ Riccati
                      floor) and oracle-free adaptive learned-resonator control
src/cdl/metrics.py    band-decomposed PSD, floor fit MSE = A/N^α + C with bootstrap CI
src/cdl/llm.py        LLM-agent harness (Anthropic API + mock backend)
experiments/          tune.py, e1_scaling.py, e4_memory.py, e5_delay.py,
                      e6_emergence.py (chunked ES training), e7_llm.py,
                      phase2.py (map / budget / robustness / nonstationary /
                      delay / nonlinear / figures), phase3.py (concept / PSD /
                      zones / robustness-2D / adaptive dynamics / spatial 1D+2D),
                      check_results.py (validates saved results against the
                      paper's numeric claims; used by `make check`)
configs/              base.yaml (all parameters), tuned_gains.json
results/data          CSV/JSON raw results        results/figures  all figures
tests/                8 unit tests (physics sanity, fit recovery, adaptive
                      frequency ID, Kalman = Riccati floor)
paper/latex/          main.tex, refs.bib, figures/ — the submitted manuscript
                      (compiles to paper/latex/main.pdf with pdflatex+bibtex)
Dockerfile, environment.yml, .github/workflows/ci.yml
```

## 6. Limitations

* 1D integrator plant. Phase 2 adds a nonlinear variant and the ordering holds,
  but higher-dimensional / genuinely chaotic plants are untested.
* The tuning grid is coarse; floors could shift slightly with finer tuning
  (the *ordering* of floors was robust across the grid).
* The adaptive controller assumes the disturbance is a sum of a few narrow
  tones (rational spectrum). Broadband or heavy-tailed disturbances are out of
  scope; its identification stage would need replacing.
* Under fast regime switching the adaptive controller *loses* to a robust PI
  (re-identification transient). We report this openly rather than tuning it
  away; it is the honest boundary of "adapt to know the domain".
* E6 uses tiny populations (8 agents) and a simple ES; emergence at scale is
  untested and it is presented as **exploratory**. E7 real-LLM runs are pending
  (harness ready; `make e7-real` with an API key).
* "Hierarchy" here is one specific cascade; other deep designs exist. Our
  claim is only that hierarchy is not *necessary* for floor removal, which a
  single counterexample suffices to establish.
* Framing: we do not claim to "finally prove" anything. We exhibit a
  counterexample to the strong flat-systems theorem and propose a sharper,
  classically-grounded resource model, verified empirically.

## 7. Data and code availability

All code, configuration, seeds, and raw results (`results/data/*.csv`) needed
to reproduce every figure and number in the paper are in this repository. A
versioned, citable snapshot of this repository is archived on Zenodo:
**[DOI: 10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)**
*(placeholder — filled in after the first Zenodo release; see §8)*.
No proprietary data or paid API access is required to reproduce the core
results; only the optional real-LLM harness (`make e7-real`, Appendix) needs
an Anthropic API key, and its results are explicitly excluded from the paper's
claims (mock-backend results are pipeline validation only).

## 8. Citation

If you use this code or build on this work, please cite:

```bibtex
@article{kuznetsov2026width,
  author  = {Kuznetsov, Oleksandr and Frontoni, Emanuele},
  title   = {Width, Memory, and Delay: A Resource Accounting for the Limits
             of Flat Multi-Agent Systems},
  journal = {IEEE Transactions on Cybernetics},
  year    = {2026},
  note    = {Code and data: \url{https://github.com/KuznetsovKarazin/causal-depth-limits},
             archived at \url{https://doi.org/10.5281/zenodo.XXXXXXX}}
}
```

## 9. License

Code is released under the [MIT License](LICENSE). The manuscript
(`paper/latex/`) is © the authors; see the paper for the applicable IEEE
copyright notice upon publication.

## 10. Contact

Oleksandr Kuznetsov — oleksandr.kuznetsov@uniecampus.it · [ORCID](https://orcid.org/0000-0003-2331-6326)
Issues and questions: please use the [GitHub issue tracker](https://github.com/KuznetsovKarazin/causal-depth-limits/issues).
