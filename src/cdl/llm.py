"""E7: LLM-agent version of the stabilization task.

Task ("drifting target tracking"): at each turn the agent receives a noisy
scalar observation of a tracking error and must reply with a numeric
correction. The underlying error is driven by a slow hidden drift plus noise —
the same two-band structure as the control experiments, translated into a
language-agent setting.

Conditions:
  flat : one LLM agent, context window limited to the last K observations
         (a bounded-memory reactive controller). Ensembles of n flat agents
         are averaged — the LLM analogue of increasing N.
  deep : the same fast agent PLUS a slow "strategist" LLM invoked every
         `period` turns; it sees a long down-sampled history, estimates the
         drift, and its one-line guidance is injected into the fast agent's
         context. This adds one nested slow loop (lambda: 1 -> 2).

Backends:
  MockLLM      : deterministic bounded-rationality surrogate used to validate
                 the pipeline without API access. NOT a substitute for real
                 LLM results — clearly labeled in outputs.
  AnthropicLLM : real API via https://api.anthropic.com (set ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

import numpy as np

FAST_SYSTEM = (
    "You are a feedback controller. Each turn you get the recent noisy "
    "observations of a tracking error (positive = too high). Reply with ONLY "
    "a single number: the correction to apply this turn (it will be "
    "subtracted from the error). Be conservative: observations are noisy."
)
STRATEGIST_SYSTEM = (
    "You are a slow supervisory controller. You get a long, down-sampled "
    "history of tracking errors. Estimate the persistent slow drift per turn "
    "(trend that averaging over many turns reveals) and reply with ONLY a "
    "single number: the constant bias correction the fast controller should "
    "add every turn to cancel that drift."
)


def _extract_number(text: str) -> float:
    m = re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", text)
    return float(m[0]) if m else 0.0


class MockLLM:
    """Deterministic surrogate with plausible bounded-rationality behavior.

    fast   : proportional response to the mean of the visible window.
    slow   : robust trend estimate (mean of history) as bias guidance.
    Used only to validate the harness; results are labeled MOCK.
    """
    name = "mock"

    def fast(self, window, guidance):
        u = 0.5 * float(np.mean(window))
        if guidance is not None:
            u += guidance
        return u

    def slow(self, history):
        return float(np.mean(history))


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to run real LLM experiments.")
        self.model = model

    def _call(self, system, user):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": self.model, "max_tokens": 50,
                             "system": system,
                             "messages": [{"role": "user", "content": user}]}).encode(),
            headers={"content-type": "application/json",
                     "x-api-key": self.key,
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        return _extract_number("".join(b.get("text", "") for b in data["content"]))

    def fast(self, window, guidance):
        obs = ", ".join(f"{v:+.2f}" for v in window)
        g = (f" A slow supervisor estimated a persistent drift; add this bias "
             f"to your correction every turn: {guidance:+.3f}.") if guidance is not None else ""
        return self._call(FAST_SYSTEM, f"Recent observations (oldest first): {obs}.{g} Correction:")

    def slow(self, history):
        obs = ", ".join(f"{v:+.2f}" for v in history)
        return self._call(STRATEGIST_SYSTEM,
                          f"Down-sampled error history (oldest first): {obs}. Bias correction per turn:")


def run_episode(llm, deep: bool, T: int = 120, window: int = 4,
                strategist_period: int = 20, drift: float = 0.25,
                sigma: float = 1.0, seed: int = 0, n_ensemble: int = 1):
    """Simulate one episode. Hidden dynamics: e[t+1] = e[t] - u[t] + drift + noise.
    The constant drift is the 'slow band'; per-turn noise is the 'fast band'.
    Returns per-turn absolute error trace."""
    rng = np.random.default_rng(seed)
    e = 0.0
    obs_hist, err_trace = [], []
    guidance = None
    for t in range(T):
        obs = e + sigma * rng.standard_normal()
        obs_hist.append(obs)
        if deep and t > 0 and t % strategist_period == 0:
            guidance = llm.slow(obs_hist[max(0, len(obs_hist) - 60)::3])
        w = obs_hist[-window:]
        u = float(np.mean([llm.fast(w, guidance) for _ in range(n_ensemble)]))
        e = e - u + drift + 0.2 * rng.standard_normal()
        err_trace.append(abs(e))
    return np.array(err_trace)
