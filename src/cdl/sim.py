"""Core simulation engine.

Plant: discrete-time integrator  x[t+1] = x[t] + u[t] + d[t]
Disturbance: d[t] = sum_j A_j sin(w_j t + phi_j) + sigma_p * eta[t]   (multi-band)
Agents: N homogeneous agents. Agent i observes y_i[t] = x[t - delay] + v_i[t],
        v_i iid N(0, sigma_m^2)  (independent measurement noise -> 1/N averaging benefit).
Aggregation: u[t] = mean_i u_i[t].

Everything is batched over S seeds and N agents: arrays of shape (S, N).

Controller classes (per agent, homogeneous parameters):
  P        : u_i = -kp * y_i                                (internal state dim d = 0)
  PI       : adds integrator z_i                            (d = 1)
  PID      : adds derivative on noisy obs                   (d = 2)
  RES(d)   : P + d resonators at given frequencies          (d = 2*n_res)  [internal model]
  PI+RES   : PI + resonators                                (d = 1 + 2*n_res)
  CASCADE  : hierarchical two-timescale controller:
             fast P loop on fresh obs + slow integral loop on an exponentially
             low-passed aggregate error (timescale-separated outer loop).
             This is the "deep" system of the paper (lambda = 2).

The flat/deep distinction in the original paper maps to:
  flat  = homogeneous agents, no designed hierarchy  (P, PI, PID, RES are all flat)
  deep  = CASCADE (explicit nested loops on separated timescales)

The decisive comparison is flat internal-model control (RES / PIRES) vs the
two-loop CASCADE at equal per-agent memory: the flat swarm matches or beats the
hierarchy, so architectural nesting is not necessary.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NoiseCfg:
    freqs: tuple = (0.01,)       # rad/step, sinusoidal drift bands
    amps: tuple = (1.0,)         # amplitudes of each band
    sigma_process: float = 0.05  # common (per-seed) process noise innovation
    sigma_meas: float = 1.0      # per-agent iid measurement noise
    ar_coef: float = 0.0         # AR(1) coefficient of process noise (0 = white)
    chirp_rate: float = 0.0      # linear drift of the FIRST band frequency, rad/step^2
    switch_period: int = 0       # if >0, FIRST band frequency alternates every switch_period steps
    switch_freqs: tuple = ()     # alternative frequencies for regime switching


@dataclass
class CtrlCfg:
    kind: str = "P"              # P | PI | PID | RES | PIRES | CASCADE
    kp: float = 0.12
    ki: float = 0.004
    kd: float = 0.30
    res_freqs: tuple = ()        # resonator frequencies (internal model)
    res_gain: float = 0.004
    res_damp: float = 0.999
    # cascade (hierarchical) parameters
    slow_alpha: float = 0.01     # EMA constant of the slow observer (timescale sep.)
    slow_ki: float = 0.004       # slow-loop integral gain


def internal_state_dim(c: CtrlCfg) -> int:
    return {"P": 0, "PI": 1, "PID": 2,
            "RES": 2 * len(c.res_freqs),
            "PIRES": 1 + 2 * len(c.res_freqs),
            "CASCADE": 2}[c.kind]


def simulate(N: int, T: int, seeds: int, delay: int,
             noise: NoiseCfg, ctrl: CtrlCfg,
             burn_in: int = 2000, rng_seed: int = 0,
             return_traj: bool = False, gamma_sin: float = 0.0):
    """Run S independent seeds of an N-agent swarm for T steps.

    Returns dict with per-seed MSE (after burn-in) and optionally the state
    trajectory x of shape (S, T) for spectral analysis.
    """
    S = seeds
    rng = np.random.default_rng(rng_seed)

    x = np.zeros(S)
    hist = np.zeros((delay + 1, S))          # ring buffer of past x for delayed obs
    z = np.zeros((S, N))                     # integrator states
    prev_y = np.zeros((S, N))                # for derivative term
    n_res = len(ctrl.res_freqs)
    rz = np.zeros((S, N, n_res, 2)) if n_res else None
    # precompute resonator rotation matrices
    if n_res:
        rot = np.stack([ctrl.res_damp * np.array([[np.cos(w), -np.sin(w)],
                                                  [np.sin(w),  np.cos(w)]])
                        for w in ctrl.res_freqs])          # (n_res, 2, 2)
    slow_ema = np.zeros(S)                   # cascade: slow low-pass of aggregate error
    slow_int = np.zeros(S)                   # cascade: slow integral state

    phases = rng.uniform(0, 2 * np.pi, size=(S, len(noise.freqs)))
    freqs = np.asarray(noise.freqs, float)
    amps = np.asarray(noise.amps)
    pn = np.zeros(S)                          # AR(1) process-noise state

    traj = np.empty((S, T)) if return_traj else None
    sq_sum = np.zeros(S)
    n_acc = 0
    unstable = False

    for t in range(T):
        # --- observations (delayed plant state + per-agent iid noise) ---
        x_del = hist[t % (delay + 1)]                       # x[t - delay]
        y = x_del[:, None] + noise.sigma_meas * rng.standard_normal((S, N))

        # --- per-agent control ---
        if ctrl.kind == "P":
            u_i = -ctrl.kp * y
        elif ctrl.kind == "PI":
            z += y
            u_i = -ctrl.kp * y - ctrl.ki * z
        elif ctrl.kind == "PID":
            z += y
            u_i = -ctrl.kp * y - ctrl.ki * z - ctrl.kd * (y - prev_y)
            prev_y = y
        elif ctrl.kind in ("RES", "PIRES"):
            u_i = -ctrl.kp * y
            if ctrl.kind == "PIRES":
                z += y
                u_i -= ctrl.ki * z
            for r in range(n_res):
                zr = rz[:, :, r, :]                          # (S, N, 2)
                new0 = rot[r, 0, 0] * zr[..., 0] + rot[r, 0, 1] * zr[..., 1] + y
                new1 = rot[r, 1, 0] * zr[..., 0] + rot[r, 1, 1] * zr[..., 1]
                rz[:, :, r, 0] = new0
                rz[:, :, r, 1] = new1
                u_i -= ctrl.res_gain * new0
        elif ctrl.kind == "CASCADE":
            # fast inner loop (per agent, memoryless)
            u_i = -ctrl.kp * y
        else:
            raise ValueError(ctrl.kind)

        u = u_i.mean(axis=1)

        if ctrl.kind == "CASCADE":
            # slow outer loop operates on the aggregate, timescale-separated
            y_mean = y.mean(axis=1)
            slow_ema += ctrl.slow_alpha * (y_mean - slow_ema)
            slow_int += ctrl.slow_ki * slow_ema
            u = u - slow_int

        # --- disturbance & plant update ---
        w = freqs.copy()
        if noise.chirp_rate:
            w[0] = freqs[0] + noise.chirp_rate * t
        if noise.switch_period:
            w[0] = noise.switch_freqs[(t // noise.switch_period) % len(noise.switch_freqs)]
        phases = phases + w[None, :]          # accumulated instantaneous phase
        d = (amps[None, :] * np.sin(phases)).sum(axis=1)
        pn = noise.ar_coef * pn + noise.sigma_process * rng.standard_normal(S)
        d = d + pn
        x = x + u + d
        if gamma_sin:
            x = x + gamma_sin * np.sin(x)

        hist[t % (delay + 1)] = x
        if return_traj:
            traj[:, t] = x
        if t >= burn_in:
            sq_sum += x * x
            n_acc += 1
        if np.any(np.abs(x) > 1e7):
            unstable = True
            break

    mse = sq_sum / max(n_acc, 1)
    out = {"mse": mse, "unstable": unstable,
           "state_dim": internal_state_dim(ctrl)}
    if return_traj:
        out["traj"] = traj
    return out
