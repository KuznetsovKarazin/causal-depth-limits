"""Centralized strong baselines.

Both controllers act on the AGGREGATE observation y[t] = x[t-tau] + v/sqrt(N)
(exact for the mean of N iid-noise agents), so the width resource enters only
through measurement-noise reduction — cleanly separating width from memory.

KALMAN  (optimal frontier, oracle model): time-varying Kalman filter on the
        augmented state [x_t, x_{t-1}, ..., x_{t-tau}, p_t] where p_t is the
        AR(1) process-noise state, plus exact feedforward of the known
        sinusoidal drift. Control implements minimum variance:
        u[t] = -E[x[t+1] | y^t, u^{t-1}] (Astrom). The filter's converged
        Riccati covariance yields a SEMI-ANALYTIC floor prediction that the
        simulation must match — this is the Block-C theorem check:
              floor(tau) = Var(x_{t+1} | y^t, u^t)
                         = disturbance innovation over the (tau+1)-step horizon
                           PLUS the delayed-sensing state-estimation uncertainty
                           (M P M^T + sigma_e^2). Under perfect, undelayed
                           sensing this reduces to the classical minimum-variance
                           floor.

ADAPTIVE (learning memory, no oracle): an adaptive INTERNAL-MODEL controller
        (not a full AR/RLS minimum-variance self-tuning regulator). It
        reconstructs noisy disturbance increments from the plant equation,
        low-pass filters them, identifies the drift tones online by peak-picking
        an exponentially-averaged periodogram of a decimated series, and cancels
        them with a bank of PLL-locked learned resonators plus soft feedback.
        Knows NOTHING about the drift frequencies a priori; the forgetting in
        the periodogram lets it track chirps and regime switches.
"""
from __future__ import annotations

import numpy as np
from .sim import NoiseCfg


def run_kalman(N, T, seeds, delay, noise: NoiseCfg, burn_in=2000, rng_seed=0,
               return_traj=False):
    """Oracle-model Kalman-predictive minimum-variance control."""
    S = seeds
    rng = np.random.default_rng(rng_seed)
    tau = delay
    n = tau + 2                                   # [x_t ... x_{t-tau}, p_t]
    a = noise.ar_coef
    sig_v = noise.sigma_meas / np.sqrt(N)         # aggregate measurement noise
    sig_e = noise.sigma_process

    # State-space (control and known sinusoids handled as known inputs):
    # x_t = x_{t-1} + u_{t-1} + s_{t-1} + p_t ;  p_t = a p_{t-1} + e_t
    A = np.zeros((n, n))
    A[0, 0] = 1.0; A[0, n - 1] = a                # x gets previous x + new p_t
    for i in range(1, tau + 1):
        A[i, i - 1] = 1.0                         # delay chain
    A[n - 1, n - 1] = a
    B = np.zeros(n); B[0] = 1.0                   # known input enters x
    G = np.zeros(n); G[0] = 1.0; G[n - 1] = 1.0   # innovation e_t enters x and p
    Q = np.outer(G, G) * sig_e ** 2
    H = np.zeros(n); H[tau] = 1.0                 # measure x_{t-tau}
    R = sig_v ** 2

    # converge Riccati offline (shared across seeds)
    P = np.eye(n)
    for _ in range(3000):
        Pp = A @ P @ A.T + Q
        K = Pp @ H / (H @ Pp @ H + R)
        P = (np.eye(n) - np.outer(K, H)) @ Pp
    # semi-analytic floor: Var(x_{t+1} | y^t) with optimal u cancelling the mean
    # x_{t+1} = x_t + u_t + s_t + p_{t+1};  unknown part: (x_t - xhat) + a(p - phat) + e
    M = np.zeros(n); M[0] = 1.0; M[n - 1] = a
    floor = float(M @ P @ M + sig_e ** 2)

    x = np.zeros(S); p = np.zeros(S)
    xhat = np.zeros((S, n))
    hist_x = np.zeros((tau + 1, S))
    phases = rng.uniform(0, 2 * np.pi, size=(S, len(noise.freqs)))
    freqs = np.asarray(noise.freqs, float)
    amps = np.asarray(noise.amps)
    u_hist = np.zeros((tau + 2, S))               # ring of applied known inputs
    traj = np.empty((S, T)) if return_traj else None
    sq, cnt = np.zeros(S), 0

    def sdrift(t, ph):                            # known sinusoidal drift at step t
        return (amps[None, :] * np.sin(ph + freqs[None, :])).sum(axis=1)

    for t in range(T):
        y = hist_x[t % (tau + 1)] + sig_v * rng.standard_normal(S)
        # predict + update (known input from tau steps ago enters the chain head
        # one step at a time; we log total known input in u_hist)
        uin = u_hist[(t - 1) % (tau + 2)]
        xhat = xhat @ A.T
        xhat[:, 0] += uin                          # known u + s applied at t-1
        innov = y - xhat[:, tau]
        xhat += innov[:, None] * K[None, :]
        # minimum-variance control with exact sinusoid feedforward
        s_now = sdrift(t, phases)
        u = -(xhat[:, 0] + noise.ar_coef * xhat[:, n - 1]) - s_now
        u_hist[t % (tau + 2)] = u + s_now          # total known input at step t

        phases = phases + freqs[None, :]
        p = a * p + sig_e * rng.standard_normal(S)
        x = x + u + s_now + p
        hist_x[t % (tau + 1)] = x
        if return_traj:
            traj[:, t] = x
        if t >= burn_in:
            sq += x * x; cnt += 1
    out = {"mse": sq / max(cnt, 1), "floor_riccati": floor}
    if return_traj:
        out["traj"] = traj
    return out


def run_adaptive_ar(N, T, seeds, delay, noise: NoiseCfg, n_tones=2,
                    lam=0.7, fb_gain=0.15, pll_gain=0.02, decim=25,
                    burn_in=3000, rng_seed=0, gamma_sin=0.0,
                    return_traj=False, log_omega=False):
    """Heuristic adaptive internal-model controller (no oracle knowledge).

    This is an engineering baseline, NOT a self-tuning minimum-variance
    regulator: there is no formal convergence guarantee. It shows the gap
    between oracle frequency knowledge and practical online learning.

    Pipeline:
      1. Reconstruct disturbance increments from the plant equation and
         low-pass them (EMA, alpha=0.1) to suppress measurement noise ~20x.
      2. Identify tone frequencies from an exponentially-averaged Hann
         PERIODOGRAM of a decimated (factor `decim`) copy of the filtered
         increments, by simple peak-picking. The averaging uses the forgetting
         factor `lam`:  Pavg <- lam*Pavg + (1-lam)*P  (smaller lam = faster
         tracking of chirps / regime switches). Decimation moves the slow
         drift bands into a well-resolved part of the spectrum.
      3. A bank of phase-locked-loop (PLL) oscillators locked to the filtered
         increments at the learned frequencies forms the internal state -- a
         LEARNED RESONATOR. Feedforward rotates each oscillator ahead by the
         total pipeline lead (reconstruction delay tau+1 plus EMA group delay)
         and cancels the predicted disturbance at the plant input.
      4. Gentle P feedback (fb_gain, inside the delay-stability margin)
         handles the residual; soft gain keeps injected measurement noise
         negligible, unlike a deadbeat minimum-variance law.

    Memory footprint: the oscillator bank is 2*n_tones states, but the
    controller ALSO carries estimator memory (the decimated spectral buffer,
    the averaged periodogram, and the EMA filter state). It is therefore a
    NO-ORACLE baseline and is not placed on the equal-per-agent-memory axis."""
    S = seeds
    rng = np.random.default_rng(rng_seed)
    tau = delay
    sig_v = noise.sigma_meas / np.sqrt(N)
    alpha = 0.1

    x = np.zeros(S)
    hist_x = np.zeros((tau + 1, S))
    u_hist = np.zeros((tau + 2, S))
    d_filt = np.zeros(S)
    LWIN = 256                                     # decimated analysis window
    dec_buf = np.zeros((LWIN, S))
    hann = np.hanning(LWIN)
    Pavg = np.zeros((S, LWIN // 2 + 1))
    omega = np.zeros((S, n_tones))                 # learned tone frequencies
    osc = np.zeros((S, n_tones, 2))                # PLL oscillators (c, s)
    phases = rng.uniform(0, 2 * np.pi, size=(S, len(noise.freqs)))
    freqs = np.asarray(noise.freqs, float)
    amps = np.asarray(noise.amps)
    pn = np.zeros(S)
    y_prev = None
    kdec = 0
    omega_hist = [] if log_omega else None
    lead = (tau + 1) + (1 - alpha) / alpha          # reconstruction + EMA group delay
    traj = np.empty((S, T)) if return_traj else None
    sq, cnt = np.zeros(S), 0

    for t in range(T):
        y = hist_x[t % (tau + 1)] + sig_v * rng.standard_normal(S)

        # ---------- 1-2: filter + decimated identification ----------
        if y_prev is not None:
            d_raw = y - y_prev - u_hist[(t - tau - 1) % (tau + 2)]
            d_filt = d_filt + alpha * (d_raw - d_filt)
            if t % decim == 0:
                dec_buf[kdec % LWIN] = d_filt
                kdec += 1
                if kdec > LWIN // 2 and kdec % 8 == 0:
                    # EW-averaged periodogram of the decimated window
                    idx = (kdec - LWIN + np.arange(LWIN)) % LWIN
                    seg = dec_buf[idx].T * hann[None, :]          # (S, LWIN)
                    P = np.abs(np.fft.rfft(seg, axis=1)) ** 2
                    Pavg = lam * Pavg + (1 - lam) * P if kdec > LWIN else P
                    for si in range(S):
                        spec = Pavg[si].copy()
                        spec[:2] = 0.0                             # kill DC leakage
                        thr = 6.0 * np.median(spec[2:])
                        picks = []
                        for _ in range(n_tones):
                            b = int(np.argmax(spec))
                            if spec[b] < thr:
                                break
                            picks.append(b)
                            spec[max(0, b - 2):b + 3] = 0.0        # mask +-2 bins
                        for j in range(n_tones):
                            if j < len(picks):
                                omega[si, j] = 2 * np.pi * picks[j] / (LWIN * decim)
                            else:
                                if omega[si, j] != 0.0:
                                    osc[si, j, :] = 0.0
                                omega[si, j] = 0.0

            # ---------- 3: PLL bank locked to filtered increments ----------
            tot = osc[:, :, 0].sum(axis=1)
            err = d_filt - tot
            cw, sw = np.cos(omega), np.sin(omega)
            c, s_ = osc[:, :, 0].copy(), osc[:, :, 1].copy()
            osc[:, :, 0] = cw * c - sw * s_
            osc[:, :, 1] = sw * c + cw * s_
            osc[:, :, 0] += pll_gain * err[:, None] * (omega > 1e-5)
        y_prev = y

        if log_omega:
            omega_hist.append(omega[0].copy())
        # ---------- 4: control ----------
        cl, sl = np.cos(omega * lead), np.sin(omega * lead)
        dff = (cl * osc[:, :, 0] - sl * osc[:, :, 1]).sum(axis=1)
        u = -fb_gain * y - np.clip(dff, -10, 10)
        u = np.clip(u, -50, 50)
        u_hist[t % (tau + 2)] = u

        # ---------- true plant ----------
        w = freqs.copy()
        if noise.chirp_rate:
            w[0] = freqs[0] + noise.chirp_rate * t
        if noise.switch_period:
            w[0] = noise.switch_freqs[(t // noise.switch_period) % len(noise.switch_freqs)]
        phases = phases + w[None, :]
        d = (amps[None, :] * np.sin(phases)).sum(axis=1) if len(freqs) else np.zeros(S)
        pn = noise.ar_coef * pn + noise.sigma_process * rng.standard_normal(S)
        x = x + u + d + pn
        if gamma_sin:
            x = x + gamma_sin * np.sin(x)
        x = np.clip(x, -1e6, 1e6)
        hist_x[t % (tau + 1)] = x
        if return_traj:
            traj[:, t] = x
        if t >= burn_in:
            sq += x * x; cnt += 1
    out = {"mse": sq / max(cnt, 1), "omega_final": omega}
    if log_omega:
        out["omega_hist"] = np.array(omega_hist)
    if return_traj:
        out["traj"] = traj
    return out
