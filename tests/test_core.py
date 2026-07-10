import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cdl.sim import simulate, NoiseCfg, CtrlCfg, internal_state_dim
from cdl.metrics import fit_floor

NOISE = NoiseCfg(freqs=(0.01,), amps=(0.3,), sigma_process=0.05, sigma_meas=5.0)


def run(kind, N=50, **kw):
    return simulate(N=N, T=6000, seeds=3, delay=5, noise=NOISE,
                    ctrl=CtrlCfg(kind=kind, **kw), burn_in=1000, rng_seed=1)


def test_stability():
    for kind in ["P", "PI", "PID", "CASCADE"]:
        assert not run(kind)["unstable"], kind


def test_p_has_drift_floor():
    m1 = run("P", N=10)["mse"].mean()
    m2 = run("P", N=1000)["mse"].mean()
    # floor: large-N MSE stays near the analytic drift residual A^2/(2 kp^2)
    analytic = 0.3 ** 2 / (2 * 0.12 ** 2)
    assert m2 > 0.5 * analytic
    assert abs(m1 - m2) / m2 < 1.0  # weak N-dependence at large N


def test_memory_beats_width():
    # flat PI at N=100 must beat flat P at N=1000 (memory > width)
    assert run("PI", N=100)["mse"].mean() < run("P", N=1000)["mse"].mean()


def test_internal_model_rejects_matched_band():
    res = run("RES", res_freqs=(0.01,), res_gain=0.002, N=200)["mse"].mean()
    p = run("P", N=200)["mse"].mean()
    assert res < 0.3 * p


def test_state_dims():
    assert internal_state_dim(CtrlCfg(kind="P")) == 0
    assert internal_state_dim(CtrlCfg(kind="PI")) == 1
    assert internal_state_dim(CtrlCfg(kind="RES", res_freqs=(0.01, 0.03))) == 4


def test_fit_floor_recovers_synthetic():
    Ns = np.array([1, 10, 100, 1000])
    rng = np.random.default_rng(0)
    mse = 5.0 / Ns[:, None] + 0.2 + 0.01 * rng.standard_normal((4, 10))
    f = fit_floor(Ns, mse, n_boot=200)
    assert abs(f["C"] - 0.2) < 0.05
    assert f["C_ci"][0] > 0.1


def test_adaptive_finds_unknown_frequency():
    from cdl.central import run_adaptive_ar
    nz = NoiseCfg(freqs=(0.01,), amps=(0.3,), sigma_process=0.05, sigma_meas=20.0)
    r = run_adaptive_ar(N=300, T=15000, seeds=2, delay=5, noise=nz, pll_gain=0.03)
    w = r["omega_final"][:, 0]
    assert np.all(np.abs(w - 0.01) < 0.003), w
    assert r["mse"].mean() < 1.0


def test_kalman_matches_riccati_floor():
    from cdl.central import run_kalman
    nz = NoiseCfg(freqs=(), amps=(), sigma_process=0.3, sigma_meas=5.0, ar_coef=0.9)
    r = run_kalman(N=100, T=15000, seeds=3, delay=5, noise=nz)
    rel = abs(r["mse"].mean() - r["floor_riccati"]) / r["floor_riccati"]
    assert rel < 0.10, rel
