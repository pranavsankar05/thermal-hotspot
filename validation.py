import numpy as np
from solver import *


def v1_exact(n_cells=50, alpha=ALPHA, gamma0=0.05, A=1.0, t_end=10.0):
    """No sources, uniform gamma, Dirichlet walls, sine initial mode. Returns RMSE."""
    h, x, X, Y = make_grid(n_cells)
    th0 = A * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    gamma = np.full_like(th0, gamma0)
    q = np.zeros_like(th0)
    dt = stable_dt(alpha, h, gamma0)
    out = simulate(th0, dt, t_end, alpha, h, gamma, q)
    rate = 2 * alpha * np.pi**2 / L**2 + gamma0
    exact = th0 * np.exp(-rate * t_end)
    return np.sqrt(np.mean((out["theta"] - exact) ** 2))


def v4_superposition(t_end=50.0):
    """Linearity: response at 90% load = 4.5 x response at 20% load (zero initial state)."""
    n = 50
    h, x, X, Y = make_grid(n)
    gamma = make_gamma(n)
    dt = stable_dt(ALPHA, h, gamma.max())
    z = np.zeros_like(gamma)
    lo = simulate(z, dt, t_end, ALPHA, h, gamma, make_q(n, RACKS, [0.2] * 6))["theta"]
    hi = simulate(z, dt, t_end, ALPHA, h, gamma, make_q(n, RACKS, [0.9] * 6))["theta"]
    return np.max(np.abs(hi - 4.5 * lo))


def v6_max_principle(t_end=100.0):
    """With q >= 0 and stable dt, theta must never go below 0."""
    n = 50
    h, x, X, Y = make_grid(n)
    gamma = make_gamma(n)
    dt = stable_dt(ALPHA, h, gamma.max())
    q = make_q(n, RACKS, [0.9, 0.9, 0.5, 0.5, 0.2, 0.2])
    out = simulate(np.zeros_like(q), dt, t_end, ALPHA, h, gamma, q)
    return out["min_hist"].min()


if __name__ == "__main__":
    e = v1_exact()
    print(f"V1 exact-solution RMSE      : {e:.3e}")
    assert e < 5e-3, "V1 failed"

    d = v4_superposition()
    print(f"V4 superposition max error  : {d:.3e}")
    assert d < 1e-9, "V4 failed"

    m = v6_max_principle()
    print(f"V6 minimum theta over run   : {m:.3e}")
    assert m >= -1e-12, "V6 failed"

    print("All Phase 1 validations passed.")