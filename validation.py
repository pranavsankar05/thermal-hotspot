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
    """Linearity: response at 90% load = 4.5 x response at 20% load."""
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


def v3_energy_balance(n_cells=50):
    """Steady state (own CG): sum q = sum gamma*theta + (alpha/h^2) * sum theta * (wall neighbours)."""
    h = L / n_cells
    gamma = make_gamma(n_cells)
    q = make_q(n_cells, RACKS, [0.9, 0.9, 0.5, 0.5, 0.2, 0.2])
    th, its = steady_solve(n_cells, ALPHA, gamma, q, method="cg")
    ti = th[1:-1, 1:-1]
    n = n_cells - 1
    nw = np.zeros((n, n))
    nw[0, :] += 1; nw[-1, :] += 1; nw[:, 0] += 1; nw[:, -1] += 1
    heat_in = q[1:-1, 1:-1].sum()
    cooling = (gamma[1:-1, 1:-1] * ti).sum()
    wall = (ALPHA / h**2) * (ti * nw).sum()
    return heat_in, cooling, wall, abs(heat_in - cooling - wall) / heat_in


def v5_steady_crosscheck(n_cells=50):
    """Time-marched steady state vs scipy spsolve vs own CG (max abs differences, in K)."""
    h = L / n_cells
    gamma = make_gamma(n_cells)
    q = make_q(n_cells, RACKS, [0.9, 0.9, 0.5, 0.5, 0.2, 0.2])
    dt = stable_dt(ALPHA, h, gamma.max())
    th_t, t_ss = run_to_steady(np.zeros_like(q), dt, ALPHA, h, gamma, q, tol=1e-8, t_max=5000.0)
    th_d, _ = steady_solve(n_cells, ALPHA, gamma, q, method="direct")
    th_c, its = steady_solve(n_cells, ALPHA, gamma, q, method="cg")
    return (np.max(np.abs(th_c - th_d)),
            np.max(np.abs(th_t - th_d)),
            its, t_ss, th_d.max())


if __name__ == "__main__":
    e = v1_exact()
    print(f"V1 exact-solution RMSE         : {e:.3e}")
    assert e < 5e-3, "V1 failed"

    d = v4_superposition()
    print(f"V4 superposition max error     : {d:.3e}")
    assert d < 1e-9, "V4 failed"

    m = v6_max_principle()
    print(f"V6 minimum theta over run      : {m:.3e}")
    assert m >= -1e-12, "V6 failed"

    qin, cool, wall, rel = v3_energy_balance()
    print(f"V3 heat in={qin:.6f}  cooling={cool:.6f}  wall={wall:.6f}  rel. imbalance={rel:.3e}")
    assert rel < 1e-8, "V3 failed"

    d_cg, d_t, its, t_ss, peak = v5_steady_crosscheck()
    print(f"V5 |CG - spsolve|  max         : {d_cg:.3e} K   (CG iterations = {its})")
    print(f"V5 |marched - spsolve| max     : {d_t:.3e} K   (steady at t = {t_ss:.1f} s)")
    print(f"   steady peak T (mixed load)  : {T_AMB + peak:.2f} C")
    assert d_cg < 1e-8, "V5 CG vs spsolve failed"
    assert d_t < 1e-3, "V5 marched vs spsolve failed"

    print("All Phase 1 + 2 validations passed.")