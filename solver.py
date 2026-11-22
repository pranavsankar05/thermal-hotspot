import numpy as np

L = 10.0            # room side length [m]
T_AMB = 25.0        # ambient temperature [deg C]
ALPHA = 0.05        # effective diffusivity [m^2/s]
Q_MAX = 0.8         # max heating rate of a rack patch [K/s]
THRESHOLD = 35.0    # hotspot threshold [deg C]

# Rack centres (x, y) in metres
RACKS = [(2.5, 2.5), (2.5, 7.5), (7.5, 2.5), (7.5, 7.5), (3.5, 5.5), (6.5, 4.0)]

# Cooling rates per quadrant [1/s]
ZONES = {"SW": 0.02, "NW": 0.05, "NE": 0.05, "SE": 0.12}

def make_grid(n_cells, L=L):
    """n_cells cells per side -> n_cells+1 nodes per side. Index order [i, j] = [x, y]."""
    h = L / n_cells
    x = np.linspace(0.0, L, n_cells + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return h, x, X, Y


def make_q(n_cells, centres, workloads, q_max=Q_MAX, size_m=1.0, L=L):
    """Heating-rate array q [K/s] on all nodes. Each rack is a size_m x size_m patch."""
    h = L / n_cells
    q = np.zeros((n_cells + 1, n_cells + 1))
    m = int(round(size_m / h))                      # patch width in nodes
    for (cx, cy), w in zip(centres, workloads):
        i0 = int(np.floor(cx / h - m / 2 + 0.5 + 1e-9))
        j0 = int(np.floor(cy / h - m / 2 + 0.5 + 1e-9))
        q[i0:i0 + m, j0:j0 + m] = w * q_max
    return q

def make_gamma(n_cells, zones=ZONES, L=L):
    h, x, X, Y = make_grid(n_cells, L)
    east = X >= L / 2 - 1e-12
    north = Y >= L / 2 - 1e-12
    g = np.empty_like(X)
    g[~east & ~north] = zones["SW"]
    g[~east & north] = zones["NW"]
    g[east & north] = zones["NE"]
    g[east & ~north] = zones["SE"]
    return g

def stable_dt(alpha, h, gamma_max, safety=0.9):
    """Sharp von Neumann limit: dt_max = 1 / (4 alpha/h^2 + gamma_max/2)."""
    return safety / (4 * alpha / h**2 + gamma_max / 2)

def step(th, dt, alpha, h, gamma, q, bc="dirichlet"):
    lap = (th[2:, 1:-1] + th[:-2, 1:-1] + th[1:-1, 2:] + th[1:-1, :-2]
           - 4 * th[1:-1, 1:-1]) / h**2
    new = th.copy()
    new[1:-1, 1:-1] += dt * (alpha * lap - gamma[1:-1, 1:-1] * th[1:-1, 1:-1]
                             + q[1:-1, 1:-1])
    if bc == "dirichlet":
        new[0, :] = 0.0; new[-1, :] = 0.0; new[:, 0] = 0.0; new[:, -1] = 0.0
    else:  # insulated (mirror copy)
        new[0, :] = new[1, :]; new[-1, :] = new[-2, :]
        new[:, 0] = new[:, 1]; new[:, -1] = new[:, -2]
    return new

def simulate(theta0, dt, t_end, alpha, h, gamma, q, bc="dirichlet", save_every=0):
    """March from t=0 to t_end. dt is shrunk slightly so the run ends exactly at t_end
    (a smaller dt stays stable)."""
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    th = theta0.copy()
    max_hist = np.empty(n_steps + 1)
    min_hist = np.empty(n_steps + 1)
    max_hist[0], min_hist[0] = th.max(), th.min()
    snaps = [(0.0, th.copy())]
    for n in range(n_steps):
        t = n * dt
        q_now = q(t) if callable(q) else q
        th = step(th, dt, alpha, h, gamma, q_now, bc)
        max_hist[n + 1], min_hist[n + 1] = th.max(), th.min()
        if save_every and (n + 1) % save_every == 0:
            snaps.append(((n + 1) * dt, th.copy()))
    return {"theta": th, "dt": dt, "n_steps": n_steps,
            "t": np.linspace(0, t_end, n_steps + 1),
            "max_hist": max_hist, "min_hist": min_hist, "snaps": snaps}

def run_to_steady(theta0, dt, alpha, h, gamma, q, bc="dirichlet",
                  tol=1e-6, t_max=2000.0):
    """March until max|theta_new - theta_old| / dt < tol [K/s]. Returns (theta, t_steady)."""
    th = theta0.copy()
    t = 0.0
    while t < t_max:
        new = step(th, dt, alpha, h, gamma, q, bc)
        rate = np.max(np.abs(new - th)) / dt
        th = new
        t += dt
        if rate < tol:
            return th, t
    return th, t   # did not converge within t_max