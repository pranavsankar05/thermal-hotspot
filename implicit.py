import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from solver import assemble


def implicit_simulate(theta0, dt, t_end, alpha, h, gamma, q, scheme="be", record=False):
    """Backward Euler ('be') or Crank-Nicolson ('cn'), Dirichlet walls (theta = 0).

    With A = alpha*(-Laplacian_h) + diag(gamma):
      BE : (I + dt A) u_new = u + dt q_new
      CN : (I + dt A/2) u_new = (I - dt A/2) u + dt (q_old + q_new)/2
    The left-hand matrix is constant, so it is factorised once with splu and reused.
    q may be an array or a function q(t) returning an array.
    dt is adjusted slightly so the run ends exactly at t_end.
    """
    n_nodes = theta0.shape[0]
    n = n_nodes - 2                                   # interior nodes per side
    A = assemble(n, h, alpha, gamma[1:-1, 1:-1])
    I = sp.identity(n * n, format="csc")

    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps

    if scheme == "be":
        lu = splu((I + dt * A).tocsc())
        B = None
    elif scheme == "cn":
        lu = splu((I + 0.5 * dt * A).tocsc())
        B = (I - 0.5 * dt * A).tocsr()
    else:
        raise ValueError("scheme must be 'be' or 'cn'")

    qf = q if callable(q) else (lambda t: q)
    u = theta0[1:-1, 1:-1].ravel().copy()
    q_old = qf(0.0)[1:-1, 1:-1].ravel()

    max_hist = np.empty(n_steps + 1)
    min_hist = np.empty(n_steps + 1)
    max_hist[0] = u.max()
    min_hist[0] = min(u.min(), 0.0)                   # walls are 0, so they count in the minimum
    fields = []

    def embed(v):
        full = np.zeros_like(theta0)
        full[1:-1, 1:-1] = v.reshape(n, n)
        return full

    if record:
        fields.append(embed(u))

    for k in range(n_steps):
        q_new = qf((k + 1) * dt)[1:-1, 1:-1].ravel()
        if scheme == "be":
            rhs = u + dt * q_new
        else:
            rhs = B @ u + 0.5 * dt * (q_old + q_new)
        u = lu.solve(rhs)
        q_old = q_new
        max_hist[k + 1] = u.max()
        min_hist[k + 1] = min(u.min(), 0.0)
        if record:
            fields.append(embed(u))

    return {"theta": embed(u), "dt": dt, "n_steps": n_steps,
            "t": np.linspace(0.0, t_end, n_steps + 1),
            "max_hist": max_hist, "min_hist": min_hist, "fields": fields}