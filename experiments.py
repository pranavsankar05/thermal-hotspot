import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # save figures, do not open windows
import matplotlib.pyplot as plt

from solver import *

os.makedirs("results", exist_ok=True)

N = 50                                     # cells per side -> 51 x 51 nodes
H, XN, XG, YG = make_grid(N)
T_END = 600.0                              # length of cold-start runs [s]

WORKLOADS = {
    "low (20%)":    [0.2] * 6,
    "medium (50%)": [0.5] * 6,
    "high (90%)":   [0.9] * 6,
    "mixed":        [0.9, 0.9, 0.5, 0.5, 0.2, 0.2],
}
MIXED = WORKLOADS["mixed"]


# ------------------------------------------------------------------ helpers
def save_csv(name, header, rows):
    with open(f"results/{name}", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def metrics(th, h=H):
    """Peak, mean, hotspot location and hotspot area from an excess-temperature field."""
    T = T_AMB + th
    i, j = np.unravel_index(np.argmax(T), T.shape)
    n_hot = int((T > THRESHOLD).sum())
    return {"peak": float(T.max()), "mean": float(T.mean()),
            "x": i * h, "y": j * h, "n_hot": n_hot, "area": n_hot * h * h}


def first_cross(t, max_hist):
    """First time the peak temperature exceeds THRESHOLD (nan if never)."""
    hot = (T_AMB + max_hist) > THRESHOLD
    return float(t[np.argmax(hot)]) if hot.any() else float("nan")


def steady_time(th0, dt, gamma, q, bc="dirichlet", t_max=5000.0):
    """Time-march to steady state. Returns (theta, t_steady, converged)."""
    th, t = run_to_steady(th0, dt, ALPHA, H, gamma, q, bc=bc, tol=1e-6, t_max=t_max)
    return th, t, t < t_max


def draw_heat(ax, T, title, vmin, vmax):
    im = ax.imshow(T.T, origin="lower", extent=[0, L, 0, L],
                   cmap="inferno", vmin=vmin, vmax=vmax)
    if T.max() > THRESHOLD:
        ax.contour(XN, XN, T.T, levels=[THRESHOLD], colors="cyan", linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return im


def march(th0, dt, n_steps, alpha, h, gamma, q, bc="dirichlet"):
    """Exactly n_steps explicit steps (used by the accuracy studies)."""
    th = th0
    for _ in range(n_steps):
        th = step(th, dt, alpha, h, gamma, q, bc)
    return th


def sine_mode(n):
    h, x, X, Y = make_grid(n)
    return h, np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


# ------------------------------------------------------------------ Experiment 1
def exp1():
    print("\n=== Experiment 1: workload (same cooling) ===")
    gamma = make_gamma(N)
    dt = stable_dt(ALPHA, H, gamma.max())
    rows, temps, hists, thetas = [], {}, {}, {}
    for name, w in WORKLOADS.items():
        q = make_q(N, RACKS, w)
        _, t_ss, ok = steady_time(np.zeros_like(q), dt, gamma, q)
        th, _ = steady_solve(N, ALPHA, gamma, q, "cg")
        out = simulate(np.zeros_like(q), dt, T_END, ALPHA, H, gamma, q)
        m = metrics(th)
        t_cross = first_cross(out["t"], out["max_hist"])
        thetas[name], temps[name] = th, T_AMB + th
        hists[name] = (out["t"], T_AMB + out["max_hist"])
        rows.append([name, round(m["peak"], 3), round(m["mean"], 3), m["x"], m["y"],
                     m["n_hot"], round(m["area"], 3), round(t_ss, 1), round(t_cross, 1)])
        print(f"{name:13s} peak={m['peak']:6.2f} C  mean={m['mean']:6.2f} C  "
              f"at ({m['x']:.1f},{m['y']:.1f})  hot cells={m['n_hot']:3d}  "
              f"area={m['area']:.2f} m2  t_steady={t_ss:7.1f} s{'' if ok else ' (NOT converged)'}  "
              f"t_cross35={t_cross:.1f} s")
    save_csv("exp1_workloads.csv",
             ["workload", "peak_C", "mean_C", "x_m", "y_m", "hot_cells", "hot_area_m2",
              "t_steady_s", "t_cross35_s"], rows)

    # superposition check (V4 applied to the steady state)
    err = np.max(np.abs(thetas["high (90%)"] - 4.5 * thetas["low (20%)"]))
    ratio = thetas["high (90%)"].max() / thetas["low (20%)"].max()
    print(f"superposition: peak-rise ratio high/low = {ratio:.4f} (expect 4.5), "
          f"max |high - 4.5 low| = {err:.2e} K")

    vmax = max(T.max() for T in temps.values())
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    for ax, (name, T) in zip(axes.ravel(), temps.items()):
        im = draw_heat(ax, T, f"Steady state, {name}", T_AMB, vmax)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Temperature [°C]", shrink=0.8)
    fig.savefig("results/exp1_workload_heatmaps.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (t, T) in hists.items():
        ax.plot(t, T, label=name)
    ax.axhline(THRESHOLD, color="k", ls="--", lw=0.8, label="35 °C threshold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("peak temperature [°C]")
    ax.set_title("Experiment 1: peak temperature vs time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("results/exp1_maxT_vs_time.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 1b
def exp1b():
    print("\n=== Experiment 1b: burst workload (rack 0 jumps 20% -> 90%) ===")
    gamma = make_gamma(N)
    dt = stable_dt(ALPHA, H, gamma.max())
    q_lo = make_q(N, RACKS, [0.2, 0.5, 0.5, 0.5, 0.5, 0.5])
    q_hi = make_q(N, RACKS, [0.9, 0.5, 0.5, 0.5, 0.5, 0.5])
    t1, t2 = 50.0, 300.0
    th0, _ = steady_solve(N, ALPHA, gamma, q_lo, "cg")       # start from the pre-burst steady state
    q_fn = lambda t: q_hi if t1 <= t < t2 else q_lo
    out = simulate(th0, dt, T_END, ALPHA, H, gamma, q_fn)
    t = out["t"]
    T = T_AMB + out["max_hist"]
    hot = T > THRESHOLD

    pre = T[0]
    peak = T.max()
    t_peak = t[np.argmax(T)]
    t_cross = t[np.argmax(hot)] if hot.any() else float("nan")
    after = np.where((t >= t2) & (~hot))[0]
    t_cool = t[after[0]] if after.size else float("nan")
    time_hot = hot.sum() * out["dt"]
    print(f"pre-burst peak = {pre:.2f} C")
    print(f"burst from t = {t1:.0f} s to {t2:.0f} s")
    print(f"peak during run = {peak:.2f} C at t = {t_peak:.1f} s")
    print(f"time to cross 35 C after burst start = {t_cross - t1:.1f} s")
    print(f"time back below 35 C after burst end = {t_cool - t2:.1f} s")
    print(f"total time above 35 C = {time_hot:.1f} s")
    save_csv("exp1b_burst_summary.csv",
             ["pre_peak_C", "peak_C", "t_peak_s", "t_cross_after_start_s",
              "t_cool_after_end_s", "time_above_35_s"],
             [[round(pre, 3), round(peak, 3), round(t_peak, 1), round(t_cross - t1, 1),
               round(t_cool - t2, 1), round(time_hot, 1)]])
    save_csv("exp1b_burst_timeseries.csv", ["t_s", "peak_T_C"],
             zip(np.round(t, 3), np.round(T, 4)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, T, label="peak temperature")
    ax.axhline(THRESHOLD, color="k", ls="--", lw=0.8, label="35 °C threshold")
    ax.axvspan(t1, t2, color="orange", alpha=0.2, label="burst (rack 0 at 90%)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("peak temperature [°C]")
    ax.set_title("Experiment 1b: burst workload")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("results/exp1b_burst.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 2
def _uniform(v):
    return {"SW": v, "NW": v, "NE": v, "SE": v}


COOLING = {
    "uniform weak (0.02)":      _uniform(0.02),
    "uniform moderate (0.05)":  _uniform(0.05),
    "uniform strong (0.12)":    _uniform(0.12),
    "4-zone baseline":          ZONES,
    "NW zone failed (0.005)":   {**ZONES, "NW": 0.005},
}


def exp2():
    print("\n=== Experiment 2: cooling layouts (mixed workload) ===")
    q = make_q(N, RACKS, MIXED)
    rows, temps, hists = [], {}, {}
    for name, zones in COOLING.items():
        gamma = make_gamma(N, zones)
        dt = stable_dt(ALPHA, H, gamma.max())
        _, t_ss, ok = steady_time(np.zeros_like(q), dt, gamma, q)
        th, _ = steady_solve(N, ALPHA, gamma, q, "cg")
        out = simulate(np.zeros_like(q), dt, T_END, ALPHA, H, gamma, q)
        m = metrics(th)
        t_cross = first_cross(out["t"], out["max_hist"])
        temps[name] = T_AMB + th
        hists[name] = (out["t"], T_AMB + out["max_hist"])
        rows.append([name, round(m["peak"], 3), m["x"], m["y"], m["n_hot"],
                     round(m["area"], 3), round(t_ss, 1), round(t_cross, 1)])
        print(f"{name:25s} peak={m['peak']:6.2f} C at ({m['x']:.1f},{m['y']:.1f})  "
              f"hot cells={m['n_hot']:3d}  t_steady={t_ss:7.1f} s{'' if ok else ' (NOT converged)'}  "
              f"t_cross35={t_cross:.1f} s")
    save_csv("exp2_cooling.csv",
             ["cooling", "peak_C", "x_m", "y_m", "hot_cells", "hot_area_m2",
              "t_steady_s", "t_cross35_s"], rows)

    vmax = max(T.max() for T in temps.values())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    axs = axes.ravel()
    for ax, (name, T) in zip(axs[:5], temps.items()):
        im = draw_heat(ax, T, name, T_AMB, vmax)
    fig.colorbar(im, ax=axs[:5].tolist(), label="Temperature [°C]", shrink=0.8)
    for name, (t, T) in hists.items():
        axs[5].plot(t, T, label=name)
    axs[5].axhline(THRESHOLD, color="k", ls="--", lw=0.8)
    axs[5].set_xlabel("time [s]")
    axs[5].set_ylabel("peak temperature [°C]")
    axs[5].set_title("Peak temperature vs time", fontsize=9)
    axs[5].legend(fontsize=7)
    axs[5].grid(alpha=0.3)
    fig.savefig("results/exp2_cooling.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 3A
def exp3a():
    print("\n=== Experiment 3A: stability (uniform gamma = 0.12, pure highest mode) ===")
    g0 = 0.12
    gamma = np.full((N + 1, N + 1), g0)
    q = np.zeros((N + 1, N + 1))
    r_sharp = 1.0 / (4 + g0 * H**2 / (2 * ALPHA))     # 4r + g dt/2 <= 1
    r_cons = 1.0 / (4 + g0 * H**2 / ALPHA)            # 4r + g dt   <= 1
    r_mid = 0.5 * (r_sharp + r_cons)
    print(f"sharp limit r <= {r_sharp:.5f},  conservative limit r <= {r_cons:.5f}")

    # initial condition: the highest Dirichlet mode (p = q = N-1), an exact discrete eigenvector
    i = np.arange(N + 1)
    s = np.sin((N - 1) * np.pi * i / N)
    th0 = 1e-3 * np.outer(s, s)

    r_list = [0.20, r_mid, 0.25, 0.30, 0.35]
    n_run, n_meas = 300, 30
    rows, hists = [], {}
    print(f"{'r':>8s} {'dt [s]':>8s} {'4r+g dt/2':>10s} {'4r+g dt':>9s} {'|1-8r|':>8s} "
          f"{'|G| exact':>10s} {'|G| meas':>9s}  region")
    for r in r_list:
        dt = r * H**2 / ALPHA
        th = th0.copy()
        hist = [np.abs(th).max()]
        for _ in range(n_run):
            th = step(th, dt, ALPHA, H, gamma, q)
            hist.append(np.abs(th).max())
        hist = np.array(hist)
        g_meas = (hist[n_meas] / hist[0]) ** (1.0 / n_meas)
        g_simple = abs(1 - 8 * r)
        g_exact = abs(1 - 8 * r * np.cos(np.pi * H / (2 * L)) ** 2 - g0 * dt)
        c_sharp = 4 * r + g0 * dt / 2
        c_cons = 4 * r + g0 * dt
        if c_cons <= 1:
            region = "below both limits"
        elif c_sharp <= 1:
            region = "between sharp and conservative"
        else:
            region = "ABOVE sharp limit (unstable)"
        hists[r] = hist
        rows.append([round(r, 5), round(dt, 5), round(c_sharp, 5), round(c_cons, 5),
                     round(g_simple, 5), round(g_exact, 5), round(g_meas, 5), region])
        print(f"{r:8.4f} {dt:8.4f} {c_sharp:10.5f} {c_cons:9.5f} {g_simple:8.4f} "
              f"{g_exact:10.5f} {g_meas:9.5f}  {region}")
    save_csv("exp3a_stability.csv",
             ["r", "dt_s", "4r+g*dt/2", "4r+g*dt", "abs(1-8r)", "abs_G_exact_discrete",
              "abs_G_measured", "region"], rows)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r, hist in hists.items():
        ax.semilogy(np.maximum(hist, 1e-30), label=f"r = {r:.4f}")
    ax.set_xlabel("time step n")
    ax.set_ylabel("max |theta| [K]")
    ax.set_title("Experiment 3A: growth or decay of the highest mode")
    ax.set_ylim(1e-25, 1e80)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("results/exp3a_stability.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 3B
def exp3b():
    print("\n=== Experiment 3B: accuracy and convergence orders ===")

    # --- spatial order: tiny fixed dt, gamma = 0, compare with exact PDE solution
    t_end, dt = 10.0, 0.001
    n_steps = int(round(t_end / dt))
    rate = 2 * ALPHA * np.pi**2 / L**2
    hs, errs = [], []
    for n in [10, 20, 40, 80]:
        h, th0 = sine_mode(n)
        z = np.zeros_like(th0)
        th = march(th0, dt, n_steps, ALPHA, h, z, z)
        errs.append(rmse(th, th0 * np.exp(-rate * t_end)))
        hs.append(h)
    s_space = fit_slope(hs, errs)
    print(f"spatial  (dt = {dt}): slope = {s_space:.3f}  (expect ~2)")
    for h, e in zip(hs, errs):
        print(f"   h = {h:7.4f} m   RMSE = {e:.3e} K")
    save_csv("exp3b_spatial.csv", ["h_m", "rmse_K"], zip(hs, errs))
    hs_sp, errs_sp = hs, errs

    # --- temporal order: fixed grid, compare with the semi-discrete exact solution
    g0 = 0.05
    h, th0 = sine_mode(N)
    gam = np.full_like(th0, g0)
    z = np.zeros_like(th0)
    lam_h = 8 * ALPHA / h**2 * np.sin(np.pi * h / (2 * L)) ** 2 + g0
    dts, errs = [], []
    for dt_k in [0.1, 0.05, 0.025, 0.0125, 0.00625]:
        n_steps = int(round(t_end / dt_k))
        th = march(th0, dt_k, n_steps, ALPHA, h, gam, z)
        errs.append(rmse(th, th0 * np.exp(-lam_h * t_end)))
        dts.append(dt_k)
    s_time = fit_slope(dts, errs)
    print(f"temporal (forward Euler): slope = {s_time:.3f}  (expect ~1)")
    for d, e in zip(dts, errs):
        print(f"   dt = {d:8.5f} s   RMSE = {e:.3e} K")
    save_csv("exp3b_temporal.csv", ["dt_s", "rmse_K"], zip(dts, errs))
    dts_t, errs_t = dts, errs

    # --- combined: r fixed at 0.2 so dt proportional to h^2
    t_end_c, r = 8.0, 0.2
    hs, errs = [], []
    for n in [10, 20, 40, 80]:
        h, th0 = sine_mode(n)
        dt_c = r * h**2 / ALPHA
        n_steps = int(round(t_end_c / dt_c))
        z = np.zeros_like(th0)
        th = march(th0, t_end_c / n_steps, n_steps, ALPHA, h, z, z)
        errs.append(rmse(th, th0 * np.exp(-rate * t_end_c)))
        hs.append(h)
    s_comb = fit_slope(hs, errs)
    print(f"combined (r = {r}, dt ~ h^2): slope = {s_comb:.3f}  (expect ~2)")
    for h, e in zip(hs, errs):
        print(f"   h = {h:7.4f} m   RMSE = {e:.3e} K")
    save_csv("exp3b_combined.csv", ["h_m", "rmse_K"], zip(hs, errs))
    hs_c, errs_c = hs, errs

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (axes[0], hs_sp, errs_sp, s_space, "h [m]", "Spatial (dt fixed = 0.001 s)"),
        (axes[1], dts_t, errs_t, s_time, "dt [s]", "Temporal (forward Euler, h = 0.2 m)"),
        (axes[2], hs_c, errs_c, s_comb, "h [m]", "Combined (r = 0.2, dt ~ h^2)"),
    ]
    for ax, xs, ys, sl, xl, title in panels:
        ax.loglog(xs, ys, "o-", label=f"fitted slope = {sl:.2f}")
        ax.set_xlabel(xl)
        ax.set_ylabel("RMSE [K]")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp3b_convergence.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 4
def exp4():
    print("\n=== Experiment 4: Dirichlet vs Neumann walls (mixed workload) ===")
    gamma = make_gamma(N)
    dt = stable_dt(ALPHA, H, gamma.max())
    q = make_q(N, RACKS, MIXED)
    z = np.zeros_like(q)

    th_d, _ = steady_solve(N, ALPHA, gamma, q, "cg")
    _, t_d, ok_d = steady_time(z, dt, gamma, q, bc="dirichlet")
    th_n, t_n, ok_n = steady_time(z, dt, gamma, q, bc="neumann")
    out_d = simulate(z, dt, T_END, ALPHA, H, gamma, q, bc="dirichlet")
    out_n = simulate(z, dt, T_END, ALPHA, H, gamma, q, bc="neumann")

    md, mn = metrics(th_d), metrics(th_n)
    rows = []
    for name, m, t_ss, ok in [("Dirichlet", md, t_d, ok_d), ("Neumann", mn, t_n, ok_n)]:
        rows.append([name, round(m["peak"], 3), round(m["mean"], 3), m["x"], m["y"],
                     m["n_hot"], round(m["area"], 3), round(t_ss, 1)])
        print(f"{name:10s} peak={m['peak']:6.2f} C  mean={m['mean']:6.2f} C  "
              f"at ({m['x']:.1f},{m['y']:.1f})  hot cells={m['n_hot']:3d}  "
              f"area={m['area']:.2f} m2  t_steady={t_ss:7.1f} s{'' if ok else ' (NOT converged)'}")
    save_csv("exp4_boundary.csv",
             ["bc", "peak_C", "mean_C", "x_m", "y_m", "hot_cells", "hot_area_m2", "t_steady_s"],
             rows)

    # heat budgets
    heat_in = q[1:-1, 1:-1].sum()
    wall_frac = 1 - (gamma[1:-1, 1:-1] * th_d[1:-1, 1:-1]).sum() / heat_in
    print(f"Dirichlet: fraction of heat leaving through the walls = {wall_frac:.3f}")
    cool_n = (gamma[1:-1, 1:-1] * th_n[1:-1, 1:-1]).sum()
    print(f"Neumann  : sum q = {heat_in:.4f}, sum gamma*theta = {cool_n:.4f}, "
          f"relative imbalance = {abs(heat_in - cool_n) / heat_in:.2e} (walls remove nothing)")

    vmax = max((T_AMB + th_d).max(), (T_AMB + th_n).max())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    im = draw_heat(axes[0], T_AMB + th_d, "Dirichlet walls (theta = 0)", T_AMB, vmax)
    draw_heat(axes[1], T_AMB + th_n, "Neumann walls (insulated)", T_AMB, vmax)
    fig.colorbar(im, ax=axes[:2].tolist(), label="Temperature [°C]", shrink=0.9)
    axes[2].plot(out_d["t"], T_AMB + out_d["max_hist"], label="Dirichlet")
    axes[2].plot(out_n["t"], T_AMB + out_n["max_hist"], label="Neumann")
    axes[2].axhline(THRESHOLD, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("peak temperature [°C]")
    axes[2].set_title("Peak temperature vs time", fontsize=10)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.savefig("results/exp4_boundary.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
ALL = {"exp1": exp1, "exp1b": exp1b, "exp2": exp2, "exp3a": exp3a, "exp3b": exp3b, "exp4": exp4}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(ALL)
    for key in todo:
        ALL[key]()
    print("\nDone. CSV and PNG files are in the results/ folder.")