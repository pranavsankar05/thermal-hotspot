import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import *
from implicit import implicit_simulate

os.makedirs("results", exist_ok=True)

N = 50
H, XN, XG, YG = make_grid(N)
MIXED = [0.9, 0.9, 0.5, 0.5, 0.2, 0.2]


# ------------------------------------------------------------------ helpers
def save_csv(name, header, rows):
    with open(f"results/{name}", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def sine_mode(n):
    h, x, X, Y = make_grid(n)
    return h, np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)


def fe_march(th0, dt, n_steps, alpha, h, gamma, q):
    th = th0
    for _ in range(n_steps):
        th = step(th, dt, alpha, h, gamma, q)
    return th


def best_time(fn, repeats=3):
    best, res = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    return res, best


def first_cross(t, max_hist):
    hot = (T_AMB + max_hist) > THRESHOLD
    return float(t[np.argmax(hot)]) if hot.any() else float("nan")


# ------------------------------------------------------------------ check
def check():
    print("\n=== Check: BE and CN reach the same steady state as CG ===")
    gamma = make_gamma(N)
    q = make_q(N, RACKS, MIXED)
    z = np.zeros_like(q)
    th_c, _ = steady_solve(N, ALPHA, gamma, q, "cg")
    for scheme in ["be", "cn"]:
        out = implicit_simulate(z, 10.0, 10000.0, ALPHA, H, gamma, q, scheme)
        d = np.max(np.abs(out["theta"] - th_c))
        print(f"{scheme.upper()} (dt = 10 s, t = 10000 s) vs own CG: max difference = {d:.3e} K")
        assert d < 1e-6, f"{scheme} steady state does not match CG"
    print("Implicit schemes verified.")


# ------------------------------------------------------------------ Experiment 3C
def exp3c():
    print("\n=== Experiment 3C: FE vs BE vs CN (error vs dt and CPU time) ===")
    g0, t_end = 0.05, 10.0
    h, th0 = sine_mode(N)
    gam = np.full_like(th0, g0)
    z = np.zeros_like(th0)
    lam_h = 8 * ALPHA / h**2 * np.sin(np.pi * h / (2 * L)) ** 2 + g0
    exact = th0 * np.exp(-lam_h * t_end)              # semi-discrete exact: isolates the time error
    print(f"explicit stability limit at gamma = {g0}: dt_max = {1/(4*ALPHA/h**2 + g0/2):.4f} s")

    plan = {
        "Forward Euler":  [0.1, 0.05, 0.025, 0.0125, 0.00625],
        "Backward Euler": [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625],
        "Crank-Nicolson": [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625],
    }
    rows, res = [], {}
    for name, dts in plan.items():
        errs, cpus = [], []
        for dt in dts:
            n_steps = int(round(t_end / dt))
            if name == "Forward Euler":
                th, cpu = best_time(lambda: fe_march(th0, dt, n_steps, ALPHA, h, gam, z))
            else:
                key = "be" if name == "Backward Euler" else "cn"
                out, cpu = best_time(lambda: implicit_simulate(th0, dt, t_end, ALPHA, h, gam, z, key))
                th = out["theta"]
            e = rmse(th, exact)
            errs.append(e)
            cpus.append(cpu)
            rows.append([name, dt, n_steps, e, round(cpu * 1000, 3)])
            print(f"{name:15s} dt={dt:8.5f} s  steps={n_steps:5d}  RMSE={e:.3e} K  CPU={cpu*1000:8.2f} ms")
        s = fit_slope(dts, errs)
        res[name] = (dts, errs, cpus, s)
        print(f"   -> fitted time-order slope for {name}: {s:.3f}")
    save_csv("exp3c_schemes.csv", ["scheme", "dt_s", "steps", "rmse_K", "cpu_ms"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    marks = {"Forward Euler": "o", "Backward Euler": "s", "Crank-Nicolson": "^"}
    for name, (dts, errs, cpus, s) in res.items():
        axes[0].loglog(dts, errs, marks[name] + "-", label=f"{name} (slope {s:.2f})")
        axes[1].loglog(np.array(cpus) * 1000, errs, marks[name] + "-", label=name)
    axes[0].set_xlabel("time step dt [s]")
    axes[0].set_ylabel("RMSE at t = 10 s [K]")
    axes[0].set_title("Error vs time step")
    axes[1].set_xlabel("CPU time [ms] (includes factorisation)")
    axes[1].set_ylabel("RMSE at t = 10 s [K]")
    axes[1].set_title("Error vs CPU time")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp3c_schemes.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 3C (switch-on)
def exp3c_switch():
    print("\n=== Experiment 3C-ii: sudden load switch-on at a large time step ===")
    gamma = make_gamma(N)
    q = make_q(N, RACKS, [0.9] * 6)
    z = np.zeros_like(q)
    t_end, dt_big, dt_ref = 100.0, 5.0, 0.1
    probe = (9, 12)                                   # node just outside the left edge of rack 0
    print(f"explicit limit = {1/(4*ALPHA/H**2 + gamma.max()/2):.4f} s; implicit dt = {dt_big} s; "
          f"reference = forward Euler with dt = {dt_ref} s")

    be = implicit_simulate(z, dt_big, t_end, ALPHA, H, gamma, q, "be", record=True)
    cn = implicit_simulate(z, dt_big, t_end, ALPHA, H, gamma, q, "cn", record=True)

    n_ref = int(round(t_end / dt_ref))
    th = z.copy()
    t_ref, p_ref, m_ref = [0.0], [0.0], [0.0]
    for k in range(n_ref):
        th = step(th, dt_ref, ALPHA, H, gamma, q)
        t_ref.append((k + 1) * dt_ref)
        p_ref.append(th[probe])
        m_ref.append(th.max())
    t_ref, p_ref, m_ref = np.array(t_ref), np.array(p_ref), np.array(m_ref)
    th_ref_final = th

    p_be = np.array([f[probe] for f in be["fields"]])
    p_cn = np.array([f[probe] for f in cn["fields"]])

    print(f"minimum theta over all nodes and times: BE = {be['min_hist'].min():.4e} K, "
          f"CN = {cn['min_hist'].min():.4e} K   (exact solution: >= 0)")
    print(f"RMSE of final field vs reference: BE = {rmse(be['theta'], th_ref_final):.3e} K, "
          f"CN = {rmse(cn['theta'], th_ref_final):.3e} K")
    print(f"{'t [s]':>6s} {'ref probe':>10s} {'BE probe':>10s} {'CN probe':>10s}   "
          f"{'ref peak':>9s} {'BE peak':>9s} {'CN peak':>9s}")
    for k in range(0, 7):
        t = k * dt_big
        i = int(round(t / dt_ref))
        print(f"{t:6.1f} {p_ref[i]:10.4f} {p_be[k]:10.4f} {p_cn[k]:10.4f}   "
              f"{m_ref[i]:9.4f} {be['max_hist'][k]:9.4f} {cn['max_hist'][k]:9.4f}")
    rows = [[be["t"][k], p_be[k], p_cn[k], be["max_hist"][k], cn["max_hist"][k],
             be["min_hist"][k], cn["min_hist"][k]] for k in range(len(be["t"]))]
    save_csv("exp3c_switch.csv",
             ["t_s", "probe_BE_K", "probe_CN_K", "peak_BE_K", "peak_CN_K", "min_BE_K", "min_CN_K"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(t_ref, p_ref, "k-", label="reference (FE, dt = 0.1 s)")
    axes[0].plot(be["t"], p_be, "s-", label="Backward Euler, dt = 5 s")
    axes[0].plot(cn["t"], p_cn, "^-", label="Crank-Nicolson, dt = 5 s")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("excess temperature at probe node (1.8 m, 2.4 m) [K]")
    axes[0].set_title("Probe next to a rack edge after switch-on")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(be["t"], be["min_hist"], "s-", label="Backward Euler")
    axes[1].plot(cn["t"], cn["min_hist"], "^-", label="Crank-Nicolson")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("minimum excess temperature [K]")
    axes[1].set_title("Undershoot (exact solution is never below 0)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp3c_switch.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 3D
def exp3d():
    print("\n=== Experiment 3D: steady-state solvers (mixed workload, Dirichlet) ===")
    steady_solve(10, ALPHA, make_gamma(10), make_q(10, RACKS, MIXED), "direct")    # warm-up import
    rows, its_list, invh, t_cg, t_dir, t_mar = [], [], [], [], [], []
    print(f"{'N':>4s} {'unknowns':>9s} {'CG its':>7s} {'CG [ms]':>9s} {'spsolve [ms]':>13s} "
          f"{'march [ms]':>11s} {'|CG-dir|':>10s} {'|march-dir|':>12s} {'peak T [C]':>11s}")
    for n in [20, 40, 80, 160]:
        h = L / n
        gamma = make_gamma(n)
        q = make_q(n, RACKS, MIXED)
        dt = stable_dt(ALPHA, h, gamma.max())
        (th_d, _), td = best_time(lambda: steady_solve(n, ALPHA, gamma, q, "direct"), 2)
        (th_c, its), tc = best_time(lambda: steady_solve(n, ALPHA, gamma, q, "cg"), 2)
        (th_m, t_ss), tm = best_time(
            lambda: run_to_steady(np.zeros_like(q), dt, ALPHA, h, gamma, q, tol=1e-8, t_max=5000.0), 1)
        d_c = np.max(np.abs(th_c - th_d))
        d_m = np.max(np.abs(th_m - th_d))
        peak = T_AMB + th_d.max()
        print(f"{n:4d} {(n-1)**2:9d} {its:7d} {tc*1000:9.1f} {td*1000:13.1f} {tm*1000:11.1f} "
              f"{d_c:10.2e} {d_m:12.2e} {peak:11.3f}")
        rows.append([n, h, (n - 1) ** 2, its, tc * 1000, td * 1000, tm * 1000, d_c, d_m, peak])
        its_list.append(its); invh.append(1 / h); t_cg.append(tc); t_dir.append(td); t_mar.append(tm)
    s = fit_slope(invh, its_list)
    print(f"CG iterations vs 1/h: fitted slope = {s:.3f}  (theory: kappa ~ 1/h^2, iterations ~ sqrt(kappa) ~ 1/h)")
    print("Last column is the grid-independence check V7: the peak settles as h -> 0.")
    save_csv("exp3d_steady_solvers.csv",
             ["N_cells", "h_m", "unknowns", "cg_iterations", "cg_ms", "spsolve_ms",
              "marching_ms", "max_diff_cg_vs_direct_K", "max_diff_march_vs_direct_K", "peak_T_C"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].loglog(invh, its_list, "o-", label=f"own CG (slope {s:.2f})")
    axes[0].set_xlabel("1/h [1/m]")
    axes[0].set_ylabel("CG iterations")
    axes[0].set_title("CG iterations vs grid refinement")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()
    axes[1].loglog([(k - 1) ** 2 for k in [20, 40, 80, 160]], np.array(t_cg) * 1000, "o-", label="own CG")
    axes[1].loglog([(k - 1) ** 2 for k in [20, 40, 80, 160]], np.array(t_dir) * 1000, "s-", label="scipy spsolve")
    axes[1].loglog([(k - 1) ** 2 for k in [20, 40, 80, 160]], np.array(t_mar) * 1000, "^-", label="time marching")
    axes[1].set_xlabel("number of unknowns")
    axes[1].set_ylabel("runtime [ms]")
    axes[1].set_title("Runtime of the three steady-state methods")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("results/exp3d_steady_solvers.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 5
def steady_peak(alpha=ALPHA, zones=ZONES, scale=1.0):
    gamma = make_gamma(N, zones)
    q = make_q(N, RACKS, MIXED, q_max=Q_MAX * scale)
    th, _ = steady_solve(N, ALPHA if alpha is None else alpha, gamma, q, "cg")
    T = T_AMB + th
    return float(T.max()), int((T > THRESHOLD).sum()), T


def _zones_scaled(factor, which=None):
    return {k: (v * factor if (which is None or k == which) else v) for k, v in ZONES.items()}


def exp5():
    print("\n=== Experiment 5: sensitivity (mixed workload, steady state, Dirichlet) ===")
    base_peak, base_n, base_T = steady_peak()
    print(f"baseline: peak = {base_peak:.3f} C, hot nodes = {base_n}")

    cases = [
        ("alpha: 0.02 / 0.10",      dict(alpha=0.02), dict(alpha=0.10)),
        ("q_max: -50% / +50%",      dict(scale=0.5), dict(scale=1.5)),
        ("all gamma: -50% / +50%",  dict(zones=_zones_scaled(0.5)), dict(zones=_zones_scaled(1.5))),
        ("gamma SW: -50% / +50%",   dict(zones=_zones_scaled(0.5, "SW")), dict(zones=_zones_scaled(1.5, "SW"))),
        ("gamma NW: -50% / +50%",   dict(zones=_zones_scaled(0.5, "NW")), dict(zones=_zones_scaled(1.5, "NW"))),
        ("gamma NE: -50% / +50%",   dict(zones=_zones_scaled(0.5, "NE")), dict(zones=_zones_scaled(1.5, "NE"))),
        ("gamma SE: -50% / +50%",   dict(zones=_zones_scaled(0.5, "SE")), dict(zones=_zones_scaled(1.5, "SE"))),
    ]
    rows, plot = [], []
    print(f"{'parameter (low / high)':26s} {'peak low':>9s} {'peak high':>10s} {'nodes low':>10s} {'nodes high':>11s}")
    for label, lo, hi in cases:
        p_lo, n_lo, _ = steady_peak(**lo)
        p_hi, n_hi, _ = steady_peak(**hi)
        print(f"{label:26s} {p_lo:9.3f} {p_hi:10.3f} {n_lo:10d} {n_hi:11d}")
        rows.append([label, round(p_lo, 3), round(p_hi, 3), n_lo, n_hi])
        plot.append((label, p_lo, p_hi, n_lo, n_hi))
    save_csv("exp5_sensitivity.csv",
             ["parameter_low_high", "peak_low_C", "peak_high_C", "hot_nodes_low", "hot_nodes_high"], rows)

    print("\nalpha sweep:")
    a_rows = []
    for a in [0.02, 0.035, 0.05, 0.075, 0.10]:
        p, n_h, _ = steady_peak(alpha=a)
        print(f"  alpha = {a:5.3f} m2/s  peak = {p:6.3f} C  hot nodes = {n_h}")
        a_rows.append([a, round(p, 3), n_h])
    save_csv("exp5_alpha_sweep.csv", ["alpha_m2_per_s", "peak_C", "hot_nodes"], a_rows)

    print("\nthreshold sweep (baseline field):")
    thr_list = [32, 33, 34, 35, 36, 37, 38]
    thr_nodes = [int((base_T > thr).sum()) for thr in thr_list]
    for thr, k in zip(thr_list, thr_nodes):
        print(f"  threshold = {thr} C  hot nodes = {k}  area = {k * H * H:.2f} m2")
    save_csv("exp5_threshold.csv", ["threshold_C", "hot_nodes", "area_m2"],
             [[t, k, round(k * H * H, 3)] for t, k in zip(thr_list, thr_nodes)])

    order = sorted(range(len(plot)), key=lambda i: abs(plot[i][2] - plot[i][1]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for pos, i in enumerate(order):
        label, p_lo, p_hi, n_lo, n_hi = plot[i]
        axes[0].barh(pos, abs(p_hi - p_lo), left=min(p_lo, p_hi))
        axes[1].barh(pos, abs(n_hi - n_lo), left=min(n_lo, n_hi))
    for ax, base, xl in [(axes[0], base_peak, "steady peak temperature [°C]"),
                         (axes[1], base_n, "hot nodes (T > 35 °C)")]:
        ax.axvline(base, color="k", lw=0.8, ls="--", label="baseline")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([plot[i][0] for i in order], fontsize=8)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3, axis="x")
        ax.legend()
    axes[0].set_title("Tornado: peak temperature")
    axes[1].set_title("Tornado: hotspot size")
    axes[2].plot(thr_list, thr_nodes, "o-")
    axes[2].set_xlabel("hotspot threshold [°C]")
    axes[2].set_ylabel("hot nodes")
    axes[2].set_title("Threshold sensitivity (baseline)")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp5_sensitivity.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ Experiment 1b-ii
def exp1b2():
    print("\n=== Experiment 1b-ii: time to trip the alarm vs burst intensity (rack 0) ===")
    gamma = make_gamma(N)
    dt = stable_dt(ALPHA, H, gamma.max())
    q_lo = make_q(N, RACKS, [0.2, 0.5, 0.5, 0.5, 0.5, 0.5])
    th0, _ = steady_solve(N, ALPHA, gamma, q_lo, "cg")
    print(f"pre-burst steady peak = {T_AMB + th0.max():.2f} C")
    rows, curves = [], {}
    print(f"{'burst w':>8s} {'steady peak [C]':>16s} {'t_trip [s]':>11s} {'peak at 300 s [C]':>18s}")
    for wb in [0.6, 0.7, 0.8, 0.9, 1.0]:
        w = [wb, 0.5, 0.5, 0.5, 0.5, 0.5]
        q_b = make_q(N, RACKS, w)
        th_ss, _ = steady_solve(N, ALPHA, gamma, q_b, "cg")
        out = simulate(th0, dt, 300.0, ALPHA, H, gamma, q_b)
        t_trip = first_cross(out["t"], out["max_hist"])
        peak300 = T_AMB + out["max_hist"][-1]
        print(f"{wb:8.2f} {T_AMB + th_ss.max():16.2f} {t_trip:11.1f} {peak300:18.2f}")
        rows.append([wb, round(T_AMB + th_ss.max(), 3), round(t_trip, 1), round(peak300, 3)])
        curves[wb] = (out["t"], T_AMB + out["max_hist"])
    save_csv("exp1b_intensity.csv",
             ["burst_workload", "steady_peak_C", "t_trip_s", "peak_at_300s_C"], rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for wb, (t, T) in curves.items():
        ax.plot(t, T, label=f"rack 0 jumps to {int(wb*100)}%")
    ax.axhline(THRESHOLD, color="k", ls="--", lw=0.8, label="35 °C threshold")
    ax.set_xlabel("time since burst start [s]")
    ax.set_ylabel("peak temperature [°C]")
    ax.set_title("Experiment 1b-ii: ride-through time vs burst intensity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig("results/exp1b_intensity.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
ALL = {"check": check, "exp3c": exp3c, "exp3c_switch": exp3c_switch,
       "exp3d": exp3d, "exp5": exp5, "exp1b2": exp1b2}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(ALL)
    for key in todo:
        ALL[key]()
    print("\nDone. CSV and PNG files are in the results/ folder.")