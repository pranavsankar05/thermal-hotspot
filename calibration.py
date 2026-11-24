import ast
import csv
import os
import re

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import ALPHA, cg

os.makedirs("results", exist_ok=True)

DATA = "data/data"                 # folder created by unpacking data.tar.xz
H_ROOM = 2.0                       # modelled column height [m]
M = 200                            # cells along the column (h = 0.01 m)
Z_U20, DZ_U = 0.979, 0.04445       # height of 20U [m] and height per U [m] (from exhaust_temp.csv)

DESIGNS = {
    "previous":    ("previousDesign",
                    "netbotz_sensor_values_door_opened_temperature.txt",
                    "inrow_cooler_sensor_door_opened.txt"),
    "retrofitted": ("retrofittedDesign",
                    "netbotz_sensor_values_door_closed_temperature.txt",
                    "inrow_cooler_sensor_door_closed.txt"),
}
PAPER_CFD_ERR = {"previous": (0.56, 1.1), "retrofitted": (0.47, 1.6)}   # (mean, max) [C], Kuzay et al. 2022

# Table 1 of Kuzay et al. (2022): (rack, first row [U], height [U], power [W])
SERVERS = [(1, 38, 1, 84), (2, 27, 2, 144), (2, 29, 2, 144), (3, 25, 2, 154), (3, 27, 2, 322),
           (3, 29, 2, 196), (3, 31, 2, 112), (3, 35, 2, 210), (4, 34, 2, 468), (4, 38, 2, 165)]
MODES = {"all racks at each height": "all", "rack 3 only": "rack3"}


def zU(u):
    return Z_U20 + DZ_U * (np.asarray(u, dtype=float) - 20)


# ------------------------------------------------------------------ data loading
def _records(path):
    """Yield (timestamp, list of single-key dicts) from the logger text files."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            for ts, items in rec.items():
                yield ts, items


def load_netbotz(path):
    """Time-averaged rear-door temperature per rack height. Returns (U, mean, std, n_samples)."""
    series = {}
    for ts, items in _records(path):
        for item in items:
            for key, val in item.items():
                m = re.search(r"(\d+)\s*U\b", key)
                if m and val > 5.0:                 # unconnected sensors read 0.0
                    series.setdefault(int(m.group(1)), []).append(float(val))
    if not series:
        raise RuntimeError(f"no sensor temperatures parsed from {path}")
    us = np.array(sorted(series))
    mean = np.array([np.mean(series[u]) for u in us])
    std = np.array([np.std(series[u]) for u in us])
    n = int(np.median([len(series[u]) for u in us]))
    return us, mean, std, n


def load_cooler(path):
    vals = {}
    for ts, items in _records(path):
        for item in items:
            for k, v in item.items():
                vals.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in vals.items()}


def pick(d, part):
    for k, v in d.items():
        if part in k:
            return v
    raise KeyError(part)


def load_cfd(folder):
    """Published CFD profile (temperature, height) or None."""
    try:
        arr = np.genfromtxt(f"{DATA}/Plots/{folder}/data.csv", delimiter=",", skip_header=1)[:, :2]
        arr = arr[np.isfinite(arr).all(axis=1)]
        return arr[:, 0], arr[:, 1]
    except Exception as e:
        print(f"  (CFD overlay skipped: {e})")
        return None


# ------------------------------------------------------------------ model
def source_density(mode, z):
    """Heat input per metre of height [W/m] from the real server powers."""
    P = np.zeros_like(z)
    for rack, first, hgt, watts in SERVERS:
        if mode == "rack3" and rack != 3:
            continue
        z0, z1 = float(zU(first)), float(zU(first + hgt))
        P[(z >= z0) & (z < z1)] += watts / (z1 - z0)
    return P


def solve_profile(ell, P, z):
    """Steady 1D model  -theta'' + theta/ell^2 = P  (alpha scaled to 1), theta(0) = 0 at the floor,
    zero flux at the top (insulated ceiling). Symmetric half-cell form at the top, so A is SPD
    and the project's own CG applies."""
    h = z[1] - z[0]
    gam = 1.0 / ell**2
    main = np.full(M, 2.0 / h**2 + gam)
    main[-1] = 1.0 / h**2 + gam / 2.0
    off = np.full(M - 1, -1.0 / h**2)
    A = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    b = P[1:].copy()
    b[-1] *= 0.5
    th, _ = cg(A, b, tol=1e-12, maxit=5000)
    return np.concatenate([[0.0], th])


def build_phi(P, ells, zs, z):
    Phi = np.empty((len(ells), len(zs)))
    for k, ell in enumerate(ells):
        Phi[k] = np.interp(zs, z, solve_profile(ell, P, z))
    return Phi


def fit(Phi, r, mask):
    """theta = s * phi(ell) is linear in s, so for each ell the best s is closed form."""
    sub, rr = Phi[:, mask], r[mask]
    s = np.maximum((sub @ rr) / (np.sum(sub * sub, axis=1) + 1e-300), 0.0)
    cost = np.sum((rr[None, :] - s[:, None] * sub) ** 2, axis=1)
    k = int(np.argmin(cost))
    return k, float(s[k]), cost


# ------------------------------------------------------------------ main
def crosscheck(us, Tm):
    try:
        arr = np.genfromtxt(f"{DATA}/Plots/previousDesign/exhaust_temp.csv", delimiter=",", skip_header=1)
        d = {int(a[0]): (a[2], a[3]) for a in arr}
        for col, name in enumerate(["column 'Previous Design (C)'", "column 'Retrofitted Design (C)'"]):
            diff = [d[u][col] - t for u, t in zip(us, Tm) if u in d]
            print(f"   my average vs exhaust_temp.csv {name}: RMS difference = {np.sqrt(np.mean(np.square(diff))):.3f} C")
    except Exception as e:
        print(f"   (exhaust_temp.csv cross-check skipped: {e})")


def main():
    z = np.linspace(0.0, H_ROOM, M + 1)
    h = z[1] - z[0]
    ells = np.logspace(-2, 0, 120)                           # decay length scan: 1 cm to 1 m
    Pm = {name: source_density(m, z) for name, m in MODES.items()}
    print("Source check (sum of P*h should be close to the listed power):")
    for name, P in Pm.items():
        print(f"   {name}: {P.sum() * h:.0f} W")
    print(f"   total listed power = {sum(s[3] for s in SERVERS)} W  (paper: 2 kW scenario)\n")

    summary = []
    for design, (folder, nb_file, cool_file) in DESIGNS.items():
        print(f"=== {design} design ===")
        us, Tm, Ts, n = load_netbotz(f"{DATA}/{folder}/{nb_file}")
        cool = load_cooler(f"{DATA}/{folder}/{cool_file}")
        T_ref = pick(cool, "MinRackInletTempMetric")
        print(f"   {len(us)} sensors, {n} samples each (10 s apart).  largest std over time = {Ts.max():.2f} C")
        print(f"   cooler: supply {pick(cool, 'SupplyAirTempMetric'):.1f} C, return {pick(cool, 'ReturnAirTempMetric'):.1f} C, "
              f"min rack inlet {T_ref:.1f} C (used as reference)")
        crosscheck(us, Tm)
        zs, r = zU(us), Tm - T_ref
        base = float(np.sqrt(np.mean((r - r.mean()) ** 2)))
        cfd = load_cfd(folder)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        axes[0].errorbar(Tm, zs, xerr=Ts, fmt="ks", capsize=2, label="measured (mean ± std, 2 h)")
        if cfd is not None:
            axes[0].plot(cfd[0], cfd[1], "r-", lw=1.5, label="published CFD (Kuzay et al.)")
        sensor_rows = {u: [u, round(float(zu), 4), round(t, 3), round(s, 3)] for u, zu, t, s in zip(us, zs, Tm, Ts)}
        header = ["U", "z_m", "measured_mean_C", "measured_std_C"]

        print(f"   {'source assumption':26s} {'ell [m]':>8s} {'s=k/a':>9s} {'RMSE':>6s} {'max err':>8s} "
              f"{'LOO RMSE':>9s} {'baseline':>9s}   gamma=alpha/ell^2   kappa=s*alpha")
        for (name, mode), colour in zip(MODES.items(), ["tab:blue", "tab:green"]):
            P = Pm[name]
            Phi = build_phi(P, ells, zs, z)
            k, s, cost = fit(Phi, r, np.ones(len(r), dtype=bool))
            pred = s * Phi[k]
            err = pred - r
            loo = []
            for j in range(len(r)):
                mask = np.ones(len(r), dtype=bool)
                mask[j] = False
                kj, sj, _ = fit(Phi, r, mask)
                loo.append(sj * Phi[kj, j] - r[j])
            rmse, mx, rmse_loo = (float(np.sqrt(np.mean(err**2))), float(np.abs(err).max()),
                                  float(np.sqrt(np.mean(np.square(loo)))))
            print(f"   {name:26s} {ells[k]:8.3f} {s:9.3e} {rmse:6.2f} {mx:8.2f} {rmse_loo:9.2f} {base:9.2f}   "
                  f"{ALPHA / ells[k]**2:12.3g}   {s * ALPHA:11.3g}")
            if k in (0, len(ells) - 1):
                print("   WARNING: best ell is at the edge of the scan; the fit is not well determined.")
            curve = T_ref + s * solve_profile(ells[k], P, z)
            axes[0].plot(curve, z, color=colour, lw=2, label=f"model: {name} (ell = {ells[k]:.2f} m)")
            axes[1].semilogx(ells, cost, color=colour, label=name)
            axes[1].plot(ells[k], cost[k], "o", color=colour)
            for u, p, e in zip(us, T_ref + pred, err):
                sensor_rows[u] += [round(float(p), 3), round(float(e), 3)]
            header += [f"model_C [{name}]", f"error_C [{name}]"]
            summary.append([design, name, round(float(ells[k]), 4), s, round(rmse, 3), round(mx, 3),
                            round(rmse_loo, 3), round(base, 3)])
        pm, px = PAPER_CFD_ERR[design]
        print(f"   baseline = RMSE of predicting the mean temperature. Paper's CFD errors: mean {pm}, max {px} C.\n")

        axes[0].set_xlabel("temperature at the rear of the rack [°C]")
        axes[0].set_ylabel("height [m]")
        axes[0].set_ylim(0, H_ROOM)
        axes[0].set_title(f"{design} design: measured vs model")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("decay length ell = sqrt(alpha/gamma) [m]")
        axes[1].set_ylabel("sum of squared errors [K²]")
        axes[1].set_title("Fit quality vs ell (best s for each ell)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(f"results/calib_{design}.png", dpi=150)
        plt.close(fig)

        with open(f"results/calib_sensors_{design}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(sensor_rows[u] for u in us)

    with open("results/calib_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "source_assumption", "ell_m", "s_kappa_over_alpha", "rmse_C",
                    "max_error_C", "loo_rmse_C", "baseline_rmse_C"])
        w.writerows(summary)
    print("Only kappa/alpha and ell = sqrt(alpha/gamma) are identifiable from a steady profile.")
    print("gamma and kappa above assume alpha = 0.05 m^2/s and are NOT transferable to the plan-view model.")
    print("Done. Files: results/calib_*.png and results/calib_*.csv")


if __name__ == "__main__":
    main()