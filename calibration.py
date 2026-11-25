import ast
import csv
import os
import re

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import cg

os.makedirs("results", exist_ok=True)

DATA = "data/data"                 # folder created by unpacking data.tar.xz
H_ROOM = 2.0                       # modelled column height [m]
M = 200                            # cells along the column
Z_U20, DZ_U = 0.979, 0.04445       # height of 20U [m] and height per U [m]
RHO_CP = 1206.0                    # volumetric heat capacity of air [J/(m^3 K)]
A_RACK = 0.6                       # assumed rack cross-section [m^2]; it cancels in q/gamma
GBG = 0.01                         # cooling between servers [1/s]; the data cannot determine it
OFFS = np.linspace(-DZ_U / 2, DZ_U / 2, 9)
ELLS = np.logspace(-2, 0, 120)     # model A: decay length scan [m]
KAPPAS = np.logspace(-6, -1, 26)   # models B, C: mixing diffusivity [m^2/s]
VS = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])   # model C: upward air speed [m/s]

DESIGNS = {
    "previous":    ("previousDesign",
                    "netbotz_sensor_values_door_opened_temperature.txt",
                    "inrow_cooler_sensor_door_opened.txt"),
    "retrofitted": ("retrofittedDesign",
                    "netbotz_sensor_values_door_closed_temperature.txt",
                    "inrow_cooler_sensor_door_closed.txt"),
}
PAPER_CFD_ERR = {"previous": (0.56, 1.1), "retrofitted": (0.47, 1.6)}   # (mean, max) [C]

# Table 1 of Kuzay et al. (2022): (rack, first row [U], height [U], power [W], air flow [m^3/s])
SERVERS = [(1, 38, 1, 84, 0.00740), (2, 27, 2, 144, 0.02390), (2, 29, 2, 144, 0.02390),
           (3, 25, 2, 154, 0.03300), (3, 27, 2, 322, 0.06700), (3, 29, 2, 196, 0.04060),
           (3, 31, 2, 112, 0.03300), (3, 35, 2, 210, 0.01900), (4, 34, 2, 468, 0.04450),
           (4, 38, 2, 165, 0.01400)]
SOURCES = {"all racks": "all", "rack 3 only": "rack3"}
MODELS = ["A diffusion only", "E energy balance only", "B flow-based", "C flow-based + upward air"]


def zU(u):
    return Z_U20 + DZ_U * (np.asarray(u, dtype=float) - 20)


# ------------------------------------------------------------------ data loading
def _records(path):
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
    return us, mean, std, int(np.median([len(series[u]) for u in us]))


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


def crosscheck(us, Tm):
    try:
        arr = np.genfromtxt(f"{DATA}/Plots/previousDesign/exhaust_temp.csv", delimiter=",", skip_header=1)
        d = {int(a[0]): (a[2], a[3]) for a in arr}
        for col, name in enumerate(["'Previous Design (C)'", "'Retrofitted Design (C)'"]):
            diff = [d[u][col] - t for u, t in zip(us, Tm) if u in d]
            print(f"   my average vs exhaust_temp.csv column {name}: RMS difference = "
                  f"{np.sqrt(np.mean(np.square(diff))):.2f} C")
    except Exception as e:
        print(f"   (exhaust_temp.csv cross-check skipped: {e})")


# ------------------------------------------------------------------ source terms
def source_density(mode, z):
    """Heat input per metre of height [W/m] from the real server powers (model A)."""
    P = np.zeros_like(z)
    for rack, first, hgt, watts, flow in SERVERS:
        if mode == "rack3" and rack != 3:
            continue
        z0, z1 = float(zU(first)), float(zU(first + hgt))
        P[(z >= z0) & (z < z1)] += watts / (z1 - z0)
    return P


def fields(mode, z):
    """q(z) [K/s] and gamma(z) [1/s] from server power and air flow; occ = cells inside a server."""
    q, g = np.zeros_like(z), np.zeros_like(z)
    occ = np.zeros(z.shape, dtype=bool)
    for rack, first, hgt, watts, flow in SERVERS:
        if mode == "rack3" and rack != 3:
            continue
        z0, z1 = float(zU(first)), float(zU(first + hgt))
        vol = A_RACK * (z1 - z0)
        m = (z >= z0) & (z < z1)
        q[m] += watts / (RHO_CP * vol)
        g[m] += flow / vol
        occ |= m
    return q, g, occ


# ------------------------------------------------------------------ models
def solve_diffusion(ell, P, z):
    """Model A: -theta'' + theta/ell^2 = P with theta = 0 at the floor, zero flux at the ceiling.
    Symmetric half-cell form at the top, so A is SPD and the project's own CG applies."""
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


def thomas(a, b, c, d):
    """Own tridiagonal solver. a: lower, b: main, c: upper (a[0] and c[-1] unused)."""
    n = len(d)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_adv(kappa, v, q, g, occ, z):
    """Models B (v = 0) and C: -kappa theta'' + v theta' + gamma(z) theta = q(z).
    theta = 0 at the floor, zero gradient at the ceiling, first-order upwind for v >= 0."""
    h = z[1] - z[0]
    gam = np.where(occ, g, GBG)[1:]
    a = np.full(M, -kappa / h**2 - v / h)
    b = 2 * kappa / h**2 + v / h + gam
    c = np.full(M, -kappa / h**2)
    a[0] = 0.0
    a[-1] = -2 * kappa / h**2 - v / h            # ghost node above the ceiling
    c[-1] = 0.0
    return np.concatenate([[0.0], thomas(a, b, c, q[1:].copy())])


# ------------------------------------------------------------------ fitting
def at_sensors(prof, z, zs):
    return np.array([np.interp(zz + OFFS, z, prof).mean() for zz in zs])


def sensor_matrix(profiles, z, zs):
    return np.array([at_sensors(p, z, zs) for p in profiles])


def best_row(Phi, r, mask, scale):
    sub, rr = Phi[:, mask], r[mask]
    if scale:                                     # model A: amplitude s is linear, closed form
        s = np.maximum((sub @ rr) / (np.sum(sub * sub, axis=1) + 1e-300), 0.0)
    else:
        s = np.ones(len(Phi))
    cost = np.sum((rr[None, :] - s[:, None] * sub) ** 2, axis=1)
    k = int(np.argmin(cost))
    return k, float(s[k])


def fit_model(Phi, r, scale):
    """Best row, in-sample errors and leave-one-out RMSE (each held-out sensor is predicted from a refit)."""
    n = len(r)
    k, s = best_row(Phi, r, np.ones(n, dtype=bool), scale)
    err = s * Phi[k] - r
    loo = []
    for j in range(n):
        mask = np.arange(n) != j
        kj, sj = best_row(Phi, r, mask, scale)
        loo.append(sj * Phi[kj, j] - r[j])
    return {"k": k, "s": s, "pred": s * Phi[k],
            "rmse": float(np.sqrt(np.mean(err**2))), "max": float(np.abs(err).max()),
            "loo": float(np.sqrt(np.mean(np.square(loo))))}


# ------------------------------------------------------------------ main
def main():
    z = np.linspace(0.0, H_ROOM, M + 1)
    h = z[1] - z[0]
    combos = [(k, v) for k in KAPPAS for v in VS]            # combos[0] = (smallest kappa, v = 0)
    v0 = np.array([i for i, (k, v) in enumerate(combos) if v == 0.0])

    print("Source check (sum of P*h; listed total = %d W):" % sum(s[3] for s in SERVERS))
    prof = {}
    for sname, mode in SOURCES.items():
        P = source_density(mode, z)
        q, g, occ = fields(mode, z)
        print(f"   {sname}: {P.sum() * h:.0f} W")
        prof[sname] = {"A": np.array([solve_diffusion(e, P, z) for e in ELLS]),
                       "C": np.array([solve_adv(k, v, q, g, occ, z) for k, v in combos])}
    print()

    summary = []
    for design, (folder, nb_file, cool_file) in DESIGNS.items():
        print(f"=== {design} design ===")
        us, Tm, Ts, n = load_netbotz(f"{DATA}/{folder}/{nb_file}")
        cool = load_cooler(f"{DATA}/{folder}/{cool_file}")
        T_ref = pick(cool, "MinRackInletTempMetric")
        zs, r = zU(us), Tm - T_ref
        base = float(np.sqrt(np.mean((r - r.mean()) ** 2)))
        print(f"   {len(us)} sensors, {n} samples each; largest std over time = {Ts.max():.2f} C")
        print(f"   reference (min rack inlet) = {T_ref:.1f} C; baseline RMSE (predict the mean) = {base:.2f} C")
        crosscheck(us, Tm)
        print(f"   {'source set':12s} {'model':26s} {'fit':>3s} {'parameters':34s} {'RMSE':>6s} "
              f"{'max err':>8s} {'LOO RMSE':>9s}")

        rows = {int(u): [int(u), round(float(zu), 4), round(float(t), 3), round(float(s_), 3)]
                for u, zu, t, s_ in zip(us, zs, Tm, Ts)}
        header = ["U", "z_m", "measured_C", "measured_std_C"]
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.4))
        ax.errorbar(Tm, zs, xerr=Ts, fmt="ks", capsize=2, label="measured (mean ± std, 2 h)")

        for sname in SOURCES:
            PA, PC = prof[sname]["A"], prof[sname]["C"]
            Phi_A = sensor_matrix(PA, z, zs)
            Phi_C = sensor_matrix(PC, z, zs)
            fA = fit_model(Phi_A, r, True)
            fB = fit_model(Phi_C[v0], r, False)
            fC = fit_model(Phi_C, r, False)
            eE = Phi_C[0] - r
            kB = combos[v0[fB["k"]]][0]
            kC, vC = combos[fC["k"]]
            res = [
                (MODELS[0], 2, f"ell = {ELLS[fA['k']]:.2f} m, s = {fA['s']:.2e}", T_ref + fA["pred"],
                 fA["rmse"], fA["max"], fA["loo"], T_ref + fA["s"] * PA[fA["k"]]),
                (MODELS[1], 0, "none (power and flow only)", T_ref + Phi_C[0],
                 float(np.sqrt(np.mean(eE**2))), float(np.abs(eE).max()), float(np.sqrt(np.mean(eE**2))),
                 T_ref + PC[0]),
                (MODELS[2], 1, f"kappa = {kB:.2e} m2/s", T_ref + fB["pred"],
                 fB["rmse"], fB["max"], fB["loo"], T_ref + PC[v0[fB["k"]]]),
                (MODELS[3], 2, f"kappa = {kC:.2e} m2/s, v = {vC:.3f} m/s", T_ref + fC["pred"],
                 fC["rmse"], fC["max"], fC["loo"], T_ref + PC[fC["k"]]),
            ]
            for name, nfit, params, pred, rmse, mx, loo, curve in res:
                print(f"   {sname:12s} {name:26s} {nfit:3d} {params:34s} {rmse:6.2f} {mx:8.2f} {loo:9.2f}")
                summary.append([design, sname, name, nfit, params, round(rmse, 3), round(mx, 3),
                                round(loo, 3), round(base, 3)])
                for u, p in zip(us, pred):
                    rows[int(u)].append(round(float(p), 3))
                header.append(f"pred_C [{sname}|{name}]")
            if fA["k"] in (0, len(ELLS) - 1):
                print("   WARNING: model A: best ell is at the edge of the scan.")
            if kB in (KAPPAS[0], KAPPAS[-1]):
                print("   WARNING: model B: best kappa is at the edge of the scan.")
            if vC == VS[-1] or kC in (KAPPAS[0], KAPPAS[-1]):
                print("   WARNING: model C: best kappa or v is at the edge of the scan.")
            if vC == 0.0:
                print("   NOTE: model C chose v = 0, so the data gave no support for upward air movement.")

            if sname == "all racks":
                for (name, nfit, params, pred, rmse, mx, loo, curve), sty in zip(
                        res, ["--", ":", "-", "-"]):
                    ax.plot(curve, z, sty, lw=2, label=f"{name}")
                bx.bar([m_.split()[0] for m_ in MODELS], [x[6] for x in res],
                       color=["gray", "tab:orange", "tab:blue", "tab:green"])
                bx.axhline(base, color="k", ls="--", lw=1, label="predict the mean (baseline)")
        print()

        ax.set_xlabel("temperature at the rear of the rack [°C]")
        ax.set_ylabel("height [m]")
        ax.set_ylim(0, H_ROOM)
        ax.set_title(f"{design} design: measured vs the four models")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        bx.set_ylabel("leave-one-out RMSE [°C]")
        bx.set_title("Error on held-out sensors (lower is better)")
        bx.legend(fontsize=8)
        bx.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(f"results/calib_{design}.png", dpi=150)
        plt.close(fig)
        with open(f"results/calib_sensors_{design}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows[int(u)] for u in us)

    with open("results/calib_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "source_set", "model", "n_fitted", "parameters", "rmse_C",
                    "max_err_C", "loo_rmse_C", "baseline_rmse_C"])
        w.writerows(summary)
    for d, (pm, px) in PAPER_CFD_ERR.items():
        print(f"Published CFD errors, {d} design: mean {pm} C, max {px} C.")
    print("Only the ratio of mixing to cooling is identifiable from a steady profile; fitted values are "
          "effective 1D numbers and are NOT transferable to the plan-view model.")
    print("Done. Files: results/calib_*.png and results/calib_*.csv")


if __name__ == "__main__":
    main()