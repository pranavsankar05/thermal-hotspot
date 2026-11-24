import csv
import os

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import cg
from calibration import DATA, DESIGNS, H_ROOM, M, DZ_U, zU, load_netbotz, load_cooler, pick

os.makedirs("results", exist_ok=True)

RHO_CP = 1206.0            # volumetric heat capacity of air [J/(m^3 K)]
A_RACK = 0.6               # assumed rack cross-section [m^2]; it cancels in q/gamma
# Table 1, Kuzay et al. (2022): (rack, first U, height U, power [W], air flow [m^3/s])
SERVERS = [(1, 38, 1, 84, 0.00740), (2, 27, 2, 144, 0.02390), (2, 29, 2, 144, 0.02390),
           (3, 25, 2, 154, 0.03300), (3, 27, 2, 322, 0.06700), (3, 29, 2, 196, 0.04060),
           (3, 31, 2, 112, 0.03300), (3, 35, 2, 210, 0.01900), (4, 34, 2, 468, 0.04450),
           (4, 38, 2, 165, 0.01400)]
MODES = {"all racks": "all", "rack 3 only": "rack3"}
KAPPAS = np.logspace(-6, -1, 26)          # vertical mixing diffusivity [m^2/s]
GBGS = [0.001, 0.003, 0.01, 0.05, 0.2, 1.0]            # cooling rate between servers [1/s]
OFFS = np.linspace(-DZ_U / 2, DZ_U / 2, 9)


def fields(mode, z):
    """q(z) [K/s], gamma(z) [1/s] from server power and air flow; occ = cells inside a server."""
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


def solve(kappa, gbg, q, g, occ, z):
    """Steady  -kappa theta'' + gamma(z) theta = q(z), theta = 0 at floor and ceiling. Own CG."""
    h = z[1] - z[0]
    gam = np.where(occ, g, gbg)
    n = len(z) - 2
    main = 2 * kappa / h**2 + gam[1:-1]
    off = np.full(n - 1, -kappa / h**2)
    A = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    th, _ = cg(A, q[1:-1].copy(), tol=1e-12, maxit=20000)
    return np.concatenate([[0.0], th, [0.0]])


def at_sensors(prof, z, zs):
    return np.array([np.interp(zz + OFFS, z, prof).mean() for zz in zs])


def main():
    z = np.linspace(0.0, H_ROOM, M + 1)
    print("Energy-balance rise per server, P / (rho c_p V):")
    for rack, first, hgt, watts, flow in SERVERS:
        print(f"   rack {rack}, {first}U: {watts / (RHO_CP * flow):5.1f} K")
    print()

    combos = [(k, gb) for k in KAPPAS for gb in GBGS]
    summary = []
    for design, (folder, nb_file, cool_file) in DESIGNS.items():
        print(f"=== {design} design ===")
        us, Tm, Ts, n = load_netbotz(f"{DATA}/{folder}/{nb_file}")
        cool = load_cooler(f"{DATA}/{folder}/{cool_file}")
        T_ref = pick(cool, "MinRackInletTempMetric")
        zs, r = zU(us), Tm - T_ref
        base = float(np.sqrt(np.mean((r - r.mean()) ** 2)))
        print(f"   reference (min rack inlet) = {T_ref:.1f} C; baseline RMSE (mean) = {base:.2f} C")
        print(f"   {'source set':14s} {'kappa':>9s} {'gamma_bg':>9s} {'RMSE':>6s} {'max err':>8s} "
              f"{'LOO RMSE':>9s} {'no-mixing RMSE':>15s}")

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.errorbar(Tm, zs, xerr=Ts, fmt="ks", capsize=2, label="measured (mean ± std, 2 h)")
        rows = {int(u): [int(u), round(float(zu), 4), round(float(t), 3)] for u, zu, t in zip(us, zs, Tm)}
        header = ["U", "z_m", "measured_C"]
        for (name, mode), colour in zip(MODES.items(), ["tab:blue", "tab:green"]):
            q, g, occ = fields(mode, z)
            Phi = np.array([at_sensors(solve(k, gb, q, g, occ, z), z, zs) for k, gb in combos])
            cost = ((Phi - r) ** 2).sum(axis=1)
            b = int(np.argmin(cost))
            err = Phi[b] - r
            loo = []
            for j in range(len(r)):
                mask = np.arange(len(r)) != j
                kb = int(np.argmin(((Phi[:, mask] - r[mask]) ** 2).sum(axis=1)))
                loo.append(Phi[kb, j] - r[j])
            rmse = float(np.sqrt(np.mean(err**2)))
            rmse_loo = float(np.sqrt(np.mean(np.square(loo))))
            e0 = Phi[0] - r                                  # smallest kappa: no mixing between streams
            rmse0 = float(np.sqrt(np.mean(e0**2)))
            print(f"   {name:14s} {combos[b][0]:9.2e} {combos[b][1]:9.2f} {rmse:6.2f} "
                  f"{np.abs(err).max():8.2f} {rmse_loo:9.2f} {rmse0:15.2f}")
            if b < len(GBGS) or b >= len(combos) - len(GBGS):
                print("   WARNING: best kappa is at the edge of the scan.")
            ax.plot(T_ref + solve(combos[b][0], combos[b][1], q, g, occ, z), z, color=colour, lw=2,
                    label=f"model: {name} (kappa = {combos[b][0]:.1e})")
            for u, p, e in zip(us, T_ref + Phi[b], err):
                rows[int(u)] += [round(float(p), 3), round(float(e), 3)]
            header += [f"model_C [{name}]", f"error_C [{name}]"]
            summary.append([design, name, combos[b][0], combos[b][1], round(rmse, 3),
                            round(float(np.abs(err).max()), 3), round(rmse_loo, 3),
                            round(rmse0, 3), round(base, 3)])
        ax.set_xlabel("temperature at the rear of the rack [°C]")
        ax.set_ylabel("height [m]")
        ax.set_ylim(0, H_ROOM)
        ax.set_title(f"{design} design: flow-based model vs measured")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"results/calib2_{design}.png", dpi=150)
        plt.close(fig)
        with open(f"results/calib2_sensors_{design}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows[int(u)] for u in us)
        print()

    with open("results/calib2_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "source_set", "kappa_m2_s", "gamma_bg_1_s", "rmse_C", "max_err_C",
                    "loo_rmse_C", "no_mixing_rmse_C", "baseline_rmse_C"])
        w.writerows(summary)
    print("Done. Files: results/calib2_*.png and results/calib2_*.csv")


if __name__ == "__main__":
    main()