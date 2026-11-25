import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration import (DATA, DESIGNS, H_ROOM, M, KAPPAS, zU, load_netbotz, load_cooler,
                         pick, fields, solve_adv, sensor_matrix)

os.makedirs("results", exist_ok=True)
Z = np.linspace(0.0, H_ROOM, M + 1)
Q, G, OCC = fields("all", Z)
PROFS = [solve_adv(k, 0.0, Q, G, OCC, Z) for k in KAPPAS]      # model B: flow-based + mixing, v = 0


def load(design):
    folder, nb, cl = DESIGNS[design]
    us, Tm, Ts, n = load_netbotz(f"{DATA}/{folder}/{nb}")
    T_ref = pick(load_cooler(f"{DATA}/{folder}/{cl}"), "MinRackInletTempMetric")
    return dict(zip(us.tolist(), (Tm - T_ref).tolist()))        # excess over cooler inlet, by U


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def best_k(Phi, r, idx):
    return int(np.argmin(((Phi[:, idx] - r[idx]) ** 2).sum(axis=1)))


def loo(Phi, r):
    n = len(r)
    errs = []
    for j in range(n):
        idx = np.array([i for i in range(n) if i != j])
        errs.append(Phi[best_k(Phi, r, idx), j] - r[j])
    return rms(errs)


def main():
    data = {d: load(d) for d in DESIGNS}
    common = sorted(set.intersection(*[set(v) for v in data.values()]))
    us = np.array(common)
    r = {d: np.array([data[d][u] for u in common]) for d in DESIGNS}
    Phi = sensor_matrix(PROFS, Z, zU(us))                        # +/- half-U averaged, as in calibration.py
    n = len(us)
    all_idx = np.arange(n)
    rows = []
    print(f"{n} sensor heights are common to both designs.\n")

    # ---------------- 1. cross-design
    print("=== 1. Cross-design test (fit kappa on one design, predict the other) ===")
    print(f"   {'fit on':12s} {'predict':12s} {'kappa':>9s} {'model RMSE':>11s} {'copy-other RMSE':>16s} {'baseline':>9s}")
    cross = []
    for a in DESIGNS:
        for b in DESIGNS:
            if a == b:
                continue
            k = best_k(Phi, r[a], all_idx)
            e_model = rms(Phi[k] - r[b])
            e_copy = rms(r[a] - r[b])
            base = rms(r[b] - r[b].mean())
            print(f"   {a:12s} {b:12s} {KAPPAS[k]:9.2e} {e_model:11.2f} {e_copy:16.2f} {base:9.2f}")
            cross.append((a, b, e_model, e_copy, base))
            rows.append(["cross_design", f"fit {a} -> predict {b}", round(e_model, 3), round(e_copy, 3), round(base, 3)])

    # ---------------- 2. sensor-height alignment
    print("\n=== 2. Sensor-height alignment (shift all sensors by dU rows, refit, leave-one-out) ===")
    shifts = np.arange(-2.0, 2.01, 0.5)
    shift_res = {d: [] for d in DESIGNS}
    print(f"   {'dU':>5s} " + " ".join(f"{d + ' LOO':>18s}" for d in DESIGNS))
    for dU in shifts:
        PhiS = sensor_matrix(PROFS, Z, zU(us + dU))
        vals = [loo(PhiS, r[d]) for d in DESIGNS]
        for d, v in zip(DESIGNS, vals):
            shift_res[d].append(v)
        print(f"   {dU:5.1f} " + " ".join(f"{v:18.2f}" for v in vals))
        rows.append(["alignment", f"dU = {dU}"] + [round(v, 3) for v in vals])
    for d in DESIGNS:
        print(f"   {d}: best shift = {shifts[int(np.argmin(shift_res[d]))]:.1f} U "
              f"(LOO {min(shift_res[d]):.2f} C; at dU = 0: {shift_res[d][int(np.where(shifts == 0)[0][0])]:.2f} C)")

    # ---------------- 3. bootstrap
    print("\n=== 3. Bootstrap of kappa (500 resamples of the sensors, seed 0) ===")
    rng = np.random.default_rng(0)
    boot = {}
    for d in DESIGNS:
        ks = []
        for _ in range(500):
            idx = rng.integers(0, n, n)
            ks.append(KAPPAS[best_k(Phi, r[d], idx)])
        lo, med, hi = np.percentile(ks, [5, 50, 95])
        boot[d] = ks
        print(f"   {d:12s} kappa median = {med:.2e}, 5-95% range = {lo:.2e} to {hi:.2e} m2/s")
        rows.append(["bootstrap", d, round(float(lo), 8), round(float(med), 8), round(float(hi), 8)])

    with open("results/calib_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "case", "value_1", "value_2", "value_3"])
        w.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    lab = [f"{a}\n->{b}" for a, b, *_ in cross]
    x = np.arange(len(cross))
    axes[0].bar(x - 0.27, [c[2] for c in cross], 0.27, label="model (kappa from the other design)")
    axes[0].bar(x, [c[3] for c in cross], 0.27, label="copy the other design's profile")
    axes[0].bar(x + 0.27, [c[4] for c in cross], 0.27, label="predict the mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lab, fontsize=8)
    axes[0].set_ylabel("RMSE [°C]")
    axes[0].set_title("1. Cross-design prediction")
    axes[0].legend(fontsize=7)
    for d in DESIGNS:
        axes[1].plot(shifts, shift_res[d], "o-", label=d)
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("sensor height shift [U rows]")
    axes[1].set_ylabel("leave-one-out RMSE [°C]")
    axes[1].set_title("2. Sensitivity to sensor alignment")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    for d in DESIGNS:
        axes[2].hist(np.log10(boot[d]), bins=np.log10(KAPPAS), alpha=0.5, label=d)
    axes[2].set_xlabel("log10 kappa  [m²/s]")
    axes[2].set_ylabel("bootstrap count")
    axes[2].set_title("3. Bootstrap of the fitted mixing")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/calib_checks.png", dpi=150)
    plt.close(fig)
    print("\nDone. Files: results/calib_checks.png and results/calib_checks.csv")


if __name__ == "__main__":
    main()