import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from solver import L, T_AMB, ALPHA, THRESHOLD, RACKS, make_q, make_gamma, steady_solve

N = 50
MIXED = [0.9, 0.9, 0.5, 0.5, 0.2, 0.2]      # baseline workloads used in the earlier experiments
W_MAX = 1.0                                  # a rack cannot run above 100% load
MARGIN = 1e-6                                # keeps LP answers strictly below the 35 C threshold
NR = len(RACKS)
GAMMA = make_gamma(N)

os.makedirs("results", exist_ok=True)


def response_maps():
    """Steady excess-temperature field of each rack alone at workload 1.0 (own CG solver)."""
    maps = []
    for k in range(NR):
        w = [0.0] * NR
        w[k] = 1.0
        th, _ = steady_solve(N, ALPHA, GAMMA, make_q(N, RACKS, w))
        maps.append(th)
    return maps


MAPS = response_maps()
PHI = np.column_stack([m[1:-1, 1:-1].ravel() for m in MAPS])      # (interior nodes) x racks


def field(w):
    return sum(wk * m for wk, m in zip(w, MAPS))


def stats(w):
    T = T_AMB + field(w)
    return float(T.max()), int((T > THRESHOLD + 1e-9).sum())


def min_peak(total):
    """Same total workload, lowest possible peak temperature."""
    c = np.r_[np.zeros(NR), 1.0]
    A_ub = np.c_[PHI, -np.ones(PHI.shape[0])]
    A_eq = np.r_[np.ones(NR), 0.0][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(PHI.shape[0]), A_eq=A_eq, b_eq=[total],
                  bounds=[(0, W_MAX)] * NR + [(None, None)], method="highs")
    assert res.status == 0, res.message
    return res.x[:NR]


def max_work(limit_c):
    """Largest total workload with every node at or below limit_c."""
    lim = limit_c - T_AMB - MARGIN
    res = linprog(-np.ones(NR), A_ub=PHI, b_ub=np.full(PHI.shape[0], lim),
                  bounds=[(0, W_MAX)] * NR, method="highs")
    assert res.status == 0, res.message
    return res.x


def main():
    mixed = np.array(MIXED)
    total = mixed.sum()
    uniform = np.full(NR, total / NR)
    w_a = min_peak(total)
    w_b = max_work(THRESHOLD)

    # ---------- check 1: superposition reproduces a full solve
    th_full, _ = steady_solve(N, ALPHA, GAMMA, make_q(N, RACKS, MIXED))
    print(f"Check 1: sum of response maps vs full solve of the mixed load, max difference = "
          f"{np.abs(field(mixed) - th_full).max():.2e} K")

    cases = [("baseline (mixed load)", mixed), ("same total, spread evenly", uniform),
             ("same total, LP minimum peak", w_a), ("LP maximum total, no hotspot", w_b)]
    print(f"\nGrid {N} x {N}, threshold {THRESHOLD:.0f} C, rack capacity {W_MAX:.1f}, "
          f"total capacity {NR * W_MAX:.1f}")
    print(f"{'case':32s} {'total':>6s} {'peak [C]':>9s} {'hot nodes':>10s}   workloads per rack")
    rows = []
    for name, w in cases:
        peak, hot = stats(w)
        print(f"{name:32s} {w.sum():6.3f} {peak:9.3f} {hot:10d}   "
              + " ".join(f"{x:.2f}" for x in w))
        rows.append([name, round(float(w.sum()), 4), round(peak, 4), hot] + [round(float(x), 4) for x in w])

    gain = w_b.sum() / total - 1
    print(f"\nHeadline: the room can run {w_b.sum():.2f} units of load with no hotspot instead of {total:.2f} "
          f"({gain * 100:+.0f}%), by moving load between racks.")
    print(f"Same {total:.1f} units, better placement: peak {stats(mixed)[0]:.2f} C -> {stats(w_a)[0]:.2f} C.")

    # ---------- check 2: full solver agrees with the LP answers
    for name, w in cases[2:]:
        th, _ = steady_solve(N, ALPHA, GAMMA, make_q(N, RACKS, list(w)))
        print(f"Check 2 ({name}): full solver peak {T_AMB + th.max():.3f} C, "
              f"LP peak {stats(w)[0]:.3f} C")

    # ---------- how total capacity changes with the temperature limit
    limits = np.arange(30.0, 41.0, 1.0)
    cap = [max_work(t).sum() for t in limits]
    print("\nMax total workload (no node above the limit):")
    for t, c in zip(limits, cap):
        print(f"   limit {t:4.0f} C: {c:5.2f} of {NR * W_MAX:.1f}")

    with open("results/lp_results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case", "total_workload", "peak_C", "hot_nodes"] + [f"rack{k + 1}" for k in range(NR)])
        wr.writerows(rows)
    with open("results/lp_capacity_curve.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["limit_C", "max_total_workload"])
        wr.writerows([[float(t), round(float(c), 4)] for t, c in zip(limits, cap)])

    # ---------- figures
    show = [("baseline (mixed load)", mixed), ("LP: same total, min peak", w_a), ("LP: max total, no hotspot", w_b)]
    vmax = max(T_AMB + field(w).max() for _, w in show)
    xs = np.linspace(0, L, N + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (name, w) in zip(axes, show):
        T = T_AMB + field(w)
        im = ax.imshow(T.T, origin="lower", extent=[0, L, 0, L], vmin=T_AMB, vmax=vmax, cmap="inferno")
        if T.max() > THRESHOLD:
            ax.contour(xs, xs, T.T, levels=[THRESHOLD], colors="cyan", linewidths=1.2)
        for k, (cx, cy) in enumerate(RACKS):
            ax.plot(cx, cy, "ws", ms=5)
            ax.text(cx + 0.25, cy + 0.25, f"R{k + 1}\n{w[k]:.2f}", color="white", fontsize=7)
        ax.set_title(f"{name}\ntotal {w.sum():.2f}, peak {T.max():.2f} °C")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, label="temperature [°C]")
    fig.tight_layout()
    fig.savefig("results/lp_maps.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(NR)
    axes[0].bar(x - 0.27, mixed, 0.27, label="baseline")
    axes[0].bar(x, w_a, 0.27, label="LP min peak (same total)")
    axes[0].bar(x + 0.27, w_b, 0.27, label="LP max total")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"R{k + 1}" for k in range(NR)])
    axes[0].set_ylabel("workload (fraction of rack capacity)")
    axes[0].set_title("Where the load goes")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].plot(limits, cap, "o-")
    axes[1].axhline(total, color="k", ls="--", lw=1, label="baseline total")
    axes[1].axvline(THRESHOLD, color="r", ls=":", lw=1, label="35 °C limit")
    axes[1].set_xlabel("allowed maximum temperature [°C]")
    axes[1].set_ylabel("maximum total workload")
    axes[1].set_title("Capacity versus temperature limit")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/lp_summary.png", dpi=150)
    plt.close(fig)
    print("\nDone. Files: results/lp_maps.png, lp_summary.png, lp_results.csv, lp_capacity_curve.csv")


if __name__ == "__main__":
    main()
    