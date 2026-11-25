import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marconi_explore import read, CP_AIR

PAT = re.compile(r"r(\d+)c(\d+)s(\d+)")


def main():
    d = {k: read(k) for k in ("inlet", "outlet", "power")}
    cols = list(d["inlet"].columns)
    pos = pd.DataFrame([[int(x) for x in PAT.fullmatch(c).groups()] for c in cols],
                       columns=["rack", "chassis", "slot"], index=cols)
    for k in d:                                    # drop the first (raw) row
        d[k] = d[k].iloc[1:][cols]
    inlet, outlet, power = d["inlet"], d["outlet"], d["power"]

    def by_rack(df):                               # time x rack, mean over the 72 nodes
        return df.T.groupby(pos["rack"]).mean().T

    # ---------------- 1. does airflow change with inlet temperature?
    dT = (outlet - inlet).to_numpy().ravel()
    P = power.to_numpy().ravel()
    Ti = inlet.to_numpy().ravel()
    ok = np.isfinite(dT) & np.isfinite(P) & np.isfinite(Ti) & (P > 100)
    dT, P, Ti = dT[ok], P[ok], Ti[ok]
    ratio = 1000.0 * dT / P                        # K per kW
    print("=== 1. Air temperature rise per kW, by inlet temperature (nodes with P > 100 W) ===")
    print(f"   {'inlet bin [C]':>14s} {'points':>9s} {'median K/kW':>12s} {'implied flow [kg/s]':>20s}")
    edges = [15, 25, 30, 35, 40, 45, 55]
    bin_mid, bin_flow = [], []
    pred_bins = np.zeros_like(dT)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (Ti >= lo) & (Ti < hi)
        if m.sum() < 1000:
            continue
        med = float(np.median(ratio[m]))
        flow = 1000.0 / (CP_AIR * med)
        slope_b = (dT[m] * P[m]).sum() / (P[m] ** 2).sum()
        pred_bins[m] = slope_b * P[m]
        bin_mid.append(0.5 * (lo + hi))
        bin_flow.append(flow)
        print(f"   {lo:>6d} to {hi:<5d} {m.sum():9d} {med:12.2f} {flow:20.4f}")
    slope = (dT * P).sum() / (P * P).sum()
    seen = pred_bins != 0
    print(f"   correlation(inlet temperature, K/kW) = {np.corrcoef(Ti, ratio)[0, 1]:.3f}")
    print(f"   RMSE, one constant flow:        {np.sqrt(np.mean((dT - slope * P) ** 2)):.2f} K")
    print(f"   RMSE, flow per inlet-temp bin:  {np.sqrt(np.mean((dT[seen] - pred_bins[seen]) ** 2)):.2f} K")

    # ---------------- 2. rack-to-rack pattern
    Tr, Pr = by_rack(inlet), by_rack(power)
    rin, rpw = Tr.mean(), Pr.mean()
    racks = list(rin.index)
    print("\n=== 2. Rack means over the period ===")
    print(f"   {'rack':>5s} {'inlet [C]':>10s} {'power [W/node]':>15s}")
    for r in racks:
        print(f"   {r:5d} {rin[r]:10.2f} {rpw[r]:15.1f}")
    x = rin.values
    ac = np.corrcoef(x[:-1], x[1:])[0, 1]
    rng = np.random.default_rng(0)
    shuf = [np.corrcoef(p[:-1], p[1:])[0, 1] for p in (rng.permutation(x) for _ in range(1000))]
    print(f"   neighbouring rack IDs: autocorrelation of rack-mean inlet = {ac:.2f} "
          f"(shuffled 95th percentile = {np.percentile(shuf, 95):.2f})")
    print(f"   correlation across racks, mean power vs mean inlet = {np.corrcoef(rpw.values, x)[0, 1]:.2f}")

    # ---------------- 3. response in time
    dP = Pr.diff().iloc[1:]
    dTr = Tr.diff().iloc[1:]
    print("\n=== 3. Does a change in rack power show up as a change in rack inlet temperature? ===")
    print("   (correlation of 10-minute changes, power at t vs inlet at t + lag)")
    lags = [0, 1, 2, 3]
    lag_mean = []
    for lag in lags:
        cs = []
        for r in racks:
            a = dP[r].values[:len(dP) - lag]
            b = dTr[r].values[lag:]
            cs.append(np.corrcoef(a, b)[0, 1])
        cs = np.array(cs)
        lag_mean.append(np.nanmean(cs))
        print(f"   lag {lag * 10:2d} min: mean {np.nanmean(cs):6.3f}   median {np.nanmedian(cs):6.3f}")
    room = Tr.mean(axis=1)
    common = np.mean([np.corrcoef(Tr[r].values, room.values)[0, 1] for r in racks])
    print(f"   mean correlation of each rack's inlet series with the room-average series = {common:.2f}")

    # ---------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].bar([str(int(b)) for b in bin_mid], bin_flow, color="tab:blue")
    axes[0].set_xlabel("inlet temperature bin centre [°C]")
    axes[0].set_ylabel("implied air flow per node [kg/s]")
    axes[0].set_title("1. Airflow implied by energy balance")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].plot(racks, rin.values, "o-", color="tab:blue")
    axes[1].set_xlabel("rack ID")
    axes[1].set_ylabel("mean inlet temperature [°C]", color="tab:blue")
    ax2 = axes[1].twinx()
    ax2.plot(racks, rpw.values, "s--", color="tab:red", alpha=0.7)
    ax2.set_ylabel("mean power per node [W]", color="tab:red")
    axes[1].set_title("2. Rack-to-rack pattern")
    axes[2].bar([f"{l * 10} min" for l in lags], lag_mean, color="tab:green")
    axes[2].set_ylabel("mean correlation of changes")
    axes[2].set_title("3. Inlet response to power changes")
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("results/marconi_explore2.png", dpi=150)
    plt.close(fig)
    print("\nDone. File: results/marconi_explore2.png")


if __name__ == "__main__":
    main()