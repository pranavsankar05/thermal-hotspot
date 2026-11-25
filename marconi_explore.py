import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "data/marconi"
FILES = {"inlet": "Marconi_A2_Inlet_Air_Temp.csv.gz",
         "outlet": "Marconi_A2_Outlet_Air_Temp.csv.gz",
         "power": "Marconi_A2_Power_Consumption.csv.gz"}
NROWS = 4000          # about four weeks at 10-minute spacing
CP_AIR = 1005.0       # J/(kg K)
RHO_AIR = 1.15        # kg/m^3

os.makedirs("results", exist_ok=True)


def read(name):
    path = os.path.join(DIR, FILES[name])
    cols = pd.read_csv(path, nrows=0).columns
    dtypes = {c: np.float32 for c in cols[1:]}
    return pd.read_csv(path, index_col=0, parse_dates=True, nrows=NROWS, dtype=dtypes)


def main():
    d = {k: read(k) for k in FILES}
    for k, df in d.items():
        print(f"{k:7s} columns: {df.shape[1]}  rows read: {df.shape[0]}  "
              f"from {df.index[0]} to {df.index[-1]}")
    common = [c for c in d["inlet"].columns
              if c in d["outlet"].columns and c in d["power"].columns]
    print(f"\nColumns common to all three files: {len(common)}")

    pat = re.compile(r"r(\d+)c(\d+)s(\d+)")
    bad = [c for c in common if not pat.fullmatch(c)]
    if bad:
        print("Names that do not match rXXXcYYsZZ:", bad[:10])
        return
    pos = pd.DataFrame([[int(x) for x in pat.fullmatch(c).groups()] for c in common],
                       columns=["rack", "chassis", "slot"], index=common)
    per_rack = pos.groupby("rack").size()
    print(f"Racks: {pos['rack'].nunique()} (ids {pos['rack'].min()} to {pos['rack'].max()})")
    print(f"Chassis index: {pos['chassis'].min()} to {pos['chassis'].max()}   "
          f"slot index: {pos['slot'].min()} to {pos['slot'].max()}")
    print(f"Nodes per rack: min {per_rack.min()}, max {per_rack.max()}")

    for k in d:                                   # drop the first (raw) row
        d[k] = d[k].iloc[1:][common]

    step = d["inlet"].index.to_series().diff().dropna()
    top = step.value_counts().index[0]
    print(f"\nMost common time step: {top} ({(step == top).mean() * 100:.1f}% of steps)")
    for k, df in d.items():
        v = df.to_numpy()
        print(f"{k:7s} missing {np.isnan(v).mean() * 100:5.2f}%   min {np.nanmin(v):7.2f}   "
              f"median {np.nanmedian(v):7.2f}   max {np.nanmax(v):7.2f}")

    # ---------- energy balance: dT = P / (m_dot cp)
    dT = (d["outlet"] - d["inlet"]).to_numpy().ravel()
    P = d["power"].to_numpy().ravel()
    ok = np.isfinite(dT) & np.isfinite(P) & (P > 100)
    dT, P = dT[ok], P[ok]
    slope = (dT * P).sum() / (P * P).sum()        # least squares through the origin
    flow = 1.0 / (CP_AIR * slope)
    print("\n=== Energy balance (nodes with P > 100 W) ===")
    print(f"points: {len(P)}   slope dT/P = {slope:.4f} K/W   "
          f"implied air flow = {flow:.4f} kg/s = {flow / RHO_AIR:.4f} m3/s per node")
    print(f"correlation(P, dT) = {np.corrcoef(P, dT)[0, 1]:.3f}   "
          f"RMSE of dT - slope*P = {np.sqrt(np.mean((dT - slope * P) ** 2)):.2f} K")

    # ---------- vertical profile
    node_mean = d["inlet"].mean()
    by_chassis = node_mean.groupby(pos["chassis"]).agg(["mean", "std"])
    print("\n=== Mean inlet temperature by chassis index (over all racks and the period) ===")
    print(by_chassis.round(2).to_string())
    rack_mean = node_mean.groupby(pos["rack"]).mean()
    print(f"\nRack-mean inlet: coolest rack {rack_mean.idxmin()} ({rack_mean.min():.2f} C), "
          f"hottest rack {rack_mean.idxmax()} ({rack_mean.max():.2f} C)")

    # ---------- figure
    g = node_mean.groupby([pos["rack"], pos["chassis"]]).mean().unstack()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for _, row in g.iterrows():
        axes[0].plot(g.columns, row.values, color="tab:blue", alpha=0.25, lw=0.8)
    axes[0].plot(g.columns, g.mean().values, "k-", lw=2, label="mean over racks")
    axes[0].set_xlabel("chassis index")
    axes[0].set_ylabel("mean inlet temperature [°C]")
    axes[0].set_title("1. Vertical profile, one line per rack")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    rng = np.random.default_rng(0)
    sel = rng.choice(len(P), size=min(200000, len(P)), replace=False)
    axes[1].hexbin(P[sel], dT[sel], gridsize=60, bins="log", cmap="viridis")
    pp = np.array([P.min(), P.max()])
    axes[1].plot(pp, slope * pp, "r-", lw=2, label=f"dT = {slope:.4f} P")
    axes[1].set_xlabel("node power [W]")
    axes[1].set_ylabel("outlet minus inlet [K]")
    axes[1].set_title("2. Energy balance")
    axes[1].legend(fontsize=8)

    first = pos.index[pos["rack"] == pos["rack"].min()]
    t = d["power"].index[:432]
    axes[2].plot(t, d["power"][first].mean(axis=1).iloc[:432].values, color="tab:red")
    axes[2].set_ylabel("rack-mean power [W]", color="tab:red")
    ax2 = axes[2].twinx()
    ax2.plot(t, d["inlet"][first].mean(axis=1).iloc[:432].values, color="tab:blue")
    ax2.set_ylabel("rack-mean inlet [°C]", color="tab:blue")
    axes[2].set_title(f"3. Rack {pos['rack'].min()}, first 3 days")
    axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig("results/marconi_explore.png", dpi=150)
    plt.close(fig)
    print("\nDone. File: results/marconi_explore.png")


if __name__ == "__main__":
    main()