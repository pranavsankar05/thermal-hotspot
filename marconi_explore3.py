import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marconi_explore import read

PAT = re.compile(r"r(\d+)c(\d+)s(\d+)")
TRAIN_FRAC = 0.7     # first 70% of the time steps for fitting, last 30% for testing
NEIGH = 2            # neighbours = racks whose ID differs by 1 or 2 (an assumption)


def main():
    inlet = read("inlet")
    power = read("power")
    cols = list(inlet.columns)
    rack_of = np.array([int(PAT.fullmatch(c).group(1)) for c in cols])
    X = inlet.iloc[1:].to_numpy(dtype=float)
    PW = power.iloc[1:][cols].to_numpy(dtype=float)
    racks = np.unique(rack_of)
    nr = len(racks)
    R = np.column_stack([X[:, rack_of == r].mean(axis=1) for r in racks])     # time x racks
    P = np.column_stack([PW[:, rack_of == r].mean(axis=1) for r in racks])

    # ---------------- 1. where does the variance of node inlet temperature come from?
    tot = X.var()
    Xn = X - X.mean(axis=0)                                  # remove each node's own mean
    f_static = 1.0 - Xn.var() / tot
    common = Xn.mean(axis=1, keepdims=True)                  # room-wide signal at each time
    Xc = Xn - common
    f_common = (Xn.var() - Xc.var()) / tot
    Rr = np.column_stack([Xc[:, rack_of == r].mean(axis=1) for r in racks])
    f_rack = Rr.var() / tot
    f_rest = 1.0 - f_static - f_common - f_rack
    print("=== 1. Variance of node inlet temperature ===")
    print(f"   position (each node's own mean):          {f_static * 100:5.1f}%")
    print(f"   room-wide signal varying in time:         {f_common * 100:5.1f}%")
    print(f"   rack-specific part varying in time:       {f_rack * 100:5.1f}%")
    print(f"   node-specific part varying in time:       {f_rest * 100:5.1f}%")

    # ---------------- 2. does power help, once position and room signal are known?
    n = R.shape[0]
    ntr = int(TRAIN_FRAC * n)
    S = (R.sum(axis=1, keepdims=True) - R) / (nr - 1)        # room signal excluding the rack itself
    dP = P - P[:ntr].mean(axis=0)                            # power change relative to training mean
    nP = np.zeros_like(dP)
    for i, r in enumerate(racks):
        mask = (np.abs(racks - r) <= NEIGH) & (racks != r)
        nP[:, i] = dP[:, mask].mean(axis=1)

    def fit_predict(use_room, extras):
        k = (2 if use_room else 1) * nr + len(extras)
        A = np.zeros((nr * ntr, k))
        y = np.zeros(nr * ntr)
        for i in range(nr):
            sl = slice(i * ntr, (i + 1) * ntr)
            A[sl, i] = 1.0
            if use_room:
                A[sl, nr + i] = S[:ntr, i]
            for j, e in enumerate(extras):
                A[sl, k - len(extras) + j] = e[:ntr, i]
            y[sl] = R[:ntr, i]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = np.zeros((n - ntr, nr))
        for i in range(nr):
            pred[:, i] = coef[i]
            if use_room:
                pred[:, i] += coef[nr + i] * S[ntr:, i]
            for j, e in enumerate(extras):
                pred[:, i] += coef[k - len(extras) + j] * e[ntr:, i]
        return pred, coef[k - len(extras):]

    truth = R[ntr:]
    models = [("rack offset only (position)", False, []),
              ("+ room-wide signal", True, []),
              ("+ own rack power", True, [dP]),
              ("+ neighbour power", True, [nP]),
              ("+ own and neighbour power", True, [dP, nP])]
    print(f"\n=== 2. Held-out test on the last {n - ntr} time steps (fit on the first {ntr}) ===")
    print(f"   {'model':32s} {'test RMSE [K]':>14s} {'vs room-signal model':>22s}")
    preds, rm = {}, {}
    for name, use_room, ex in models:
        p, c = fit_predict(use_room, ex)
        preds[name] = p
        rm[name] = float(np.sqrt(np.mean((p - truth) ** 2)))
        if name == "+ own and neighbour power":
            coefs = c
    base = rm["+ room-wide signal"]
    for name, _, _ in models:
        print(f"   {name:32s} {rm[name]:14.3f} {(rm[name] / base - 1) * 100:+21.1f}%")
    persist = float(np.sqrt(np.mean((R[ntr:] - R[ntr - 1:-1]) ** 2)))
    print(f"   {'repeat the last value (context)':32s} {persist:14.3f} {(persist / base - 1) * 100:+21.1f}%")
    print(f"\n   fitted effect of +100 W per node, own rack: {coefs[0] * 100:+.3f} K; "
          f"neighbouring racks: {coefs[1] * 100:+.3f} K")

    # ---------------- 3. same test, changes only (removes the static position part)
    d_true = np.diff(truth, axis=0)
    d_base = np.diff(preds["+ room-wide signal"], axis=0)
    d_full = np.diff(preds["+ own and neighbour power"], axis=0)
    print("\n=== 3. Predicting 10-minute changes in rack inlet temperature (test period) ===")
    print(f"   RMSE of 'no change' guess:            {np.sqrt(np.mean(d_true ** 2)):.3f} K")
    print(f"   RMSE of room-signal model:            {np.sqrt(np.mean((d_base - d_true) ** 2)):.3f} K")
    print(f"   RMSE with own + neighbour power:      {np.sqrt(np.mean((d_full - d_true) ** 2)):.3f} K")

    # ---------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].bar(["position", "room-wide", "rack-specific", "node-specific"],
                [f_static * 100, f_common * 100, f_rack * 100, f_rest * 100], color="tab:blue")
    axes[0].set_ylabel("share of variance [%]")
    axes[0].set_title("1. Where the variation comes from")
    axes[0].grid(alpha=0.3, axis="y")
    names = [m[0] for m in models] + ["repeat last"]
    vals = [rm[m[0]] for m in models] + [persist]
    axes[1].barh(range(len(names)), vals, color=["gray", "tab:orange", "tab:blue", "tab:blue", "tab:green", "k"])
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("held-out RMSE [K]")
    axes[1].set_title("2. Does power help?")
    axes[1].grid(alpha=0.3, axis="x")
    i = int(np.argmax(P.std(axis=0)))
    last = min(1008, n - ntr)
    t = np.arange(last)
    axes[2].plot(t, truth[-last:, i], "k", lw=1.2, label="measured")
    axes[2].plot(t, preds["+ room-wide signal"][-last:, i], color="tab:orange", lw=1, label="room signal")
    axes[2].plot(t, preds["+ own and neighbour power"][-last:, i], color="tab:green", lw=1, label="+ power")
    axes[2].set_xlabel("time step (10 min)")
    axes[2].set_ylabel("rack-mean inlet [°C]")
    axes[2].set_title(f"3. Rack {racks[i]} (largest power variation)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/marconi_explore3.png", dpi=150)
    plt.close(fig)
    print("\nDone. File: results/marconi_explore3.png")


if __name__ == "__main__":
    main()