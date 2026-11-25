import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from solver import L, T_AMB, ALPHA, THRESHOLD, RACKS, make_q, make_gamma, steady_solve

st.set_page_config(page_title="Optimizer", layout="wide")

N = 50
NR = len(RACKS)
MIXED = [0.9, 0.9, 0.5, 0.5, 0.2, 0.2]      # baseline workloads used in the experiments
MARGIN = 1e-6                                # keeps LP answers strictly below the limit


@st.cache_resource(show_spinner="Computing the response of each rack (one-time)...")
def load_maps():
    """Steady excess-temperature field of each rack alone at workload 1.0 (own CG solver)."""
    gamma = make_gamma(N)
    maps = []
    for k in range(NR):
        w = [0.0] * NR
        w[k] = 1.0
        th, _ = steady_solve(N, ALPHA, gamma, make_q(N, RACKS, w))
        maps.append(th)
    phi = np.column_stack([m[1:-1, 1:-1].ravel() for m in maps])
    return maps, phi


MAPS, PHI = load_maps()


def field(w):
    return sum(wk * m for wk, m in zip(w, MAPS))


def stats(w, limit):
    T = T_AMB + field(w)
    return float(T.max()), int((T > limit + 1e-9).sum())


def min_peak(total, cap):
    """Same total workload, lowest possible peak temperature."""
    c = np.r_[np.zeros(NR), 1.0]
    A_ub = np.c_[PHI, -np.ones(PHI.shape[0])]
    A_eq = np.r_[np.ones(NR), 0.0][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(PHI.shape[0]), A_eq=A_eq, b_eq=[total],
                  bounds=[(0, cap)] * NR + [(None, None)], method="highs")
    return res.x[:NR] if res.status == 0 else None


def max_work(limit, cap):
    """Largest total workload with every node at or below the limit."""
    lim = limit - T_AMB - MARGIN
    res = linprog(-np.ones(NR), A_ub=PHI, b_ub=np.full(PHI.shape[0], lim),
                  bounds=[(0, cap)] * NR, method="highs")
    return res.x if res.status == 0 else None


@st.cache_data
def capacity_curve(cap):
    limits = np.arange(30.0, 41.0, 1.0)
    return limits, [max_work(t, cap).sum() for t in limits]


st.title("Optimizer: where should the load go?")
st.write("The room has six racks. Each rack heats the room more or less depending on where it "
         "stands. Because the heat equation is linear, the temperature map of any workload is the "
         "sum of each rack's own map. That turns 'where to put the load' into a linear program "
         "(LP) that is solved exactly. Move the sliders and see what the best placement does.")

c0, c1 = st.columns(2)
limit = c0.slider("Temperature limit [°C]", 30.0, 40.0, float(THRESHOLD), 0.5)
cap = c1.slider("Maximum load per rack (1.0 = full capacity)", 0.5, 1.5, 1.0, 0.05)

with st.expander("Baseline workload of each rack (default: the mixed load from the experiments)"):
    cols = st.columns(NR)
    base = np.array([cols[k].slider(f"R{k + 1}", 0.0, 1.5, float(MIXED[k]), 0.05, key=f"b{k}")
                     for k in range(NR)])

total = float(base.sum())
total_lp = min(total, NR * cap)
if total > NR * cap:
    st.warning(f"The baseline total ({total:.2f}) is above what the racks can hold at this capacity "
               f"({NR * cap:.2f}). The 'same total' case uses {total_lp:.2f} instead.")

w_a = min_peak(total_lp, cap)
w_b = max_work(limit, cap)
if w_a is None or w_b is None:
    st.error("The LP has no solution for these settings. Try a higher limit or a larger rack capacity.")
    st.stop()

cases = [("Baseline", base), ("LP: same total, lowest peak", w_a), ("LP: maximum total, no hotspot", w_b)]

rows = []
for name, w in cases:
    peak, hot = stats(w, limit)
    rows.append({"case": name, "total load": round(float(w.sum()), 3), "peak [°C]": round(peak, 2),
                 f"nodes above {limit:g} °C": hot, **{f"R{k + 1}": round(float(x), 2) for k, x in enumerate(w)}})
st.dataframe(pd.DataFrame(rows), hide_index=True)

peak0, _ = stats(base, limit)
peak_a, _ = stats(w_a, limit)
m0, m1 = st.columns(2)
m0.metric("Same load, better placement: peak temperature [°C]", f"{peak_a:.2f}",
          f"{peak_a - peak0:+.2f} vs baseline", delta_color="inverse")
if total > 0:
    m1.metric("Load the room can run with no hotspot", f"{w_b.sum():.2f}",
              f"{(w_b.sum() / total - 1) * 100:+.0f}% vs baseline total {total:.2f}")

vmax = max(T_AMB + field(w).max() for _, w in cases)
xs = np.linspace(0, L, N + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (name, w) in zip(axes, cases):
    T = T_AMB + field(w)
    im = ax.imshow(T.T, origin="lower", extent=[0, L, 0, L], vmin=T_AMB, vmax=vmax, cmap="inferno")
    if T.max() > limit:
        ax.contour(xs, xs, T.T, levels=[limit], colors="cyan", linewidths=1.2)
    for k, (cx, cy) in enumerate(RACKS):
        ax.plot(cx, cy, "ws", ms=5)
        ax.text(cx + 0.25, cy + 0.25, f"R{k + 1}\n{w[k]:.2f}", color="white", fontsize=7)
    ax.set_title(f"{name}\ntotal {w.sum():.2f}, peak {T.max():.2f} °C")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, label="temperature [°C]")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

limits, caps = capacity_curve(cap)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.arange(NR)
axes2[0].bar(x - 0.27, base, 0.27, label="baseline")
axes2[0].bar(x, w_a, 0.27, label="LP lowest peak (same total)")
axes2[0].bar(x + 0.27, w_b, 0.27, label="LP maximum total")
axes2[0].set_xticks(x)
axes2[0].set_xticklabels([f"R{k + 1}" for k in range(NR)])
axes2[0].set_ylabel("workload (fraction of rack capacity)")
axes2[0].set_title("Where the load goes")
axes2[0].legend(fontsize=8)
axes2[0].grid(alpha=0.3, axis="y")
axes2[1].plot(limits, caps, "o-")
axes2[1].axhline(total, color="k", ls="--", lw=1, label="baseline total")
axes2[1].axvline(limit, color="r", ls=":", lw=1, label="chosen limit")
axes2[1].set_xlabel("allowed maximum temperature [°C]")
axes2[1].set_ylabel("maximum total workload")
axes2[1].set_title("Capacity versus temperature limit")
axes2[1].legend(fontsize=8)
axes2[1].grid(alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

with st.expander("How this works and what it assumes"):
    st.markdown(
        "- Each rack's steady temperature map is computed once with the project's own CG solver. "
        "Any workload map is the weighted sum of these six maps (superposition).\n"
        "- **Same total, lowest peak:** minimise the hottest node while keeping the total load fixed.\n"
        "- **Maximum total:** maximise the total load while no node exceeds the limit.\n"
        "- Both are linear programs, solved exactly with `scipy.optimize.linprog`.\n"
        "- Parameters are illustrative. The result assumes load can be moved freely between racks."
    )