import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Real data", layout="wide")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def read(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


st.title("Real data: does the model match measurements?")
st.write("Dataset: Kuzay et al. (2022), a small air-cooled data centre in Istanbul with a 2 kW "
         "load. Temperatures were measured at 12 heights on the rear door of one rack, every "
         "10 s for 2 hours. The data is nearly constant in time, so we compare a steady "
         "vertical profile.")

summ = read("calib2_summary.csv")
if summ is None:
    st.warning("results/calib2_summary.csv not found. Run `python calibration2.py` first.")
    st.stop()

design = st.radio("Design", ["previous", "retrofitted"], horizontal=True,
                  format_func=lambda s: "Previous (door open)" if s == "previous"
                  else "Retrofitted (aisles isolated, door closed)")
sens = read(f"calib2_sensors_{design}.csv")
std = read(f"calib_sensors_{design}.csv")
if sens is None:
    st.warning(f"results/calib2_sensors_{design}.csv not found.")
    st.stop()

sets = [c.split("[")[1].rstrip("]") for c in sens.columns if c.startswith("model_C")]
choice = st.radio("Which racks heat the air at the sensors?", sets, horizontal=True)
row = summ[(summ["design"] == design) & (summ["source_set"] == choice)].iloc[0]

fig, ax = plt.subplots(figsize=(6.5, 5.5))
if std is not None and "measured_std_C" in std.columns:
    ax.errorbar(sens["measured_C"], sens["z_m"], xerr=std["measured_std_C"].values, fmt="ks",
                capsize=2, label="measured (mean ± std over 2 h)")
else:
    ax.plot(sens["measured_C"], sens["z_m"], "ks", label="measured")
order = np.argsort(sens["z_m"].values)
ax.plot(sens[f"model_C [{choice}]"].values[order], sens["z_m"].values[order], "o-", color="tab:blue",
        label="model (power and air flow of each server)")
ax.set_xlabel("temperature at the rear of the rack [°C]")
ax.set_ylabel("height [m]")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()

c1, c2 = st.columns([1.3, 1])
c1.pyplot(fig)
plt.close(fig)
with c2:
    st.metric("Model error (RMSE) [°C]", f"{row['rmse_C']:.2f}")
    st.metric("On held-out sensors (leave-one-out) [°C]", f"{row['loo_rmse_C']:.2f}")
    st.metric("Guessing the average [°C]", f"{row['baseline_rmse_C']:.2f}")
    st.metric("Largest error at one sensor [°C]", f"{row['max_err_C']:.2f}")
    st.metric("Energy balance only, almost nothing fitted [°C]", f"{row['no_mixing_rmse_C']:.2f}")
    gain = 100 * (1 - row["loo_rmse_C"] / row["baseline_rmse_C"])
    st.caption(f"On sensors left out of the fit, the model's error is {gain:.0f}% below "
               f"guessing the average. The published CFD model reaches 0.5 to 1.6 °C, "
               f"so this simple model is rougher.")

st.subheader("What the model uses")
st.write("Each server is a heat source with its measured power. Air moving through it has a "
         "measured flow rate, which sets how fast that heat is carried away. Two numbers are "
         "fitted: how strongly neighbouring air streams mix, and how fast air is cooled between "
         "servers. The fit gives "
         f"κ = {row['kappa_m2_s']:.1e} m²/s. The second number is poorly determined by "
         "the data, so it is not reported.")

with st.expander("First attempt: diffusion only (did not work)"):
    first = read("calib_summary.csv")
    if first is not None:
        st.dataframe(first[first["design"] == design], hide_index=True)
    st.write("A smooth diffusion-and-decay profile cannot make the sharp hot band seen at some "
             "heights. Its error on held-out sensors is almost the same as guessing the "
             "average. This motivates the flow-based sources above.")

with st.expander("Limits of this comparison"):
    st.write("Only 12 sensors on one rack door. A 1D vertical model, not the plan-view model of the "
             "other tabs, so the fitted mixing cannot be transferred to it. Which rack heats "
             "which sensor is uncertain. The reference temperature is the cooler's minimum rack "
             "inlet reading.")