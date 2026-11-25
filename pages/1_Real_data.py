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
st.write("Dataset: Kuzay et al. (2022), a small air-cooled data centre with a 2 kW load. "
         "Temperatures were measured at 12 heights on the rear door of one rack, every 10 s "
         "for 2 hours. The data is nearly constant in time, so a steady vertical profile is compared. "
         "Each model below adds one physical ingredient to the one before.")

summ = read("calib_summary.csv")
if summ is None:
    st.warning("results/calib_summary.csv not found. Run `python calibration.py` first.")
    st.stop()

c0, c1 = st.columns(2)
design = c0.radio("Design", ["previous", "retrofitted"], horizontal=True,
                  format_func=lambda s: "Previous (door open)" if s == "previous"
                  else "Retrofitted (aisles isolated, door closed)")
srcs = list(summ[summ["design"] == design]["source_set"].unique())
src = c1.radio("Which racks heat the air at the sensors?", srcs, horizontal=True)
sub = summ[(summ["design"] == design) & (summ["source_set"] == src)].reset_index(drop=True)
sens = read(f"calib_sensors_{design}.csv")
if sens is None:
    st.warning(f"results/calib_sensors_{design}.csv not found.")
    st.stop()

models = list(sub["model"])
choice = st.radio("Model", models, index=len(models) - 1, horizontal=True)
row = sub[sub["model"] == choice].iloc[0]
col = f"pred_C [{src}|{choice}]"

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.errorbar(sens["measured_C"], sens["z_m"], xerr=sens["measured_std_C"], fmt="ks", capsize=2,
            label="measured (mean ± std over 2 h)")
order = np.argsort(sens["z_m"].values)
ax.plot(sens[col].values[order], sens["z_m"].values[order], "o-", color="tab:blue", label=choice)
ax.set_xlabel("temperature at the rear of the rack [°C]")
ax.set_ylabel("height [m]")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()

left, right = st.columns([1.3, 1])
left.pyplot(fig)
plt.close(fig)
with right:
    st.metric("Model error (RMSE) [°C]", f"{row['rmse_C']:.2f}")
    st.metric("On held-out sensors (leave-one-out) [°C]", f"{row['loo_rmse_C']:.2f}")
    st.metric("Guessing the average [°C]", f"{row['baseline_rmse_C']:.2f}")
    st.metric("Largest error at one sensor [°C]", f"{row['max_err_C']:.2f}")
    st.caption(f"Fitted numbers: {int(row['n_fitted'])}. Parameters: {row['parameters']}. "
               "The published CFD model reaches 0.5 to 1.6 °C, so these simple models are rougher.")

st.subheader("The model ladder")
st.dataframe(sub[["model", "n_fitted", "parameters", "rmse_C", "loo_rmse_C", "max_err_C"]],
             hide_index=True)
fig2, bx = plt.subplots(figsize=(7, 3))
bx.bar([m.split()[0] for m in sub["model"]], sub["loo_rmse_C"],
       color=["gray", "tab:orange", "tab:blue", "tab:green"][:len(sub)])
bx.axhline(sub["baseline_rmse_C"].iloc[0], color="k", ls="--", lw=1, label="guessing the average")
bx.set_ylabel("held-out RMSE [°C]")
bx.legend(fontsize=8)
bx.grid(alpha=0.3, axis="y")
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

with st.expander("What each model is"):
    st.markdown(
        "- **A diffusion only:** heat from the servers spreads and decays smoothly. The same equation as the plan-view model.\n"
        "- **E energy balance only:** each server heats the air passing through it by power divided by air flow. Nothing fitted.\n"
        "- **B flow-based:** E plus a fitted mixing between neighbouring air streams.\n"
        "- **C flow-based + upward air:** B plus a fitted upward air speed (advection)."
    )
with st.expander("Limits of this comparison"):
    st.write("Only 12 sensors on one rack door. A 1D vertical model, not the plan-view model of the "
             "other pages, so the fitted numbers cannot be transferred to it. Which rack heats "
             "which sensor is uncertain. Cooling between servers is fixed because the data cannot "
             "determine it. The reference temperature is the cooler's minimum rack inlet reading.")