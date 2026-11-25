# Thermal Hotspot Prediction in a Server Room (2D Heat Equation)

MFC end-semester project, Theme B (PDEs).

A 2D transient heat model of a 10 m x 10 m server room with finite rack heat patches and
zoned cooling. Solved with explicit finite differences written from scratch, and
cross-checked with an own Conjugate Gradient steady-state solver and Backward Euler and
Crank-Nicolson schemes. A 1D vertical version of the model is compared with real
measurements from a published data-centre dataset.

**Data note:** the plan-view experiments use synthetic data generated from the parameters
below (illustrative, qualitative conclusions only). The real-data comparison uses a public
dataset, described in the Real data section.

## Model

    d(theta)/dt = alpha * laplacian(theta) - gamma(x,y) * theta + q(x,y,t)

theta = T - 25 C (excess temperature). alpha = effective diffusivity, gamma = cooling rate,
q = rack heating rate.

## Parameters (solver.py)

| Item | Value |
|---|---|
| Room | 10 m x 10 m, 50 x 50 cells (51 x 51 nodes), h = 0.2 m |
| alpha | 0.05 m^2/s |
| q_max | 2.4 K/s (tuned, illustrative) |
| Racks | 6 patches of 1 m x 1 m (5 x 5 nodes) |
| Cooling zones (SW, NW, NE, SE) | 0.02, 0.05, 0.05, 0.12 1/s |
| Hotspot threshold | 35 C |
| Time step | 0.9 / (4 alpha/h^2 + gamma_max/2) = 0.1779 s |
| Steady state | max abs(change)/dt < 1e-6 K/s |

## Setup

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Real data

Kuzay et al. (2022), "Numerical and experimental dataset for a retrofitted data center",
Data in Brief 45:108587, DOI 10.1016/j.dib.2022.108587.
Data: https://doi.org/10.5281/zenodo.7035829

Download data.tar.xz and unpack it into data/ (the raw data is not stored in this repo;
the code expects data/data/previousDesign and data/data/retrofittedDesign).

The dataset has 12 rear-door temperature sensors at different heights on one rack, sampled
every 10 s for about 2 hours, plus in-row cooler readings. The temperatures are almost
constant in time, so a steady vertical profile is compared.

Results (leave-one-out RMSE, all racks as sources):

| Design | Baseline (predict mean) | A diffusion only | E energy balance only | B flow-based + mixing | C B + upward air |
|---|---|---|---|---|---|
| Previous | 3.66 C | 3.50 C | 3.08 C | 2.14 C | 2.14 C (v = 0) |
| Retrofitted | 3.37 C | 3.23 C | 3.12 C | 1.97 C | 1.97 C (v = 0) |

Model B (measured power and air flow of each server, plus a fitted mixing of about
1.6e-4 to 2.5e-4 m^2/s) reduces the held-out error by about 40% compared with diffusion
alone and about 42% compared with predicting the mean. Fitting an upward air speed gave no
improvement (the fit chose v = 0). Largest single-sensor errors are 3.6 to 3.8 C, against
1.1 to 1.6 C for the published CFD. The fitted vertical mixing cannot be transferred to the
plan-view model. Cooling between servers is not identifiable from this data and is fixed.

## Run

    bash run_all.sh                    # rebuilds everything below

or one at a time:

    python validation.py               # V1, V3, V4, V5, V6
    python run_demo.py                 # steady heatmap
    python experiments.py              # experiments 1, 1b, 2, 3A, 3B, 4
    python experiments.py exp3a        # a single one: exp1 exp1b exp2 exp3a exp3b exp4
    python experiments2.py             # check, 3C, 3D, 5, 1b-ii
    python experiments2.py check       # BE and CN against CG
    python calibration.py              # diffusion-only fit to real data
    python calibration2.py             # flow-based fit to real data
    streamlit run app.py               # interactive app (4 tabs + Real data page)

The model is deterministic, so there are no random seeds.

## Files

| File | Contents |
|---|---|
| solver.py | grid, rack patches, gamma map, stable dt, explicit step, steady detector, sparse assembly, own CG |
| implicit.py | Backward Euler and Crank-Nicolson (splu factorised once, Dirichlet walls only) |
| validation.py | V1 exact solution, V3 energy balance, V4 superposition, V5 solver cross-check, V6 maximum principle |
| experiments.py | Experiments 1, 1b, 2, 3A, 3B, 4 |
| experiments2.py | Experiments 3C, 3D, 5, 1b-ii |
| calibration.py | loads the real data; 1D diffusion-only fit |
| calibration2.py | 1D flow-based fit using measured power and air flow |
| app.py | Streamlit app (4 tabs) |
| pages/1_Real_data.py | Streamlit page for the real-data comparison |
| run_all.sh | rebuilds every result |

## Figure to command

| Figure | Command |
|---|---|
| demo_heatmap.png | python run_demo.py |
| exp1_workload_heatmaps.png, exp1_maxT_vs_time.png | python experiments.py exp1 |
| exp1b_burst.png | python experiments.py exp1b |
| exp2_cooling.png | python experiments.py exp2 |
| exp3a_stability.png | python experiments.py exp3a |
| exp3b_convergence.png | python experiments.py exp3b |
| exp4_boundary.png | python experiments.py exp4 |
| exp3c_schemes.png, exp3c_switch.png | python experiments2.py exp3c exp3c_switch |
| exp3d_steady_solvers.png | python experiments2.py exp3d |
| exp5_sensitivity.png | python experiments2.py exp5 |
| exp1b_intensity.png | python experiments2.py exp1b2 |
| calib_previous.png, calib_retrofitted.png | python calibration.py |
| calib2_previous.png, calib2_retrofitted.png | python calibration2.py |

CSV tables with matching names are written to results/.

## Key validation results

| Check | Result |
|---|---|
| V1 exact sine-mode solution (RMSE) | 8.6e-4 K |
| V3 energy balance (relative imbalance) | 2.5e-11 |
| V4 superposition (90% vs 4.5 x 20%) | 3.6e-15 K |
| V5 own CG vs SciPy spsolve | 2.7e-10 K (141 iterations) |
| V6 minimum theta | 0 |
| Fitted order: space / time (FE) / combined | 1.98 / 1.00 / 1.97 |
| Fitted order in time: BE / CN | 0.99 / 2.00 |

## Limitations

2D plan view, no rack height, no computed airflow (effective diffusivity instead),
illustrative plan-view parameters, model time scale of seconds (real rooms respond over
minutes to hours). Implicit schemes and own CG assembly support Dirichlet walls only; the
Neumann steady state comes from time marching. The real-data comparison uses only 12
sensors on one rack door and a 1D vertical model.

## Status

- [x] Solver, validation V1 to V6, own CG, BE and CN
- [x] Experiments 1 to 5
- [x] Streamlit app
- [x] Real-data calibration
- [ ] Report, slides, references (not in this repo)