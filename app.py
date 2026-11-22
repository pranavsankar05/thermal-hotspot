import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

from solver import *

st.set_page_config(page_title="Server-room hotspot model", layout="wide")

N = 50
H, XN, XG, YG = make_grid(N)
MIXED = [0.9, 0.9, 0.5, 0.5, 0.2, 0.2]
ZONE_NAMES = ["SW", "NW", "NE", "SE"]
BASE_Z = (0.02, 0.05, 0.05, 0.12)
RES = "results"
FAIL_GAMMA = 0.005          # "cooling off" value (not exactly 0, keeps the model well behaved)
CUR = "Your current setup (sidebar)"

# name -> (rack workloads, cooling (SW, NW, NE, SE), walls)
PRESETS = {
    "Medium load, all racks 50%": ([0.5] * 6, BASE_Z, "dirichlet"),
    "Mixed load (baseline)": (MIXED, BASE_Z, "dirichlet"),
    "Light load, all racks 20%": ([0.2] * 6, BASE_Z, "dirichlet"),
    "Heavy load, all racks 90%": ([0.9] * 6, BASE_Z, "dirichlet"),
    "Mixed load, NW cooling failed": (MIXED, (0.02, 0.005, 0.05, 0.12), "dirichlet"),
    "Mixed load, strong cooling everywhere": (MIXED, (0.12,) * 4, "dirichlet"),
    "Mixed load, insulated walls": (MIXED, BASE_Z, "neumann"),
}


# ------------------------------------------------------------------ session-state callbacks
def apply_preset():
    pw, pz, pbc = PRESETS[st.session_state["preset"]]
    for k in range(6):
        st.session_state[f"w{k}"] = int(round(pw[k] * 100))
    for i, name in enumerate(ZONE_NAMES):
        st.session_state[f"z{name}"] = float(pz[i])
        st.session_state.pop(f"zsave{name}", None)
    st.session_state["bc"] = pbc


def set_all_racks():
    for k in range(6):
        st.session_state[f"w{k}"] = st.session_state["master_w"]


def set_all_racks_tab():
    v = st.session_state["tab_master_w"]
    st.session_state["master_w"] = v
    for k in range(6):
        st.session_state[f"w{k}"] = v


def set_all_zones():
    for name in ZONE_NAMES:
        st.session_state[f"z{name}"] = st.session_state["master_z"]


def set_zone(name, on):
    """Cooling on: restore the saved value. Cooling off: remember the value, then use FAIL_GAMMA."""
    i = ZONE_NAMES.index(name)
    cur = float(st.session_state[f"z{name}"])
    if on:
        if cur <= FAIL_GAMMA + 1e-9:
            st.session_state[f"z{name}"] = float(st.session_state.get(f"zsave{name}", BASE_Z[i]))
    else:
        if cur > FAIL_GAMMA + 1e-9:
            st.session_state[f"zsave{name}"] = cur
        st.session_state[f"z{name}"] = FAIL_GAMMA


def toggle_zone(name):
    set_zone(name, st.session_state[f"zon{name}"])


def all_cooling(on):
    for name in ZONE_NAMES:
        set_zone(name, on)


st.session_state.setdefault("preset", "Mixed load (baseline)")
st.session_state.setdefault("master_w", 50)
st.session_state.setdefault("master_z", 0.05)
if "w0" not in st.session_state:
    apply_preset()


# ------------------------------------------------------------------ model helpers
def zones_from(z):
    return dict(zip(ZONE_NAMES, z))


def zone_of(cx, cy):
    return ("N" if cy >= L / 2 else "S") + ("E" if cx >= L / 2 else "W")


@st.cache_data(show_spinner=False)
def steady_field(w, z, bc):
    """Steady excess temperature. Dirichlet: own CG. Neumann: time marching."""
    gamma = make_gamma(N, zones_from(z))
    q = make_q(N, RACKS, list(w))
    if bc == "dirichlet":
        th, _ = steady_solve(N, ALPHA, gamma, q, "cg")
    else:
        dt = stable_dt(ALPHA, H, gamma.max())
        th, _ = run_to_steady(np.zeros_like(q), dt, ALPHA, H, gamma, q, bc=bc,
                              tol=1e-6, t_max=5000.0)
    return th


@st.cache_data(show_spinner=False)
def run_event(w, z, bc, mode, t_end, rack_idx, zone_name):
    """mode: 'cold' (switch on), 'spike' (one rack to 100%), 'fail' (one zone loses cooling).
    Event modes start from the steady state of the chosen setup and trigger at t = 60 s."""
    gamma = make_gamma(N, zones_from(z))
    q = make_q(N, RACKS, list(w))
    gamma_after, q_after, t_ev = gamma, q, 60.0
    if mode == "cold":
        th = np.zeros_like(q)
        t_ev = None
    else:
        th = steady_field(w, z, bc).copy()
        if mode == "spike":
            w2 = list(w)
            w2[rack_idx] = 1.0
            q_after = make_q(N, RACKS, w2)
        else:
            z2 = dict(zones_from(z))
            z2[zone_name] = min(z2[zone_name], FAIL_GAMMA)
            gamma_after = make_gamma(N, z2)
    dt0 = stable_dt(ALPHA, H, max(gamma.max(), gamma_after.max()))
    n_steps = int(np.ceil(t_end / dt0))
    dt = t_end / n_steps
    every = max(1, n_steps // 100)
    mx = np.empty(n_steps + 1)
    mx[0] = th.max()
    ft, fields = [0.0], [th.copy()]
    for n in range(n_steps):
        after = (t_ev is None) or (n * dt >= t_ev)
        th = step(th, dt, ALPHA, H, gamma_after if after else gamma,
                  q_after if after else q, bc)
        mx[n + 1] = th.max()
        if (n + 1) % every == 0:
            ft.append((n + 1) * dt)
            fields.append(th.copy())
    return {"t": np.linspace(0.0, t_end, n_steps + 1), "peak": T_AMB + mx,
            "ft": np.array(ft), "fields": np.array(fields), "t_ev": t_ev}


def near_rack(x, y):
    d = [np.hypot(x - cx, y - cy) for cx, cy in RACKS]
    k = int(np.argmin(d))
    return k + 1, d[k]


def hot_nodes(T):
    return int((T > THRESHOLD).sum())


def get_setup(name):
    """Return (workloads tuple, zones tuple, walls) for a preset or the current sidebar setup."""
    if name == CUR:
        return w, z, bc
    pw_, pz_, pb_ = PRESETS[name]
    return tuple(pw_), tuple(pz_), pb_


# ------------------------------------------------------------------ drawing helpers
def add_racks(ax):
    for k, (cx, cy) in enumerate(RACKS):
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, fill=False, ec="white", lw=0.9))
        ax.text(cx, cy, str(k + 1), color="white", ha="center", va="center", fontsize=8)


def draw_room(ax, T, vmax, title):
    im = ax.imshow(T.T, origin="lower", extent=[0, L, 0, L],
                   cmap="inferno", vmin=T_AMB, vmax=vmax)
    if T.max() > THRESHOLD:
        ax.contour(XN, XN, T.T, levels=[THRESHOLD], colors="cyan", linewidths=1.2)
    add_racks(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    return im


def draw_setup(w_, z_, bc_):
    """Small map of the current controls: rack workload (red) and cooling zones (blue)."""
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    g = make_gamma(N, zones_from(z_))
    ax.imshow(g.T, origin="lower", extent=[0, L, 0, L], cmap="Blues", vmin=0, vmax=0.2)
    ax.axvline(L / 2, color="k", lw=0.5)
    ax.axhline(L / 2, color="k", lw=0.5)
    for k, (cx, cy) in enumerate(RACKS):
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, fc=plt.cm.Reds(0.1 + 0.9 * w_[k]),
                               ec="k", lw=0.8))
        ax.text(cx, cy, f"{k + 1}\n{int(round(w_[k] * 100))}%", ha="center", va="center",
                fontsize=6.5, color="white" if w_[k] > 0.6 else "black")
    pos = {"SW": (0.2, 0.25), "NW": (0.2, 9.55), "NE": (5.2, 9.55), "SE": (5.2, 0.25)}
    for i, name in enumerate(ZONE_NAMES):
        state = "OFF" if z_[i] <= FAIL_GAMMA + 1e-9 else f"γ={z_[i]:.3f}"
        ax.text(*pos[name], f"{name} {state}", fontsize=6.5)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.tick_params(labelsize=7)
    ax.set_title("walls: " + ("ambient" if bc_ == "dirichlet" else "insulated"), fontsize=8)
    fig.tight_layout()
    return fig


def draw_frame(run, k, vmax):
    T = T_AMB + run["fields"][k]
    tk = run["ft"][k]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
    im = draw_room(a, T, vmax, f"Room at t = {tk:.0f} s   (cyan line = 35 °C)")
    fig.colorbar(im, ax=a, label="Temperature [°C]", fraction=0.046)
    m = run["t"] <= tk
    b.plot(run["t"][m], run["peak"][m], color="tab:red")
    b.axhline(THRESHOLD, color="k", ls="--", lw=0.8, label="35 °C alarm level")
    if run["t_ev"] is not None:
        b.axvline(run["t_ev"], color="tab:blue", ls=":", label="event starts")
    b.set_xlim(0, run["t"][-1])
    b.set_ylim(run["peak"].min() - 1, max(run["peak"].max(), THRESHOLD) + 1)
    b.set_xlabel("time [s]")
    b.set_ylabel("hottest point in the room [°C]")
    b.set_title("Peak temperature so far", fontsize=10)
    b.grid(alpha=0.3)
    b.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def show_saved(options):
    label = st.selectbox("Saved result", list(options))
    png, csv = options[label]
    if os.path.exists(f"{RES}/{png}"):
        st.image(f"{RES}/{png}")
    else:
        st.warning(f"{RES}/{png} not found. Run the experiment scripts first.")
    if csv and os.path.exists(f"{RES}/{csv}"):
        st.dataframe(pd.read_csv(f"{RES}/{csv}"), hide_index=True)


# ------------------------------------------------------------------ trust-tab helpers
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


@st.cache_data(show_spinner=False)
def exact_test():
    h, x, X, Y = make_grid(N)
    g0, t_end = 0.05, 10.0
    th0 = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    gam = np.full_like(th0, g0)
    out = simulate(th0, stable_dt(ALPHA, h, g0), t_end, ALPHA, h, gam, np.zeros_like(th0))
    rate = 2 * ALPHA * np.pi**2 / L**2 + g0
    exact = th0 * np.exp(-rate * t_end)
    mid = N // 2
    return x, out["theta"][:, mid], exact[:, mid], rmse(out["theta"], exact)


@st.cache_data(show_spinner=False)
def stability_curve(pct, g0=0.12, n_run=150):
    dt = pct / 100.0 * (1.0 / (4 * ALPHA / H**2 + g0 / 2))
    gamma = np.full((N + 1, N + 1), g0)
    q = np.zeros((N + 1, N + 1))
    i = np.arange(N + 1)
    s = np.sin((N - 1) * np.pi * i / N)          # fastest-oscillating mode on the grid
    th = 1e-3 * np.outer(s, s)
    hist = [np.abs(th).max()]
    for _ in range(n_run):
        th = step(th, dt, ALPHA, H, gamma, q)
        hist.append(np.abs(th).max())
    return dt, np.array(hist)


@st.cache_data(show_spinner=False)
def energy_balance():
    gamma = make_gamma(N)
    q = make_q(N, RACKS, MIXED)
    th, _ = steady_solve(N, ALPHA, gamma, q, "cg")
    ti = th[1:-1, 1:-1]
    n = N - 1
    nw = np.zeros((n, n))
    nw[0, :] += 1; nw[-1, :] += 1; nw[:, 0] += 1; nw[:, -1] += 1
    heat_in = q[1:-1, 1:-1].sum()
    cooling = (gamma[1:-1, 1:-1] * ti).sum()
    wall = (ALPHA / H**2) * (ti * nw).sum()
    return heat_in, cooling, wall


@st.cache_data(show_spinner=False)
def solver_agreement():
    gamma = make_gamma(N)
    q = make_q(N, RACKS, MIXED)
    dt = stable_dt(ALPHA, H, gamma.max())
    th_t, _ = run_to_steady(np.zeros_like(q), dt, ALPHA, H, gamma, q, tol=1e-8, t_max=5000.0)
    th_d, _ = steady_solve(N, ALPHA, gamma, q, "direct")
    th_c, its = steady_solve(N, ALPHA, gamma, q, "cg")
    return (np.max(np.abs(th_c - th_d)), np.max(np.abs(th_t - th_d)), its,
            T_AMB + th_d.max())


# ------------------------------------------------------------------ sidebar: control panel
st.sidebar.header("Control panel")
map_box = st.sidebar.container()            # filled after the sliders, so it is never stale

st.sidebar.selectbox("Scenario", list(PRESETS), key="preset", on_change=apply_preset)
st.sidebar.button("Reset to scenario", on_click=apply_preset)

st.sidebar.markdown("**Racks (workload)**")
st.sidebar.slider("All racks together [%]", 0, 100, step=5, key="master_w",
                  on_change=set_all_racks)
for k, (cx, cy) in enumerate(RACKS):
    st.sidebar.slider(f"Rack {k + 1} ({zone_of(cx, cy)} zone) [%]", 0, 100, step=5, key=f"w{k}")

st.sidebar.markdown("**Cooling zones (γ, 1/s)**")
st.sidebar.slider("All zones together", 0.005, 0.200, step=0.005, format="%.3f",
                  key="master_z", on_change=set_all_zones)
for name in ZONE_NAMES:
    st.sidebar.slider(f"{name} zone", 0.005, 0.200, step=0.005, format="%.3f", key=f"z{name}")

st.sidebar.markdown("**Walls**")
st.sidebar.radio("Wall condition", ["dirichlet", "neumann"], key="bc", label_visibility="collapsed",
                 format_func=lambda s: "Held at ambient 25 °C" if s == "dirichlet"
                 else "Insulated (no heat through walls)")

w = tuple(st.session_state[f"w{k}"] / 100 for k in range(6))
z = tuple(float(st.session_state[f"z{name}"]) for name in ZONE_NAMES)
bc = st.session_state["bc"]

dt_limit = 1.0 / (4 * ALPHA / H**2 + max(z) / 2)
st.sidebar.markdown("**Numerical safety**")
st.sidebar.caption(f"Largest stable time step: {dt_limit:.4f} s. The app uses 90% of it.")

with map_box:
    st.markdown("**Your room right now**")
    fig = draw_setup(w, z, bc)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Red squares = racks (darker = harder working). Blue = cooling "
               "(darker = stronger). Drag any slider and this map updates.")

# ------------------------------------------------------------------ header and tabs
st.title("Where does a server room overheat, and how fast?")
st.caption("A 2D heat-equation model of a 10 m × 10 m room. Synthetic, illustrative parameters. "
           "Qualitative conclusions only.")

tab0, tab1, tab2, tab3 = st.tabs(["1. The idea", "2. Watch it happen",
                                  "3. What if?", "4. Can we trust it?"])

# ================================================================== TAB 0
with tab0:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("The problem")
        st.write("Servers turn electricity into heat. If some racks work harder than others, "
                 "and the cooling is stronger in some parts of the room than in others, "
                 "parts of the room can overheat.")
        st.write("This app predicts **where** the hot spots form, **how fast** they appear, "
                 "and **how much** the cooling and the workload change them.")
        st.subheader("The model in three sentences")
        st.markdown(
            "1. Each rack is a heat source.\n"
            "2. Heat spreads sideways through the air (diffusion).\n"
            "3. Cooling units remove heat, strongly in some zones and weakly in others."
        )
        st.latex(r"\frac{\partial \theta}{\partial t}="
                 r"\underbrace{\alpha\nabla^2\theta}_{\text{spreading}}"
                 r"-\underbrace{\gamma(x,y)\,\theta}_{\text{cooling}}"
                 r"+\underbrace{q(x,y,t)}_{\text{racks}}")
        st.caption("θ = temperature above the 25 °C room ambient. α = how fast heat spreads "
                   "[m²/s]. γ = cooling rate [1/s]. q = rack heating rate [K/s].")
        st.subheader("How the computer solves it")
        st.write("The room is cut into a 50 × 50 grid. At every grid point, the program looks at the "
                 "four neighbours, adds the rack heat, subtracts the cooling, and steps forward a "
                 "fraction of a second. Repeat a few thousand times and you have a movie of the room.")
        st.info("Use the **Control panel** on the left (or the Quick controls in the next tab) "
                "to set every rack and every cooling zone, then press Play in "
                "**2. Watch it happen**.")
    with c2:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        g = make_gamma(N, zones_from(z))
        im = ax.imshow(g.T, origin="lower", extent=[0, L, 0, L], cmap="Blues", vmin=0, vmax=0.2)
        for name, (px, py) in {"SW": (1.4, 0.4), "NW": (1.4, 9.5), "NE": (8.6, 9.5), "SE": (8.6, 0.4)}.items():
            ax.text(px, py, f"{name}: γ = {z[ZONE_NAMES.index(name)]:.3f}", fontsize=8,
                    ha="center", color="k")
        ax.axvline(5, color="k", lw=0.6)
        ax.axhline(5, color="k", lw=0.6)
        for k, (cx, cy) in enumerate(RACKS):
            ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, fc=plt.cm.Reds(0.1 + 0.9 * w[k]), ec="k"))
            ax.text(cx, cy, f"{k + 1}", color="white" if w[k] > 0.6 else "black",
                    ha="center", va="center", fontsize=9)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("The room: red = racks, blue = cooling strength (darker = stronger)")
        fig.colorbar(im, ax=ax, label="cooling rate γ [1/s]", fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
    with st.expander("What this model leaves out (limitations)"):
        st.write("2D plan view only. No rack height. Airflow is not computed; an effective "
                 "diffusivity stands in for air mixing. Parameters are illustrative, and the model's "
                 "time scale (tens of seconds) is faster than a real room (minutes to hours). "
                 "Use it to compare scenarios, not to predict real temperatures.")

# ================================================================== TAB 1
with tab1:
    st.write("Set the racks and cooling, choose an event, press **Play**, and watch the room "
             "(left) and the hottest point (right). The alarm level is 35 °C.")

    # ---------------- quick controls (linked to the sidebar)
    with st.container(border=True):
        st.markdown("**Quick controls**")
        avg = int(round(np.mean(w) * 100 / 5) * 5)
        st.session_state["tab_master_w"] = avg          # always show the current average
        st.slider("All racks workload [%] (shows the average; moving it sets every rack)",
                  0, 100, step=5, key="tab_master_w", on_change=set_all_racks_tab)

        st.markdown(f"Cooling zones (off = γ {FAIL_GAMMA}, a failed unit that still leaks a little)")
        cols = st.columns([1, 1, 1, 1, 1.3, 1.3])
        for i, name in enumerate(ZONE_NAMES):
            st.session_state[f"zon{name}"] = z[i] > FAIL_GAMMA + 1e-9   # mirror the sidebar
            cols[i].checkbox(f"{name} cooling on", key=f"zon{name}",
                             on_change=toggle_zone, args=(name,))
            cols[i].caption("OFF" if z[i] <= FAIL_GAMMA + 1e-9 else f"γ = {z[i]:.3f}")
        cols[4].button("All cooling ON", on_click=all_cooling, args=(True,))
        cols[5].button("All cooling OFF", on_click=all_cooling, args=(False,))

    ctl1, ctl2, ctl3 = st.columns([2, 1.3, 1])
    mode_label = ctl1.radio("What happens?",
                            ["A rack spikes to 100%", "A cooling zone fails",
                             "Racks switch on from a cold room"], horizontal=False)
    mode = {"A rack spikes to 100%": "spike", "A cooling zone fails": "fail",
            "Racks switch on from a cold room": "cold"}[mode_label]
    rack_idx, zone_name = 0, "NW"
    if mode == "spike":
        rack_idx = ctl2.selectbox("Which rack?", range(6), format_func=lambda k: f"Rack {k + 1}")
    elif mode == "fail":
        zone_name = ctl2.selectbox("Which zone?", ZONE_NAMES, index=1)
    speed = ctl3.select_slider("Speed", ["Slow", "Normal", "Fast"], value="Normal")
    stride = {"Slow": 1, "Normal": 2, "Fast": 4}[speed]

    t_end = 400.0 if mode == "cold" else 240.0
    run = run_event(w, z, bc, mode, t_end, rack_idx, zone_name)
    vmax = max(40.0, float(run["peak"].max()))

    if mode == "cold":
        st.caption("The room starts at 25 °C and every rack switches on at t = 0, "
                   "with the racks and cooling set above.")
    else:
        what = (f"rack {rack_idx + 1} jumps to 100% load" if mode == "spike"
                else f"the {zone_name} cooling zone drops to γ = {FAIL_GAMMA}")
        st.caption(f"The room starts in the steady state of your current setup. "
                   f"At t = 60 s, {what}.")

    play = st.button("▶ Play", type="primary")
    holder = st.empty()
    n_fr = len(run["ft"])
    t_sel = st.slider("Or scrub through time [s]", 0.0, t_end, t_end, 1.0, key=f"scrub_{mode}")
    if play:
        for k in list(range(0, n_fr, stride)) + [n_fr - 1]:
            fig = draw_frame(run, k, vmax)
            holder.pyplot(fig)
            plt.close(fig)
    else:
        k = int(np.argmin(np.abs(run["ft"] - t_sel)))
        fig = draw_frame(run, k, vmax)
        holder.pyplot(fig)
        plt.close(fig)

    # ---- plain-language verdict for the whole run
    t, pk, t_ev = run["t"], run["peak"], run["t_ev"]
    hot = pk > THRESHOLD
    T_end = T_AMB + run["fields"][-1]
    i, j = np.unravel_index(np.argmax(T_end), T_end.shape)
    rk, dist = near_rack(XN[i], XN[j])
    where = f"next to rack {rk}" if dist < 1.5 else "between racks"
    st.subheader("What happened")
    if not hot.any():
        st.success(f"The room never reaches {THRESHOLD:.0f} °C. Highest temperature: {pk.max():.1f} °C.")
    elif t_ev is not None and pk[t <= t_ev].max() > THRESHOLD:
        st.warning(f"The room was already above {THRESHOLD:.0f} °C before the event. "
                   f"It peaks at {pk.max():.1f} °C.")
    else:
        t_first = float(t[np.argmax(hot)])
        delay = t_first - (t_ev or 0.0)
        label = "after the event starts" if t_ev is not None else "after switch-on"
        st.error(f"The alarm level is crossed **{delay:.0f} s {label}**. Highest temperature: {pk.max():.1f} °C. "
                 + ("A spike shorter than this would not trip the alarm." if mode == "spike" else ""))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Highest temperature [°C]", f"{pk.max():.2f}")
    m2.metric("Hottest spot at the end", f"({XN[i]:.1f}, {XN[j]:.1f}) m", where, delta_color="off")
    m3.metric("Hot area at the end [m²]", f"{hot_nodes(T_end) * H * H:.2f}",
              f"{hot_nodes(T_end)} grid nodes", delta_color="off")
    m4.metric("Room average at the end [°C]", f"{T_end.mean():.2f}")

# ================================================================== TAB 2
with tab2:
    st.write("Compare two setups once the room has settled. Pick **Your current setup** to "
             "compare whatever you built in the Control panel against a preset. "
             "Both use the same colour scale.")
    options = [CUR] + list(PRESETS)
    a, b = st.columns(2)
    pa = a.selectbox("Scenario A", options, index=2)       # Mixed load (baseline)
    pb = b.selectbox("Scenario B", options, index=0)       # your current setup
    res = {}
    for key, name in (("A", pa), ("B", pb)):
        ww, zz, bb = get_setup(name)
        T = T_AMB + steady_field(tuple(ww), tuple(zz), bb)
        i, j = np.unravel_index(np.argmax(T), T.shape)
        res[key] = (name, T, float(T.max()), hot_nodes(T), (XN[i], XN[j]))
    vmax = max(40.0, res["A"][2], res["B"][2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, key in zip(axes, ("A", "B")):
        name, T, pkv, nh, loc = res[key]
        im = draw_room(ax, T, vmax, f"{key}: {name}")
    fig.colorbar(im, ax=axes.tolist(), label="Temperature [°C]", shrink=0.9)
    st.pyplot(fig)
    plt.close(fig)

    rows = []
    for key in ("A", "B"):
        name, T, pkv, nh, loc = res[key]
        rows.append({"scenario": f"{key}: {name}", "peak [°C]": round(pkv, 2),
                     "hot area [m²]": round(nh * H * H, 2),
                     "hottest spot (x, y) [m]": f"({loc[0]:.1f}, {loc[1]:.1f})"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    dA, dB = res["A"], res["B"]
    diff = dB[2] - dA[2]
    st.markdown(f"**In one sentence:** going from A to B changes the steady peak by "
                f"**{diff:+.2f} °C** and the hot area from **{dA[3] * H * H:.2f} m²** to "
                f"**{dB[3] * H * H:.2f} m²**.")
    st.caption("Steady state means the temperature no longer changes with time. "
               "Hot area = number of grid nodes above 35 °C × h².")

# ================================================================== TAB 3
with tab3:
    st.write("There are no real measurements to compare against, so the model has to prove "
             "itself. Four independent checks (these always use the fixed baseline setup, "
             "not your sliders):")

    st.subheader("1. Does it match a case with a known exact answer?")
    x, num, ex, err = exact_test()
    c1, c2 = st.columns([2, 1])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(x, ex, "k-", lw=2, label="exact formula")
    ax.plot(x, num, "r--", lw=1.5, label="simulation")
    ax.set_xlabel("x along the room's middle line [m]")
    ax.set_ylabel("temperature excess [K]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    c1.pyplot(fig)
    plt.close(fig)
    c2.metric("Error (RMSE) [K]", f"{err:.1e}")
    c2.caption("A sine-shaped heat pattern has a known exact decay. The two curves overlap. "
               "The error shrinks 4× each time the grid is refined (order 2).")

    st.subheader("2. Does it blow up if the time step is too big?")
    pct = st.slider("Time step as % of the predicted stability limit", 50, 130, 90, 5)
    dts, hist = stability_curve(pct)
    c1, c2 = st.columns([2, 1])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.semilogy(np.maximum(hist, 1e-30))
    ax.set_xlabel("time step number")
    ax.set_ylabel("size of the fastest wiggle [K]")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    c1.pyplot(fig)
    plt.close(fig)
    if pct <= 100:
        c2.success(f"Δt = {dts:.3f} s: the wiggle dies out. Stable.")
    else:
        c2.error(f"Δt = {dts:.3f} s: the wiggle grows. Unstable.")
    c2.caption("The limit comes from a pen-and-paper derivation. The simulation blows up "
               "just above it, as predicted.")

    st.subheader("3. Is energy conserved?")
    qin, cool, wall = energy_balance()
    c1, c2 = st.columns([2, 1])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.barh(["Heat removed"], [cool], color="tab:blue", label="by cooling units")
    ax.barh(["Heat removed"], [wall], left=[cool], color="tab:green", label="through the walls")
    ax.barh(["Heat produced"], [qin], color="tab:red", label="by racks")
    ax.set_xlabel("heating rate summed over the room [K/s]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    c1.pyplot(fig)
    plt.close(fig)
    c2.metric("Imbalance", f"{abs(qin - cool - wall) / qin:.1e}")
    c2.caption(f"{qin:.1f} produced = {cool:.1f} + {wall:.1f} removed. Nothing is created or lost.")

    st.subheader("4. Do three different solvers agree?")
    d_cg, d_t, its, pk = solver_agreement()
    c1, c2, c3 = st.columns(3)
    c1.metric("Own CG vs SciPy direct solve [K]", f"{d_cg:.1e}")
    c2.metric("Time marching vs direct solve [K]", f"{d_t:.1e}")
    c3.metric("CG iterations", f"{its}")
    st.caption(f"Steady state computed three ways (mixed load, peak {pk:.2f} °C). "
               "Largest difference anywhere in the room is shown.")

    with st.expander("Saved figures from the report (Experiments 3A to 3D and 5)"):
        show_saved({
            "3A: stability": ("exp3a_stability.png", "exp3a_stability.csv"),
            "3B: convergence orders": ("exp3b_convergence.png", None),
            "3C: scheme comparison": ("exp3c_schemes.png", "exp3c_schemes.csv"),
            "3C-ii: BE vs CN switch-on": ("exp3c_switch.png", None),
            "3D: steady-state solvers": ("exp3d_steady_solvers.png", "exp3d_steady_solvers.csv"),
            "5: sensitivity": ("exp5_sensitivity.png", "exp5_sensitivity.csv"),
        })