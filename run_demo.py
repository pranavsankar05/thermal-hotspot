import numpy as np
import matplotlib.pyplot as plt
from solver import *

n = 50
h, x, X, Y = make_grid(n)
gamma = make_gamma(n)
q = make_q(n, RACKS, [0.9, 0.9, 0.5, 0.5, 0.2, 0.2])
dt = stable_dt(ALPHA, h, gamma.max())
print(f"h = {h} m, dt = {dt:.4f} s")

th, t_ss = run_to_steady(np.zeros_like(q), dt, ALPHA, h, gamma, q)
T = T_AMB + th
print(f"steady at t = {t_ss:.1f} s, peak T = {T.max():.1f} C")
i, j = np.unravel_index(np.argmax(T), T.shape)
print(f"hotspot at x = {x[i]:.1f} m, y = {x[j]:.1f} m")
print(f"cells above {THRESHOLD} C: {(T > THRESHOLD).sum()}")

plt.imshow(T.T, origin="lower", extent=[0, L, 0, L], cmap="inferno")
plt.colorbar(label="Temperature [°C]")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.title("Steady-state temperature, mixed workload")
plt.savefig("results/demo_heatmap.png", dpi=150)
plt.show()