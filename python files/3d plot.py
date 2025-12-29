import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pulse_amplitudes = [1.23, 1.25, 1.27, 1.3, 1.33, 1.4]
time_min, time_max = 1e-9, 1e-3
res_min, res_max = 1e3, 1e9

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10(np.linspace(0, 1, len(pulse_amplitudes)))

# Plot a "guide" line for each amplitude spanning the full time/resistance range in log-log style
for idx, pamp in enumerate(pulse_amplitudes):
    # Use 2 points at each corner
    t_vals = np.logspace(np.log10(time_min), np.log10(time_max), 2)
    r_vals = np.logspace(np.log10(res_min), np.log10(res_max), 2)
    for t in t_vals:
        ax.plot([pamp, pamp], [t, t], [res_min, res_max], color=colors[idx], alpha=0.5, linewidth=2)
    for r in r_vals:
        ax.plot([pamp, pamp], [time_min, time_max], [r, r], color=colors[idx], alpha=0.5, linewidth=2)
    # Optional: connect the corners (visual box for each amplitude position)
    ax.plot([pamp, pamp], [time_min, time_max], [res_min, res_max], color=colors[idx], linestyle='dashed', alpha=0.3)

ax.set_xlabel('Pulse amplitude (V)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Resistance (Ω)')
#ax.set_yscale('log')
#ax.set_zscale('log')
ax.set_xlim(min(pulse_amplitudes) - 0.01, max(pulse_amplitudes) + 0.01)
ax.set_ylim(time_min, time_max)
ax.set_zlim(res_min, res_max)
ax.view_init(elev=28, azim=40)
plt.tight_layout()
plt.show()