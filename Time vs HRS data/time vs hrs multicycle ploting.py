import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Folder containing all files ---
folder = "/home/lavanya/Documents/3/time vs HRS 10k data"

# --- Voltage groups and colors ---
files_dict = {
    "1.23V": (["1.23V(1).csv", "1.23V(2).csv"], "blue"),
    "1.3V": (["1.30V(1).csv", "1.30V(2).csv"], "red"),
    "1.4V": (["1.40V(1).csv", "1.40V(2).csv"], "black"),
}

plt.figure(figsize=(8, 6))

for voltage, (file_list, color) in files_dict.items():
    combined_time = []
    combined_hrs = []

    for file_name in file_list:
        file_path = os.path.join(folder, file_name)
        df = pd.read_csv(file_path)

        # Each column pair is one cycle (Time, HRS)
        for i in range(0, df.shape[1], 2):
            time_col = df.columns[i]
            hrs_col = df.columns[i + 1]

            time = df[time_col].dropna().values
            hrs = df[hrs_col].dropna().values

            combined_time.append(time)
            combined_hrs.append(hrs)

    # Concatenate all cycles into one long array
    all_time = np.concatenate(combined_time)
    all_hrs = np.concatenate(combined_hrs)
    # Remove the V from label
    voltage_numeric = voltage.replace("V", "")
    mask = all_hrs <= 2e8
    plt.loglog(all_time[mask], all_hrs[mask], '-', color=color, alpha=0.3, markersize=2, label=f"{voltage_numeric}")

plt.xlim(5e-10, 2e-3)   # ✅ set x-axis range

# Axis labels and ticks
plt.xlabel("time (s)", fontsize=24, fontweight='bold')
plt.ylabel("Resistance (Ω)", fontsize=24, fontweight='bold')
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')

# ✅ Bigger legend marker size and darker symbols
legend = plt.legend(fontsize=14, loc="best", frameon=True, markerscale=3, title=r"$\mathbf{V_p\ (V)\ =}$")
legend.get_title().set_fontsize(14)      # same size as legend values
legend.get_title().set_fontweight('bold')  # match label weight
legend.get_title().set_color('black')    # match label color
# Make legend values bold and dark
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('black')
for handle in legend.legendHandles:
    handle.set_markersize(14)        # make legend dots big
    handle.set_alpha(1.0)            # fully opaque (dark)
plt.tight_layout()
plt.show()