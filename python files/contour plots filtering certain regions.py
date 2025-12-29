import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter, ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter, LogLocator
from matplotlib import rcParams
# ------------------------------------------------------------
# File paths: Input Excel files for mean, std, and CV data
# ------------------------------------------------------------
mean_file = "/home/lavanya/Documents/4/mean data.xlsx"
std_file = "/home/lavanya/Documents/4/std data.xlsx"
cv_file = "/home/lavanya/Documents/4/cv data.xlsx"
# ============================================================
# Function to create smooth log-scale contour plots
# ============================================================
def plot_contour(df, z_label, output_name, sigma_strong, sigma_gentle, patch_threshold):
    # Extract amplitude (amp), duration (dur), and Z-values (e.g., mean/std/CV)
    amp = df.iloc[:, 0].values
    dur = df.iloc[:, 1].values
    z = df.iloc[:, 2].values
    # Convert duration and Z values to log10 scale for smooth interpolation
    dur_log = np.log10(dur)
    z_log = np.log10(z)
    # Define the grid range and resolution for interpolation
    amp_min, amp_max = 1.5, 2.3
    dur_min, dur_max = np.log10(1e-9), np.log10(1e-3)
    amp_grid = np.linspace(amp_min, amp_max, 400)
    dur_grid = np.linspace(dur_min, dur_max, 400)
    AMP, DUR = np.meshgrid(amp_grid, dur_grid)
    # ------------------------------------------------------------
    # Interpolation step:
    # "linear" gives smooth regions; "nearest" fills NaN holes.
    # ------------------------------------------------------------
    Z_linear = griddata((amp, dur_log), z_log, (AMP, DUR), method="linear")
    Z_near = griddata((amp, dur_log), z_log, (AMP, DUR), method="nearest")

    # Compute difference between linear and nearest interpolations
    # Regions with large differences indicate discontinuities or sparse data
    diff = np.abs(Z_linear - Z_near)

    # Mark "patchy" zones: where linear is NaN or differs too much from nearest
    patch_mask = np.isnan(Z_linear) | (diff > patch_threshold)

    # Replace missing (NaN) linear values with nearest-neighbor estimates
    Z_filled = np.where(np.isnan(Z_linear), Z_near, Z_linear)
    # ------------------------------------------------------------
    # Apply Gaussian smoothing:
    # Gentle smoothing → applied to entire data
    # Strong smoothing → applied only to patchy zones
    # ------------------------------------------------------------
    Z_gentle = gaussian_filter(Z_filled, sigma=sigma_gentle)
    Z_strong = gaussian_filter(np.nan_to_num(Z_filled, nan=0.0), sigma=sigma_strong)
    # Blend both smoothed datasets:
    # Use gentle smoothing for normal areas, strong smoothing for patchy ones
    Z_blend = Z_gentle.copy()
    Z_blend[patch_mask] = Z_strong[patch_mask]
    # Convert back from log10(Z) → linear Z values for plotting
    Z_plot = 10 ** Z_blend
    # Print min and max for verification
    print(f"{output_name}: min={np.nanmin(Z_plot):.2e}, max={np.nanmax(Z_plot):.2e}")
    plt.figure(figsize=(8, 6))
    # ✅ define vmin before using it

    # ✅ Always define vmin and vmax first
    vmin = np.nanmin(Z_plot)
    vmax = np.nanmax(Z_plot)

    # ✅ Apply specific scaling rules
    if "std" in output_name.lower():
        vmin = 1e2 # keep std upper limit 1e9
        vmax = 8e8
    elif "cv" in output_name.lower():
        vmax = 9.54
        # ------------------------------------------------------------
    # Contour plot
    # ------------------------------------------------------------
    contour = plt.contourf(
        AMP, 10 ** DUR, Z_plot,
        levels=np.logspace(np.log10(vmin), np.log10(vmax), 300),
        cmap="rainbow", norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    # ------------------------------------------------------------
    # Colorbar formatting with 10^x labels
    # ------------------------------------------------------------
    cbar = plt.colorbar(contour)
    cbar.set_label(z_label, fontsize=24, fontweight="bold",labelpad=15)
    # Define function for scientific-style colorbar labels
    rcParams['mathtext.default'] = 'regular'  # default math font style

    if "cv" in output_name.lower():
        # Define custom tick values for CV
        cbar_ticks = [0.1, 1, 10, 100]
        cbar_ticks = [t for t in cbar_ticks if vmin <= t <= vmax]

        # Use fixed locator/formatter for numeric labels
        cbar.locator = FixedLocator(cbar_ticks)
        cbar.formatter = FixedFormatter([str(t) for t in cbar_ticks])
        cbar.update_ticks()

    else:

        # Use 10^x style for mean/std
        def power_of_ten(x, pos):
            exponent = int(np.log10(x))
            #return rf"$10^{{{exponent}}}$"
            return rf"$\mathbf{{10^{{{exponent}}}}}$"

        cbar.locator = LogLocator(base=10)
        cbar.formatter = FuncFormatter(power_of_ten)
        cbar.update_ticks()

    # 🔹 Apply bold styling *after* update_ticks()
    for tick_label in cbar.ax.get_yticklabels():
      tick_label.set_fontsize(20)
      tick_label.set_fontweight("bold")

    cbar.draw_all()
    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    plt.xlim(amp_min, amp_max)
    plt.ylim(1e-9, 1e-3)
    plt.yscale("log")
    plt.xlabel(r"$\mathbf{V_p\ (V)}$", fontsize=24)
    plt.ylabel(r"$\mathbf{t_{pw}\ (s)}$", fontsize=24)
    # Define nice ticks in seconds
    y_ticks = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    y_labels = ["1 ns", "10 ns", "100 ns", "1 µs", "10 µs", "100 µs", "1 ms" ]
    plt.yticks(y_ticks, y_labels, fontsize=20, fontweight="bold")
    plt.xticks(np.arange(amp_min, amp_max + 0.1, 0.2), fontsize=20, fontweight="bold")
    ax = plt.gca()
    ax.tick_params(axis='x', pad=12)  # More space for x-axis
    ax.tick_params(axis='y', pad=12)  # More space for y-axis
    plt.tight_layout()
    plt.savefig(f"/home/lavanya/Documents/4/{output_name}.png", dpi=300)
    plt.show()
# ============================================================
# Generate and save contour plots for mean, std, and CV data
# ============================================================
# Mean plot: no smoothing (original surface)
plot_contour(pd.read_excel(mean_file), "Mean, μ (Ω)", "mean_contour",
             sigma_strong=0, sigma_gentle=0, patch_threshold=0)

# Std plot: moderate smoothing for stability
plot_contour(pd.read_excel(std_file), "Std, σ (Ω)", "std_contour",
             sigma_strong=7, sigma_gentle=3, patch_threshold=0.09)

# CV plot: strong smoothing to remove discrete patches
plot_contour(pd.read_excel(cv_file), "Coeff. of Variance (σ/μ)", "cv_contour",
             sigma_strong=20, sigma_gentle=20, patch_threshold=0.05)

