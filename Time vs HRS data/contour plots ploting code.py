import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib import font_manager as fm

# File paths
mean_file = "/home/lavanya/Documents/4/mean data.xlsx"
std_file = "/home/lavanya/Documents/4/std data.xlsx"
cv_file = "/home/lavanya/Documents/4/cv data.xlsx"

# Helper function to plot contour
def plot_contour(df, title, z_label, output_name, sigma):
    amp = df.iloc[:, 0].values   # 1st column: amplitude
    dur = df.iloc[:, 1].values   # 2nd column: duration
    z = df.iloc[:, 2].values     # 3rd column: mean/std/cv

    # Log transform for interpolation stability
    dur_log = np.log10(dur)
    z_log = np.log10(z)

    # Create grid
    amp_min, amp_max = 1.7, 2      # start x-axis from 1.2
    dur_max = 1e-4
    dur_grid = np.linspace(min(dur_log), np.log10(dur_max), 400)
    amp_grid = np.linspace(amp_min, amp_max, 400)
    AMP, DUR = np.meshgrid(amp_grid, dur_grid)

    Z = griddata((amp, dur_log), z_log, (AMP, DUR), method='linear')
    Z_smooth = gaussian_filter(np.nan_to_num(Z, nan=np.nanmin(Z)), sigma=sigma)

    print(np.nanmin(10 ** Z_smooth), np.nanmax(10 ** Z_smooth))

    # Plot
    plt.figure(figsize=(7, 6))
    #contour = plt.contourf(AMP, 10**DUR, 10**Z_smooth, levels=160, cmap='viridis',norm=LogNorm())
    vmin = np.nanmin(10 ** Z_smooth)
    vmax = np.nanmax(10 ** Z_smooth)
    contour = plt.contourf(AMP, 10 ** DUR, 10 ** Z_smooth, levels=np.logspace(np.log10(vmin), np.log10(vmax), 200),
                 cmap='rainbow', norm=LogNorm(vmin=vmin, vmax=vmax))
   # plt.colorbar()
    # Set up colorbar after contourf
    cbar = plt.colorbar(contour)
    cbar.set_label(z_label, fontsize=26, fontweight='bold')

    # Use logarithmic locator for ticks
    cbar.locator = LogLocator(base=10)
    cbar.update_ticks()

    # Custom formatter for tick labels
    def power_of_ten(x, pos):
        exponent = int(np.log10(x))
        sign = '-' if exponent < 0 else ''
        return rf"$10^{{{sign}{abs(exponent)}}}$"

    cbar.locator = LogLocator(base=10)
    cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(power_of_ten))

    # Set **fontsize and bold** explicitly for tick labels
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(22)
        tick.set_fontweight('bold')

    # Redraw colorbar
    cbar.draw_all()


    plt.xlim(amp_min, amp_max)
    plt.ylim(5e-9, dur_max)  # ✅ limit y-axis to 1e-6
    plt.yscale('log')

    plt.xlabel(r"$\mathbf{V_p\ (V)\ }$", fontsize=24)
    plt.ylabel(r"$\mathbf{t_{pw}\ (s)\ }$", fontsize=24)


    #plt.xticks(fontsize=22, fontweight='bold')
    plt.xticks(np.arange(amp_min, amp_max , 0.1), fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    #plt.title(title, fontsize=16, fontweight='bold')


    plt.tight_layout()
    plt.savefig(f"/home/lavanya/Documents/4/{output_name}.png", dpi=300)
    plt.show()

# ✅ Different sigma values for each plot
plot_contour(pd.read_excel(mean_file), title="", z_label="Mean,μ (Ω)", output_name="mean_contour", sigma=0)
plot_contour(pd.read_excel(std_file), title="", z_label="Std,σ (Ω)", output_name="std_contour", sigma=0)
plot_contour(pd.read_excel(cv_file), title="", z_label="Coeff. of Variance (σ/μ)", output_name="cv_contour", sigma=0)