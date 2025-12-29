import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import brentq
import numpy as np
from sklearn.metrics import max_error

# --- Load and process resistance data ---
print("Loading resistance data...")
res_df = pd.read_excel("/home/lavanya/Documents/1.9V-1us 1million data.xlsx")
#res_df = pd.read_csv("/home/lavanya/Documents/1.99V,2.73e-7.csv")
resistance_vals = pd.concat([res_df[col] for col in res_df.columns], ignore_index=True).dropna().values
resistance_log = np.log10(resistance_vals)
print(f"Loaded {len(resistance_log)} resistance values (log10 scale)")

# --- STEP 1: Histogram ---
print("STEP 1: Creating histogram...")
num_bins = 50
counts, bin_edges = np.histogram(resistance_log, bins=num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

# --- STEP 2: Calculate PDF from histogram ---
print("STEP 2: Computing PDF from histogram...")
pdf = counts / (len(resistance_log) * bin_width)  # PDF = counts / (total_samples * bin_width)


# --- STEP 3: Define Gaussmod model (Origin) ---
def guessmod(x, y0, A, xc, w, t0):
    z = (x - xc)/w - w/t0
    return y0 + (A/t0)*np.exp(0.5*(w/t0)**2 - (x - xc)/t0) * (erf(z/np.sqrt(2.0)) + 1.0)/2.0
# --- STEP 4: Fit Gaussmod to raw counts ---
print("Fitting guessmod model to histogram counts...")
# initial guess: [y0, A, xc, w, t0]
y0_init = counts.min()
A_init  = counts.max() - counts.min()
xc_init = bin_centers[np.argmax(counts)]
w_init  = (bin_edges[-1] - bin_edges[0]) / 10.0   # some width
t0_init = w_init                                  # same order as w

p0 = [y0_init, A_init, xc_init, w_init, t0_init]

popt_guessmod, pcov = curve_fit(
    guessmod,
    bin_centers,
    counts,
    p0=p0,
    maxfev=10000
)
y0_opt, A_opt, xc_opt, w_opt, t0_opt = popt_guessmod
# --- Compute R-squared ---
counts_fit_at_bins = guessmod(bin_centers, y0_opt, A_opt, xc_opt, w_opt, t0_opt)
ss_res = np.sum((counts - counts_fit_at_bins)**2)
ss_tot = np.sum((counts - np.mean(counts))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nR-squared for guessmod fit: {r_squared:.6f}")
# --- STEP 5: Plot PDF + Fit ---
x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
# fitted *counts* curve
counts_fit = guessmod(x_fit, y0_opt, A_opt, xc_opt, w_opt, t0_opt)
# normalize to PDF scale (like before)
pdf_fit = counts_fit / (len(resistance_log) * bin_width)

# --- Using your fitted guessmod parameters ---
print("=== THRESHOLD CALCULATION ===")
print("Using fitted guessmod parameters:")
print(f"y0 = {y0_opt:.2f}, A = {A_opt:.2f}, xc = {xc_opt:.4f}, "
      f"w = {w_opt:.4f}, t0 = {t0_opt:.4f}")

# --- Define fitted Gaussmod function ---
def gaussmod_fitted(x):
    # this is actually guessmod with fitted parameters
    return guessmod(x, y0_opt, A_opt, xc_opt, w_opt, t0_opt)

# --- Define x-range from your data ---
x_min = bin_edges[0]
x_max = bin_edges[-1]
print(f"x-range: {x_min:.3f} to {x_max:.3f}")

# --- Calculate total area under fitted curve ---
total_area, err = quad(gaussmod_fitted, x_min, x_max)
print(f"Total area under fitted curve: {total_area:.2f} (±{err:.2f})")

# --- Calculate thresholds ---
bsl = 2048  # Number of quantization levels (2^11)
target_area = total_area / bsl
print(f"\nComputing {bsl} equal-area thresholds...")
print(f"Target area per partition: {target_area:.2f}")

def find_threshold(i):
    """Find x where cumulative area from x_min = i * target_area"""
    def objective(x):
        area, _ = quad(gaussmod_fitted, x_min, x)
        return area - (i * target_area)
    return brentq(objective, x_min, x_max)

# Compute thresholds
print("Computing thresholds:", end=" ")
thresholds = [x_min]
for i in range(1, bsl):
    if i % 128 == 0:
        print(".", end="", flush=True)
    thresh = find_threshold(i)
    thresholds.append(thresh)
thresholds.append(x_max)
thresholds = np.array(thresholds)
print(f"\n✅ Computed {len(thresholds)} thresholds!")
# --- Results & Verification ---
print(f"\nThreshold Results:")
print(f"First 5:  {thresholds[:5].round(4)}")
print(f"Last 5:   {thresholds[-5:].round(4)}")
print(f"Min spacing: {np.min(np.diff(thresholds)):.4f}")
print(f"Max spacing: {np.max(np.diff(thresholds)):.4f}")

# Verify equal partitioning at sample points
print("\nVerification (sample cumulative areas):")
for i in [0, 64, 128, 256, 512, 768, 1023]:
    area, _ = quad(gaussmod_fitted, x_min, thresholds[i])
    frac = area / total_area
    print(f"  Partition {i:3d}: area={area:.1f} ({frac:.4f})")

# --- Plot thresholds on fitted curve ---
x_plot = np.linspace(x_min, x_max, 2000)
y_plot = gaussmod_fitted(x_plot)

plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_plot, 'r-', linewidth=5, label='Fitted GaussAmp')
#plt.bar(bin_centers, counts, width=bin_width*0.6, alpha=0.3, label='Histogram', color='gray')

# Plot only a subset of thresholds as markers
step = max(1, bsl // 20)
sample_thr = thresholds[1:-1:step]
plt.scatter(sample_thr,
            gaussmod_fitted(sample_thr),
            c='cyan', s=200, edgecolors='k', label='Sample thresholds')

# Set x-axis limits
plt.xlim(5, 8)
# --- ADD: dotted red line in the middle of the visible plot ---
x_mid = 6.14
plt.axvline(x_mid, color='red', linestyle=':', linewidth=5)

# Histogram with different colors left/right of dotted line
colors = ['lightpink' if (bin_centers[i] < x_mid) else 'lightgreen'
          for i in range(len(bin_centers))]

plt.bar(bin_centers, counts, width=bin_width*0.6, alpha=0.6, color=colors)

# --- Define axis limits and box ---
x_left, x_right = 5.3, 8
y_bottom, y_top = -2e3, y_plot.max()*1.05

plt.xlim(x_left, x_right)
plt.ylim(y_bottom, y_top)

# --- Draw all four sides manually for box ---
#plt.axvline(x_left, color='black', linewidth=2)    # left y-axis
#plt.axvline(x_right, color='black', linewidth=3)   # right y-axis
#plt.axhline(y_bottom, color='black', linewidth=3)  # bottom x-axis
#plt.axhline(y_top, color='black', linewidth=2)     # top x-axis


# --- REMOVE axis numeric values only ---
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Remove extra whitespace
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()




