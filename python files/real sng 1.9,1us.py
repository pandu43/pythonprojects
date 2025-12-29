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
#res_df = pd.read_excel("/home/lavanya/Documents/1.9V-1us 1million data.xlsx")
res_df = pd.read_csv("/home/lavanya/Documents/1.67V,1.37e-5 sec.csv")
resistance_vals = pd.concat([res_df[col] for col in res_df.columns], ignore_index=True).dropna().values
resistance_log = np.log10(resistance_vals)
print(f"Loaded {len(resistance_log)} resistance values (log10 scale)")

# --- STEP 1: Histogram ---
print("STEP 1: Creating histogram...")
num_bins = 50
counts, bin_edges = np.histogram(resistance_log, bins=num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

plt.figure(figsize=(10, 6))
plt.bar(bin_centers, counts, width=bin_width, alpha=0.7, edgecolor='black', label='Histogram (Counts)')
plt.xlabel('Log10(Resistance)')
plt.ylabel('Counts')
plt.title('STEP 1: Resistance Histogram')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- STEP 2: Calculate PDF from histogram ---
print("STEP 2: Computing PDF from histogram...")
pdf = counts / (len(resistance_log) * bin_width)  # PDF = counts / (total_samples * bin_width)

plt.figure(figsize=(10, 6))
plt.bar(bin_centers, pdf, width=bin_width, alpha=0.7, edgecolor='black', label='PDF (Probability Density)')
plt.xlabel('Log10(Resistance)')
plt.ylabel('Probability Density')
plt.title('STEP 2: PDF from Histogram\n(Probability vs Resistance)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"PDF range: {pdf.min():.3e} to {pdf.max():.3e}")
print(f"Integral of PDF should ≈ 1: {np.trapz(pdf, bin_centers):.4f}")

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

plt.figure(figsize=(12, 6))
plt.bar(bin_centers, pdf, width=bin_width*0.8, alpha=0.6,
        label='PDF from Histogram', edgecolor='black')
plt.plot(x_fit, pdf_fit, 'r-', linewidth=3, label='guessmod Fit to PDF')
plt.xlabel('Log10(Resistance)')
plt.ylabel('Probability Density')
plt.title('STEP 3: PDF + guessmod Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

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
for i in [0, 64, 128, 256, 512, 768, 1024,2048]:
    area, _ = quad(gaussmod_fitted, x_min, thresholds[i])
    frac = area / total_area
    print(f"  Partition {i:3d}: area={area:.1f} ({frac:.4f})")

# --- Plot thresholds on fitted curve ---
x_plot = np.linspace(x_min, x_max, 2000)
y_plot = gaussmod_fitted(x_plot)

plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_plot, 'r-', linewidth=3, label='Fitted GaussAmp')
plt.bar(bin_centers, counts, width=bin_width*0.6, alpha=0.3, label='Histogram', color='gray')

# Plot only a subset of thresholds as markers
step = max(1, bsl // 20)
sample_thr = thresholds[1:-1:step]
plt.scatter(sample_thr,
            gaussmod_fitted(sample_thr),
            c='cyan', s=30, edgecolors='k', label='Sample thresholds')

plt.xlabel('Log10(Resistance)')
plt.ylabel('Counts')
plt.title(f'GaussAmp Fit with sample of {bsl} equal-area thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Save thresholds ---
np.save("gaussamp_thresholds.npy", thresholds)
print(f"\n✅ Thresholds saved to 'gaussamp_thresholds.npy'")
print("Ready for real SNG inference!")
print(f"Use 'thresholds' array in your real_SNG function:")
print("def real_SNG(input_val, resistance_slice, thresholds):")
print("    idx = int(np.clip(input_val, 0, len(thresholds)-1))")
print("    return (resistance_slice <= thresholds[idx]).astype(int)")

# Assume these are already prepared elsewhere:
# resistance_log : 1D numpy array of log10(resistance) from your 1M data
# thresholds     : 1D numpy array of length 1025 (bsl+1) from GaussAmp integration
# bsl_pdf        : number of probability levels used to compute thresholds (1024 here)
bsl_pdf = 2048  # for clarity

N_res = len(resistance_log)

def value_to_level(x, n_bits_pdf=bsl_pdf):
    # Clip x to [-1, 1] just in case
    x_clamped = np.clip(x, -1.0, 1.0)
    p = (x_clamped + 1.0) * 0.5             # probability in [0,1]
    idx = int(np.floor(p * n_bits_pdf))     # integer in [0, n_bits_pdf]
    if idx == n_bits_pdf:
        idx = n_bits_pdf - 1
    return idx

def real_SNG_scalar(x, BSL, resistance_log, thresholds):

    N = len(resistance_log)
    level_idx = value_to_level(x, n_bits_pdf=bsl_pdf)
    thr = thresholds[level_idx]  # threshold in log10(R)
    # Take BSL consecutive resistance samples (wrap-around for randomness)
    # In practice you might want to offset by a random start index.
    start = np.random.randint(0, N)
    end = start + BSL
    if end <= N:
        slice_log = resistance_log[start:end]
    else:
        # wrap
        slice_log = np.concatenate((resistance_log[start:], resistance_log[:end - N]))
    # Convert resistance log values to stochastic bits:
    # 1 if log10(R) <= thr, else 0
    bits = (slice_log <= thr).astype(np.int32)
    return bits

def real_SNG_batch(x_vec, BSL, resistance_log, thresholds):
    n = len(x_vec)
    streams = np.zeros((n, BSL), dtype=np.int32)
    for i, xv in enumerate(x_vec):
        streams[i, :] = real_SNG_scalar(xv, BSL, resistance_log, thresholds)
    return streams


# Sanity check of real SNG: p_real vs p_ideal
xs = np.linspace(-0.99, 1, 21)   # test inputs
BSL_list = [2 ** i for i in range(3, 11)]  # [8,16,32,64,128,256,512,1024]
num_streams = 100              # average over several runs for each x

p_real_list = []
p_ideal_list = []
mean_err_per_BSL = []
max_err_per_BSL = []
rmse_per_BSL = []
rrmse_per_BSL = []

for BSL_test in BSL_list:
    errs = []
    p_ideal_all = []
    for x in xs:
      probs = []
      for _ in range(num_streams):
         bits = real_SNG_scalar(x, BSL_test, resistance_log, thresholds)
         probs.append(bits.mean())
      p_real = np.mean(probs)
      p_ideal = (x + 1.0) / 2.0
      errs.append(abs(p_real - p_ideal))
      p_ideal_all.append(p_ideal)
      print(f"x={x: .2f}, p_real={p_real:.3f}, p_ideal={p_ideal:.3f}")

    p_real_list.append(p_real)
    p_ideal_list.append(p_ideal)
    mean_err_per_BSL.append(np.mean(errs))
    max_err_per_BSL.append(np.max(errs))
    print(f"BSL={BSL_test}, mean |error|={mean_err_per_BSL[-1]:.4f},max error={max(errs)}")

    # ----------- RMSE for this BSL (over 21 x-values) ------------
    rmse = np.sqrt(np.mean(np.array(errs) ** 2))
    rmse_per_BSL.append(rmse)
    print(f"RMSE for BSL {BSL_test}: {rmse:.6f}")

    # ---------------------------------------------------------
    # RELATIVE RMSE (over 21 x values)
    # ---------------------------------------------------------
    errs = np.array(errs)
    p_ideal_all = np.array(p_ideal_all)

    rel_errs = errs / p_ideal_all  # relative error for each x
    rrmse = np.sqrt(np.mean(rel_errs ** 2))
    rrmse_per_BSL.append(rrmse)
    print(f"RRMSE for BSL {BSL_test}: {rrmse:.6f}\n")

# ============================================================
#  SAVE ALL ERROR METRICS INTO ONE EXCEL SHEET
# ============================================================
df_out = pd.DataFrame({
    "BSL": BSL_list,
    "Mean |p_real - p_ideal|": mean_err_per_BSL,
    "Max |p_real - p_ideal|": max_err_per_BSL,
    "RMSE": rmse_per_BSL,
    "RRMSE": rrmse_per_BSL
})

output_file = "BSL_Error_Metrics1.9V,1us.xlsx"
df_out.to_excel(output_file, index=False)
print(f"\nExcel saved: {output_file}")

plt.figure(figsize=(5,5))
plt.plot(BSL_list, mean_err_per_BSL, 'o-')
plt.xscale("log", base=2)
plt.yscale("log", base=10)
plt.xlabel('BSL')
plt.ylabel('mean |p_real - p_ideal| over x')
plt.title('SNG accuracy vs bitstream length')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------- PLOT max error ----------
plt.figure(figsize=(5,5))
plt.plot(BSL_list, max_err_per_BSL, 'o-')
plt.xscale("log", base=2)
plt.yscale("log", base=10)
plt.xlabel('BSL')
plt.ylabel('Max absolute error')
plt.title('Max |p_real - p_ideal| vs BSL')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- Plot RMSE vs BSL ----------
plt.figure(figsize=(5,5))
plt.plot(BSL_list, rmse_per_BSL, 'o-', linewidth=2)
plt.xscale("log", base=2)
plt.yscale("log")     # RMSE shrinks fast, log-scale is clearer
plt.xlabel("Bitstream Length (BSL)")
plt.ylabel("RMSE (absolute error)")
plt.title("RMSE of p_real vs p_ideal over 21 input values")
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# PLOT: RRMSE vs BSL
# ---------------------------------------------------------
plt.figure(figsize=(5,5))
plt.plot(BSL_list, rrmse_per_BSL, 'o-', linewidth=2)
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Bitstream Length (BSL)")
plt.ylabel("Relative RMSE")
plt.title("Relative RMSE of p_real vs p_ideal")
plt.grid(True)
plt.tight_layout()
plt.show()

###############################################################################
# Real-SNG-based SC inference (MNIST MLP)
###############################################################################
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################ Btanh and helper functions #######################

def Btanh(r, t):
    Smax = r - 1
    Shalf = r / 2
    S = Shalf
    n, bsl = t.shape
    output_bitstream = []

    for i in range(bsl):
        V = 2 * np.sum(t[:, i]) - n
        S = min(max(S + V, 0), Smax)
        output_bitstream.append(1 if S > Shalf else 0)

    result = np.array(output_bitstream)
    return (2 * np.sum(result) / len(result)) - 1

def find_r(n, s):
    q = 1.835 * ((2 * n) ** (-0.5552))
    r_prime = ((2 * (1 - s) * (n - 1)) / (s * (1 - q))) + 2 * n
    r = 2 * np.round(r_prime / 2).astype(np.int64)
    return r

######################## Matrix multiplication with real SNG ##################
def matrixMultiplication_real(a, b, resistance_log, thresholds, n, iterations=1):
    """
    a : 1D np.array (inputs + bias)
    b : weights (1D or 2D), already scaled by *2 outside
    n : bitstream length (BSL)
    """
    # x streams: shape (iterations, len(a), n)
    x = np.zeros((iterations, a.shape[0], n), dtype=int)
    for it in range(iterations):
        for i in range(a.shape[0]):
            x[it, i, :] = real_SNG_scalar(float(a[i]), n, resistance_log, thresholds)

    if b.ndim == 1:
        # ----- Single output neuron (same r as ideal code) -----
        c = 0.0
        r = find_r(b.shape[0], 2)          # changed: was find_r(input_dim, 2)
        y = np.zeros((iterations, b.shape[0], n), dtype=int)

        for it in range(iterations):
            for i in range(b.shape[0]):
                y[it, i, :] = real_SNG_scalar(float(b[i]), n, resistance_log, thresholds)

        for it in range(iterations):
            t = np.logical_not(np.logical_xor(x[it], y[it])).astype(int)
            c += Btanh(r, t)

    else:
        # ----- Multiple output neurons: b has shape (out_dim, in_dim) -----
        out_dim, input_dim = b.shape
        c = np.zeros(out_dim, dtype=float)
        r = find_r(out_dim, 2)             # changed: was find_r(input_dim, 2)
        y = np.zeros((iterations, out_dim, input_dim, n), dtype=int)

        for it in range(iterations):
            for j in range(out_dim):
                for i in range(input_dim):
                    y[it, j, i, :] = real_SNG_scalar(float(b[j, i]), n,
                                                     resistance_log, thresholds)

        for it in range(iterations):
            for j in range(out_dim):
                t = np.logical_not(np.logical_xor(x[it], y[it, j])).astype(int)
                c[j] += Btanh(r, t)

    return c / iterations

############################ MLP with real SNG ################################

def MLPSN_real(x, modelDN, resistance_log, thresholds, n):
    """
    Same structure and scaling as MLPSN (ideal SNG).
    """
    fc1 = modelDN['fc1'] * 2.0    # keep exactly as ideal
    fc2 = modelDN['fc2'] * 2.0

    x = np.append(x, 1.0)         # bias
    x = matrixMultiplication_real(x, fc1, resistance_log, thresholds, n=n)

    x = np.append(x, 1.0)         # bias for second layer
    x = matrixMultiplication_real(x, fc2, resistance_log, thresholds, n=n)
    return x

############################### Inference loop ################################
def inference_real(batch, modelDN, resistance_log, thresholds, n):
    """
    Use the same input preprocessing as in the ideal SNG inference.
    If ideal MLPSN uses raw img.view(14*14), do the same here.
    """
    correct = 0
    total = 0
    for img, label in batch:
        x = np.array(img.view(14 * 14))    # no extra 2*x-1 if ideal code doesn’t use it
        out = MLPSN_real(x, modelDN, resistance_log, thresholds, n=n)
        predicted = np.argmax(out)
        total += 1
        correct += int(predicted == label.item())
    return correct, total

############################ Load SC-trained model ###########################

BSL_default = 2048  # used during original training
probs = None
model = torch.load('models-SC/model_BSL_2048_epoch_100.pth')

# concatenate weights and biases as in ideal SNG code
model['fc1'] = torch.cat((model['fc1.weight'],model['fc1.bias'].reshape(-1, 1)), dim=1)
model['fc2'] = torch.cat((model['fc2.weight'],model['fc2.bias'].reshape(-1, 1)), dim=1)

modelDN = {k: v.detach().numpy() for k, v in model.items()}

############################ Load MNIST test set #############################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(14, antialias=True),
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

num_cores = multiprocessing.cpu_count()
chunk_size = len(test_dataset) // num_cores
chunks = torch.utils.data.random_split(
    test_dataset,
    [chunk_size] * (num_cores - 1) + [len(test_dataset) - chunk_size * (num_cores - 1)]
)
chunks = [DataLoader(chunk, shuffle=False) for chunk in chunks]

############################ BSL sweep with real SNG #########################

BSL_list = [2 ** i for i in range(3, 11)]  # 8 → 2048
acc_list = []

for BSL in BSL_list:
    print(f"\nRunning inference with real SNG, BSL={BSL} ...")
    f = partial(inference_real,
                modelDN=modelDN,
                resistance_log=resistance_log,
                thresholds=thresholds,
                n=BSL)
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(f, chunks)

    correct, total = zip(*results)
    acc = sum(correct) / sum(total)
    acc_list.append(acc)
    print(f"BSL={BSL}, Accuracy(real SNG)={acc:.4f}")

# Save and plot
df_acc = pd.DataFrame({'BSL': BSL_list, 'Accuracy_realSNG': acc_list})
df_acc.to_csv('accuracy_vs_BSL_realSNG1.9,1us.csv', index=False)
print("CSV file saved as 'accuracy_vs_BSL_realSNG1.9V,1us.csv'")

plt.figure(figsize=(7, 5))
plt.plot(BSL_list, acc_list, marker='o')
plt.xscale("log", base=2)
plt.xlabel("Bitstream Length (BSL)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs BSL for SC MLP (Real SNG)")
plt.grid(True)
plt.tight_layout()
plt.show()