import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import brentq
import numpy as np
# --- Load and process resistance data ---
print("Loading resistance data...")
res_df = pd.read_excel("/home/lavanya/Documents/1.9V-1us 1million data.xlsx")
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

# --- STEP 3: Define GaussAmp model (Origin) ---
def gaussamp(x, y0, A, xc, w):
    return y0 + A * np.exp(-0.5 * ((x - xc)/w)**2)

# --- STEP 4: Fit GaussAmp to raw counts ---
print("Fitting GaussAmp model to histogram counts...")
initial_guess = [min(counts), max(counts), np.mean(bin_centers), np.std(bin_centers)]
popt_gaussamp, pcov = curve_fit(gaussamp, bin_centers, counts, p0=initial_guess, maxfev=5000)
y0_opt, A_opt, xc_opt, w_opt = popt_gaussamp
print(f"Fitted parameters: y0={y0_opt:.2f}, A={A_opt:.2f}, xc={xc_opt:.3f}, w={w_opt:.3f}")

# --- STEP 5: Plot PDF + Fit ---
x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
pdf_fit = gaussamp(x_fit, y0_opt, A_opt, xc_opt, w_opt) / (len(resistance_log)*bin_width)  # normalized fit for PDF scaleq
plt.figure(figsize=(12, 6))
plt.bar(bin_centers, pdf, width=bin_width*0.8, alpha=0.6, label='PDF from Histogram', edgecolor='black')
plt.plot(x_fit, pdf_fit, 'r-', linewidth=3, label='GaussAmp Fit to PDF')
plt.xlabel('Log10(Resistance)')
plt.ylabel('Probability Density')
plt.title('STEP 3: PDF + GaussAmp Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- Using your fitted GaussAmp parameters ---
# From the robust fitting above: y0_opt, A_opt, xc_opt, w_opt

print("=== THRESHOLD CALCULATION ===")
print(f"Using fitted GaussAmp parameters:")
print(f"y0 = {y0_opt:.2f}, A = {A_opt:.2f}, xc = {xc_opt:.4f}, w = {w_opt:.4f}")

# --- Define fitted GaussAmp function ---
def gaussamp_fitted(x):
    """Your fitted GaussAmp function"""
    return y0_opt + A_opt * np.exp(-0.5 * ((x - xc_opt) / w_opt)**2)

# --- Define x-range from your data ---
x_min = bin_edges[0]
x_max = bin_edges[-1]
print(f"x-range: {x_min:.3f} to {x_max:.3f}")

# --- Calculate total area under fitted curve ---
total_area, err = quad(gaussamp_fitted, x_min, x_max)
print(f"Total area under fitted curve: {total_area:.2f} (±{err:.2f})")

# --- Calculate thresholds ---
bsl = 2048  # Number of quantization levels (2^11)
target_area = total_area / bsl
print(f"\nComputing {bsl} equal-area thresholds...")
print(f"Target area per partition: {target_area:.2f}")

def find_threshold(i):
    """Find x where cumulative area from x_min = i * target_area"""
    def objective(x):
        area, _ = quad(gaussamp_fitted, x_min, x)
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
for i in [0, 64, 128, 256, 512, 768, 1023,2048]:
    area, _ = quad(gaussamp_fitted, x_min, thresholds[i])
    frac = area / total_area
    print(f"  Partition {i:3d}: area={area:.1f} ({frac:.4f})")

# --- Plot thresholds on fitted curve ---
x_plot = np.linspace(x_min, x_max, 2000)
y_plot = gaussamp_fitted(x_plot)

plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_plot, 'r-', linewidth=3, label='Fitted GaussAmp')
plt.bar(bin_centers, counts, width=bin_width*0.6, alpha=0.3, label='Histogram', color='gray')

# Plot only a subset of thresholds as markers
step = max(1, bsl // 20)
sample_thr = thresholds[1:-1:step]
plt.scatter(sample_thr,
            gaussamp_fitted(sample_thr),
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
for BSL_test in BSL_list:
    errs = []
    for x in xs:
      probs = []
      for _ in range(num_streams):
         bits = real_SNG_scalar(x, BSL_test, resistance_log, thresholds)
         probs.append(bits.mean())
      p_real = np.mean(probs)
      p_ideal = (x + 1.0) / 2.0
      errs.append(abs(p_real - p_ideal)/p_ideal)
      print(f"x={x: .2f}, p_real={p_real:.3f}, p_ideal={p_ideal:.3f}")
    p_real_list.append(p_real)
    p_ideal_list.append(p_ideal)
    mean_err_per_BSL.append(np.mean(errs))
    print(f"BSL={BSL_test}, mean |error|={mean_err_per_BSL[-1]:.4f}")


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
###############################################################################
# Btanh and helper functions
###############################################################################

MAC_GAIN_CORR = 1.0   # initialized; will be updated after calibration

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
    val = (2 * np.sum(result) / len(result)) - 1
    # gain correction to compensate systematic attenuation in real MAC
    return val / MAC_GAIN_CORR   # uses the global variable

def find_r(n, s):
    q = 1.835 * ((2 * n) ** (-0.5552))
    r_prime = ((2 * (1 - s) * (n - 1)) / (s * (1 - q))) + 2 * n
    r = 2 * np.round(r_prime / 2).astype(np.int64)
    return r


def SNG(x, n, probs=None, iterations=1):
    v = int(((x + 1) * n) // 2)  # number of 1s
    y = np.zeros((iterations, n), dtype=int)
    for i in range(iterations):
        y[i][np.random.choice(np.arange(n), v, replace=False)] = 1
    return y

###############################################################################
# Real MAC and gain estimation
###############################################################################
def matrixMultiplication(a, b, probs, n, iterations=1):
    x = np.zeros((iterations, a.shape[0], n), dtype=int)
    for i in range(a.shape[0]):
        x[:, i] = SNG(a[i], n, probs, iterations)

    if b.ndim == 1:
        c = 0
        r = find_r(b.shape[0], 2)
        y = np.zeros((iterations, b.shape[0], n), dtype=int)
        for i in range(b.shape[0]):
            y[:, i] = SNG(b[i], n, probs, iterations)

        for i in range(iterations):
            t = np.logical_not(np.logical_xor(x[i], y[i])).astype(int)
            c += Btanh(r, t)
    else:
        c = np.zeros(b.shape[0])
        r = find_r(b.shape[0], 2)
        y = np.zeros((iterations, b.shape[0], b.shape[1], n), dtype=int)
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                y[:, i, j] = SNG(b[i, j], n, probs, iterations)

        for i in range(iterations):
            for j in range(b.shape[0]):
                t = np.logical_not(np.logical_xor(x[i], y[i][j])).astype(int)
                c[j] += Btanh(r, t)

    return c / iterations
def matrixMultiplication_real(a, b, resistance_log, thresholds, n, iterations=1):
    """
    a : 1D np.array (inputs + bias)
    b : weights (1D or 2D), already scaled by *2 outside
    n : bitstream length (BSL)
    """
    x = np.zeros((iterations, a.shape[0], n), dtype=int)
    for it in range(iterations):
        for i in range(a.shape[0]):
            x[it, i, :] = real_SNG_scalar(float(a[i]), n, resistance_log, thresholds)

    if b.ndim == 1:
        c = 0.0
        r = find_r(b.shape[0], 2)
        y = np.zeros((iterations, b.shape[0], n), dtype=int)

        for it in range(iterations):
            for i in range(b.shape[0]):
                y[it, i, :] = real_SNG_scalar(float(b[i]), n, resistance_log, thresholds)

        for it in range(iterations):
            t = np.logical_not(np.logical_xor(x[it], y[it])).astype(int)
            c += Btanh(r, t)

    else:
        out_dim, input_dim = b.shape
        c = np.zeros(out_dim, dtype=float)
        r = find_r(out_dim, 2)
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

def estimate_mac_gain(num_tests=200, vec_len=50, BSL=2048):
    """
    Requires: ideal matrixMultiplication (from your ideal SNG code),
              matrixMultiplication_real,
              resistance_log, thresholds in scope.
    """
    errs = []
    ideals = []
    for _ in range(num_tests):
        a = np.random.uniform(-1, 1, vec_len)
        b = np.random.uniform(-1, 1, vec_len)
        ideal = matrixMultiplication(a, b, probs=None, n=BSL, iterations=5)
        real  = matrixMultiplication_real(a, b, resistance_log, thresholds,
                                          n=BSL, iterations=5)
        ideals.append(ideal)
        errs.append(real)
    ideals = np.array(ideals)
    errs   = np.array(errs)
    k = (ideals * errs).sum() / (ideals * ideals).sum()
    return k

# IMPORTANT: call this only AFTER:
# - resistance_log and thresholds are defined
# - matrixMultiplication (ideal) is imported/defined
print("Estimating MAC gain...")
k_mac = estimate_mac_gain()
print("Estimated real/ideal gain:", k_mac)

# update the global gain correction for Btanh
MAC_GAIN_CORR = k_mac

###############################################################################
# MLP with real SNG
###############################################################################

def MLPSN_real(x, modelDN, resistance_log, thresholds, n):
    fc1 = modelDN['fc1'] * 2.0
    fc2 = modelDN['fc2'] * 2.0

    x = np.append(x, 1.0)
    x = matrixMultiplication_real(x, fc1, resistance_log, thresholds, n=n)

    x = np.append(x, 1.0)
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
df_acc.to_csv('accuracy_vs_BSL_realSNG.csv', index=False)
print("CSV file saved as 'accuracy_vs_BSL_realSNG.csv'")

plt.figure(figsize=(7, 5))
plt.plot(BSL_list, acc_list, marker='o')
plt.xscale("log", base=2)
plt.xlabel("Bitstream Length (BSL)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs BSL for SC MLP (Real SNG)")
plt.grid(True)
plt.tight_layout()
plt.show()