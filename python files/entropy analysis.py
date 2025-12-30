import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.stats import entropy
from datetime import datetime
import warnings
import math

warnings.filterwarnings("ignore")


# ==========================================================
#                     PUF DESIGN CLASS (NO CSV, BSL PLOT)
# ==========================================================
class PUFDesign:
    def __init__(self, crp_bit, file_path, usecols, skiprows=None, log_transform=True):
        self.file_path = file_path
        self.usecols = usecols
        self.skiprows = skiprows if skiprows is not None else []
        self.crp_bit = crp_bit

        # Load Excel data
        df = pd.read_excel(file_path, usecols=usecols, skiprows=self.skiprows, engine="openpyxl")
        data = df.values.flatten().astype(np.float32)

        # Log transform
        if log_transform:
            eps = 1e-9
            data = np.log10(data + eps)
            print("✔ Log10 transformation applied")

        # Shuffle
        np.random.shuffle(data)
        self.data = data
        self.data_len = len(data)

        print(f"✔ Loaded {self.data_len} log-resistance samples")
        print(f"✔ Log-R range: [{data.min():.3f}, {data.max():.3f}]")

        del df
        gc.collect()

    @staticmethod
    def get_response(a1, a2):
        """PUF response bit: compare log-resistances"""
        return (a1 > a2).astype(np.uint8)

    def process_challenge_batch(self, indices, challenge_len):
        """Generate CRPs matching second code's method (no file saving)"""
        results = []
        data = self.data

        for idx in indices:
            a1 = np.random.choice(data, challenge_len, replace=False)
            remaining = np.setdiff1d(data, a1)
            a2 = np.random.choice(remaining, challenge_len, replace=False)

            challenge_bits = np.array(
                list(map(int, np.binary_repr(idx, width=challenge_len))),
                dtype=np.uint8
            )
            response_bits = self.get_response(a1, a2)

            results.append((challenge_bits, response_bits))
        return results

    def generate_crps(self, challenge_len):
        """Generate CRPs in memory (no CSV files)"""
        total_challenges = 2 ** self.crp_bit
        batch_size = 200000
        num_workers = min(cpu_count(), 8)

        print(f"✔ Generating {total_challenges} CRPs using {num_workers} cores")

        all_crps = []
        with Pool(num_workers) as pool:
            for start in range(0, total_challenges, batch_size):
                end = min(start + batch_size, total_challenges)
                indices = np.array_split(np.arange(start, end), num_workers)
                func = partial(self.process_challenge_batch, challenge_len=challenge_len)
                batch_res = pool.map(func, indices)
                batch_crps = [item for sub in batch_res for item in sub]
                all_crps.extend(batch_crps)

        # Extract response bits only
        response_rows = [resp for _, resp in all_crps]
        return np.array(response_rows, dtype=np.uint8)


# ==========================================================
#                SHANNON ENTROPY vs BSL (ORIGINAL PLOT)
# ==========================================================
def shannon_entropy(bits):
    """Entropy for bitstream of any length"""
    bits = np.asarray(bits, dtype=np.uint8)
    values, counts = np.unique(bits, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)


def flatten_bits_for_bsl(response_rows, bsl):
    """Create bitstream of exact BSL by cycling through CRPs"""
    flat = []
    i = 0
    while len(flat) < bsl:
        flat.extend(response_rows[i % len(response_rows)])
        i += 1
    return np.array(flat[:bsl], dtype=np.uint8)


# ==========================================================
#                       MAIN EXECUTION
# ==========================================================
def main_puf_entropy(challenge_len=8, crp_bits=10, runs=5, bsl_list=None):
    if bsl_list is None:
        bsl_list = [2 ** i for i in range(3, 11)]  # 8 → 1024

    excel_path = "/home/lavanya/Documents/1.9V-1us 1million data.xlsx"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    entropy_runs = []

    # Multiple runs for statistics
    for run in range(runs):
        print(f"\n===== RUN {run + 1}/{runs} =====")
        np.random.seed(run)

        # Generate CRPs (second code method, no files)
        puf = PUFDesign(
            crp_bit=crp_bits,
            excel_path=excel_path,
            usecols=[0],
            skiprows=[0],
            log_transform=True
        )

        response_rows = puf.generate_crps(challenge_len)
        print(f"✔ Generated {len(response_rows)} response bitstrings")

        # Compute entropy vs BSL (first code method)
        entropy_resp = []
        for bsl in bsl_list:
            bits = flatten_bits_for_bsl(response_rows, bsl)
            entropy_resp.append(shannon_entropy(bits))

        entropy_runs.append(entropy_resp)

        # Memory cleanup
        del puf, response_rows
        gc.collect()

    # Statistics
    entropy_runs = np.array(entropy_runs)
    entropy_mean = entropy_runs.mean(axis=0)
    entropy_std = entropy_runs.std(axis=0)

    # ==========================================================
    #                       PUBLICATION-READY PLOT
    # ==========================================================
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        bsl_list,
        entropy_mean,
        yerr=entropy_std,
        fmt='o-',
        capsize=5,
        linewidth=2.5,
        label="Response entropy (log-R PUF)",
        markersize=8
    )

    plt.xscale("log", base=2)
    plt.xticks(bsl_list, [f"$2^{i}$" for i in range(3, 11)])
    plt.ylim(0, 1.05)
    plt.xlabel("Bitstream Length (BSL)", fontsize=13, fontweight="bold")
    plt.ylabel("Shannon Entropy (bits/bit)", fontsize=13, fontweight="bold")
    plt.title("MoS₂-RRAM PUF: Entropy vs Bitstream Length", fontsize=15, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plot_file = f"PUF_Entropy_vs_BSL_C{challenge_len}R{crp_bits}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✔ Plot saved: {plot_file}")
    print(f"✔ Final entropy @ BSL=1024: {entropy_mean[-1]:.4f} ± {entropy_std[-1]:.4f}")

