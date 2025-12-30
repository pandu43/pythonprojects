import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.stats import entropy
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
#                     PUF DESIGN CLASS
# ==========================================================
class PUFDesign:
    def __init__(self, crp_bit, excel_path, usecols, skiprows=None, log_transform=True):
        self.crp_bit = crp_bit

        df = pd.read_excel(
            excel_path,
            usecols=usecols,
            skiprows=skiprows if skiprows else [],
            engine="openpyxl"
        )

        data = df.values.flatten().astype(np.float32)

        if log_transform:
            data = np.log10(data + 1e-9)
            print("✔ Log10 transformation applied")

        np.random.shuffle(data)
        self.data = data

        print(f"✔ Loaded {len(data)} log-resistance samples")
        print(f"✔ Log-R range: [{data.min():.3f}, {data.max():.3f}]")

        del df
        gc.collect()

    @staticmethod
    def get_response(a1, a2):
        return (a1 > a2).astype(np.uint8)

    def process_challenge_batch(self, indices, challenge_len):
        out = []
        data = self.data

        for idx in indices:
            r1 = np.random.choice(data, challenge_len, replace=False)
            remaining = np.setdiff1d(data, r1)
            r2 = np.random.choice(remaining, challenge_len, replace=False)

            response_bits = self.get_response(r1, r2)
            out.append(response_bits)

        return out

    def generate_crps(self, challenge_len):
        total = 2 ** self.crp_bit
        batch_size = 200000
        num_workers = min(cpu_count(), 8)

        responses = []

        with Pool(num_workers) as pool:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                indices = np.array_split(np.arange(start, end), num_workers)
                func = partial(self.process_challenge_batch, challenge_len=challenge_len)
                batch = pool.map(func, indices)
                responses.extend([b for sub in batch for b in sub])

        return responses


# ==========================================================
#                SHANNON ENTROPY
# ==========================================================
def shannon_entropy(bits):
    values, counts = np.unique(bits, return_counts=True)
    return entropy(counts / counts.sum(), base=2)


# ==========================================================
def bitstream_entropy(bits):
    """Measured Shannon entropy from bitstream"""
    values, counts = np.unique(bits, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)


def theoretical_entropy_from_p(p1):
    """Theoretical entropy H(p)"""
    if p1 == 0 or p1 == 1:
        return 0.0
    return entropy([p1, 1 - p1], base=2)

# ==========================================================
#                       MAIN (DIAGNOSTIC)
# ==========================================================
if __name__ == "__main__":

    excel_path = "/home/lavanya/Documents/1.9V-1us 1million data.xlsx"
    challenge_len = 8
    crp_bits = 10        # 1024 CRPs
    runs = 1

    bsl_list = [2 ** i for i in range(3, 11)]  # 8 → 1024

    entropy_measured = []
    entropy_theoretical = []
    p1_list = []

    np.random.seed(0)

    puf = PUFDesign(
        crp_bit=crp_bits,
        excel_path=excel_path,
        usecols=[0],
        skiprows=[0],
        log_transform=True
    )

    responses = puf.generate_crps(challenge_len)

    # ---- Convert response strings to bit rows once ----
    response_bits = []
    for r in responses:
        response_bits.extend([int(b) for b in r])

    response_bits = np.array(response_bits, dtype=np.uint8)

    # ---- Loop over BSL ----
    for bsl in bsl_list:

        bits = response_bits[:bsl]

        # ---- p(1) ----
        p1 = np.mean(bits)
        p1_list.append(p1)

        # ---- Entropies ----
        H_meas = bitstream_entropy(bits)
        H_theory = theoretical_entropy_from_p(p1)

        entropy_measured.append(H_meas)
        entropy_theoretical.append(H_theory)

        print(f"BSL={bsl:4d} | p(1)={p1:.4f} | H_meas={H_meas:.4f} | H_theory={H_theory:.4f}")

    # ===================== PLOTS =====================

    # ---- Plot p(1) vs BSL ----
    plt.figure(figsize=(8, 5))
    plt.plot(bsl_list, p1_list, "o-", linewidth=2)
    plt.axhline(0.5, linestyle="--", color="black")
    plt.xscale("log", base=2)
    plt.xticks(bsl_list, [f"$2^{i}$" for i in range(3, 11)])
    plt.xlabel("Bitstream Length (BSL)")
    plt.ylabel("p(1)")
    plt.title("Probability of '1' vs Bitstream Length")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # ---- Plot entropy comparison ----
    plt.figure(figsize=(8, 5))
    plt.plot(bsl_list, entropy_measured, "o-", label="Measured entropy", linewidth=2)
    plt.plot(bsl_list, entropy_theoretical, "s--", label="Theoretical H(p)", linewidth=2)
    plt.xscale("log", base=2)
    plt.xticks(bsl_list, [f"$2^{i}$" for i in range(3, 11)])
    plt.ylim(0, 1.05)
    plt.xlabel("Bitstream Length (BSL)")
    plt.ylabel("Entropy (bits/bit)")
    plt.title("Measured vs Theoretical Entropy")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()