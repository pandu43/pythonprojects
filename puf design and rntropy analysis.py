import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
#                     PUF DESIGN CLASS (LOG RESISTANCE)
# ==========================================================
class PUFDesign:
    def __init__(self, excel_path, col_index=0, log_transform=True):
        """Initialize PUF with log-transformed resistance data."""
        # Load resistance data
        df = pd.read_excel(excel_path, usecols=[col_index])
        self.data = df.values.flatten().astype(np.float32)

        # Apply logarithmic transformation
        if log_transform:
            # Add small epsilon to avoid log(0), then take log10
            epsilon = 1e-9
            self.data = np.log10(self.data + epsilon)
            print("Applied log10 transformation to resistance values")

        # Shuffle to remove correlation
        np.random.shuffle(self.data)
        self.data_len = len(self.data)

        print(f"Loaded {self.data_len} log-resistance samples")
        print(f"Log-resistance range: [{self.data.min():.3f}, {self.data.max():.3f}]")

        del df
        gc.collect()

    # ------------------------------------------------------
    # Resistance comparison (PUF response) - uses log values
    # ------------------------------------------------------
    @staticmethod
    def get_response(a1, a2):
        """Generate response bit: 1 if log(R1) > log(R2), else 0."""
        return (a1 > a2).astype(np.uint8)

    # ------------------------------------------------------
    # Binary → Decimal
    # ------------------------------------------------------
    @staticmethod
    def bin_to_dec(arr):
        """Convert binary array to decimal."""
        powers = 1 << np.arange(len(arr) - 1, -1, -1)
        return int(np.dot(arr, powers))

    # ------------------------------------------------------
    # Process one batch of challenges
    # ------------------------------------------------------
    def process_batch(self, indices, challenge_len):
        """Process batch using log-resistance values."""
        results = []
        data = self.data  # Already log-transformed

        for idx in indices:
            # Select two disjoint log-resistance vectors
            r1 = np.random.choice(data, challenge_len, replace=False)
            remaining = np.setdiff1d(data, r1)
            r2 = np.random.choice(remaining, challenge_len, replace=False)

            # Challenge bits from index binary representation
            challenge_bits = np.array(
                list(np.binary_repr(idx, width=challenge_len)),
                dtype=np.uint8
            )

            # Response bits from log-resistance comparison
            response_bits = self.get_response(r1, r2)

            results.append((
                ''.join(map(str, challenge_bits)),
                ''.join(map(str, response_bits)),
                self.bin_to_dec(challenge_bits),
                self.bin_to_dec(response_bits)
            ))
        return results

    # ------------------------------------------------------
    # Generate CRPs and save CSV
    # ------------------------------------------------------
    def generate_crp(self, crp_bits, challenge_len, out_csv):
        """Generate CRPs using log-resistance and save to CSV."""
        total_challenges = 2 ** crp_bits
        batch_size = 200000
        num_workers = min(cpu_count(), 8)

        print(f"Generating {total_challenges} CRPs (log-resistance) using {num_workers} cores")

        all_results = []

        with Pool(num_workers) as pool:
            for start in range(0, total_challenges, batch_size):
                end = min(start + batch_size, total_challenges)
                print(f"Processing {start} → {end}")

                indices = np.array_split(
                    np.arange(start, end),
                    num_workers
                )

                func = partial(self.process_batch, challenge_len=challenge_len)
                batch_res = pool.map(func, indices)
                batch_res = [item for sub in batch_res for item in sub]
                all_results.extend(batch_res)

        # Save CRPs
        df_out = pd.DataFrame(
            all_results,
            columns=[
                f"Challenge_{challenge_len}_bin",
                f"Response_{challenge_len}_bin_LOG_R",
                "Challenge_dec",
                "Response_dec"
            ]
        )

        df_out.to_csv(out_csv, index=False)
        print(f"Log-resistance CRPs saved to {out_csv}")
        return df_out


# ==========================================================
#                SHANNON ENTROPY FUNCTION
# ==========================================================
def shannon_entropy(bitstream):
    """Calculate Shannon entropy for binary bitstream."""
    bits = np.array(list(bitstream), dtype=np.uint8)
    values, counts = np.unique(bits, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)


# ==========================================================
#                       MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # ---------------- PUF PARAMETERS ----------------
    excel_path = "/home/lavanya/Documents/1.9V-1us 1million data.xlsx"  # Update path
    challenge_len = 8  # Bits per challenge/response
    crp_bits = 10  # Total challenges = 2^10 = 1024
    out_csv = "PUF_CRP_LOG_RESISTANCE.csv"

    # ---------------- GENERATE LOG-RESISTANCE CRPs ----------------
    puf = PUFDesign(excel_path, log_transform=True)  # log10(R + ε)
    df_crp = puf.generate_crp(crp_bits, challenge_len, out_csv)

    # ---------------- BUILD BITSTREAMS ----------------
    challenge_col = df_crp.columns[0]
    response_col = df_crp.columns[1]

    challenge_stream = ''.join(df_crp[challenge_col].astype(str).values)
    response_stream = ''.join(df_crp[response_col].astype(str).values)

    print(f"\n=== LOG-RESISTANCE PUF RESULTS ===")
    print(f"Total challenge bits: {len(challenge_stream)}")
    print(f"Total response bits:  {len(response_stream)}")

    bits = np.array(list(response_stream), dtype=int)
    print(f"P(1) = {bits.mean():.4f}")
    print(f"P(0) = {1 - bits.mean():.4f}")

    # ---------------- ENTROPY vs BITSTREAM LENGTH ----------------
    bsl_list = [2 ** i for i in range(3, 11)]  # 8 → 1024 bits
    entropy_challenge = []
    entropy_response = []

    for bsl in bsl_list:
        entropy_challenge.append(shannon_entropy(challenge_stream[:bsl]))
        entropy_response.append(shannon_entropy(response_stream[:bsl]))

    # ---------------- PUBLICATION-QUALITY PLOT ----------------
    plt.figure(figsize=(9, 6))
    plt.plot(bsl_list, entropy_challenge, 'o-', linewidth=2.5, markersize=9,
             label="Challenge entropy", color='tab:blue')
    plt.plot(bsl_list, entropy_response, 's-', linewidth=2.5, markersize=9,
             label="Response entropy (log R)", color='tab:orange')

    plt.xscale("log", base=2)
    plt.xticks(bsl_list, [f'$2^{i}$' for i in range(3, 11)], fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlabel("Bitstream Length (BSL)", fontsize=14, fontweight='bold')
    plt.ylabel("Shannon Entropy (bits/bit)", fontsize=14, fontweight='bold')
    plt.title("MoS₂-RRAM PUF: Log-Resistance Randomness Analysis",
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("PUF_LogResistance_Entropy.png", dpi=300, bbox_inches='tight')
    plt.show()