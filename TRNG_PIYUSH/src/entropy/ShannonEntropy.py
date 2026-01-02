import numpy as np
import pandas as pd
import ast
from scipy.stats import entropy
from datetime import datetime
import os


class ShannonEntropy:
    def __init__(self, base=2, timestamp=None, file_name=None, challenge_size=None, crp=None, run_idx=None):
        self.base = base
        self.data = []
        self.entropies = []
        self.timestamp = timestamp
        self.file_name = file_name
        self.challenge_size = challenge_size
        self.crp = crp
        self.run_idx = run_idx
    

    @staticmethod
    def generate_timestamp():
        now = datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S")

    # =======================
    # CSV Loader
    # =======================
    def load_csv(self, csv_path, usecols=None, skiprows=None):
        usecols = usecols if usecols is not None else [1]
        skiprows = skiprows if skiprows is not None else []

        df = pd.read_csv(csv_path, usecols=usecols, skiprows=skiprows)

        def parse_item(x):
            # If it's a string looking like a list "[0, 1, 0]"
            if isinstance(x, str) and x.strip().startswith('['):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    pass # Fallback to digit parsing if eval fails
            
            # If it's a number or string of digits "101101"
            s = str(x).strip()
            # Filter to keep only digits 0 or 1? Or just take all digits?
            # User example: 101101... -> list
            return [int(c) for c in s if c.isdigit()]

        self.data = df.iloc[:, 0].apply(parse_item).tolist()
        self.data = df.iloc[:, 0].apply(parse_item).tolist()
        return self.data

    # =======================
    # Single Bitstream Loader
    # =======================
    def load_bitstream(self, bitstream):
        if not isinstance(bitstream, (list, np.ndarray)):
            raise TypeError("Bitstream must be a list or numpy array")

        self.data = [bitstream]
        self.data = [bitstream]
        return self.data

    # =======================
    # Shannon Entropy
    # =======================
    def compute(self):
        if not self.data:
            raise ValueError("No data loaded")

        self.entropies = []

        for bitstream in self.data:
            values, counts = np.unique(bitstream, return_counts=True)
            probabilities = counts / counts.sum()
            ent = entropy(probabilities, base=self.base)
            self.entropies.append(ent)

        return self.entropies

    # =======================
    # Entropy Summary
    # =======================
    def summary(self,csv_save=False):
        if not self.entropies:
            raise ValueError("Entropy not computed")

        stats = {
            "max": float(np.max(self.entropies)),
            "min": float(np.min(self.entropies)),
            "average": float(np.mean(self.entropies))
        }

        print("Max Entropy :", stats["max"])
        print("Min Entropy :", stats["min"])
        print("Average Shannon Entropy :", stats["average"])

        if csv_save:
            # Create directory if it doesn't exist - save to Result/Entropy
            output_dir = os.path.join('Result', 'Entropy')
            os.makedirs(output_dir, exist_ok=True)
            
            # Use timestamp for filename if provided
            timestamp_to_use = self.timestamp if self.timestamp else self.generate_timestamp()
            
            # Build filename in format: timestamp_C#R#_run#_Entropy.csv
            filename_parts = [timestamp_to_use]
            
            if self.challenge_size is not None and self.crp is not None:
                filename_parts.append(f"C{self.challenge_size}R{self.crp}")
            
            if self.run_idx is not None:
                filename_parts.append(f"run{self.run_idx}")
            
            filename_parts.append("Entropy.csv")
            filename = "_".join(filename_parts)
            
            output_path = os.path.join(output_dir, filename)
            
            df = pd.DataFrame(stats, index=[0])
            df.to_csv(output_path, index=False)
            print(f"[INFO] Saved entropy stats to {output_path}")

        return stats

    # =======================
    # OPTIONAL Bitstream Test
    # =======================
    def bitstream_test(self):
        """
        Monobit (balance) test.
        This function runs ONLY when explicitly called.
        """
        if not self.data:
            raise ValueError("No bitstream loaded")

        results = []

        for idx, bitstream in enumerate(self.data):
            bitstream = np.array(bitstream)

            ones = np.sum(bitstream == 1)
            zeros = np.sum(bitstream == 0)
            total = len(bitstream)

            results.append({
                "stream_id": idx,
                "length": total,
                "ones": int(ones),
                "zeros": int(zeros),
                "P(1)": ones / total,
                "P(0)": zeros / total,
                "balanced": abs((ones / total) - 0.5) < 0.01
            })

        return results


# ==================================================
# Script Entry Point
# ==================================================
def main():
    csv_path = r"src\entropy\TRNG_bitstream_data_C16R8_batch0_run0.csv"
    entropy_calc = ShannonEntropy(base=2,timestamp="2025-12-29_23-20-21")
    
    try:
        # Only entropy runs automatically when script is run directly
        entropy_calc.load_csv(csv_path, usecols=[1])
        entropy_calc.compute()
        entropy_calc.summary(csv_save=True)
    except Exception as e:
        print(f"[ERROR] Could not run default entropy check: {e}")

if __name__ == "__main__":
    main()
