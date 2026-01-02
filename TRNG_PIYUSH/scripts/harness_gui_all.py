"""
PUF Harness GUI – Corrected & Path-Safe Version
"""

import sys
import threading
import queue
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# =========================================================
# PATH SETUP (CORRECT FOR YOUR STRUCTURE)
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent        # TRNG_PIYUSH/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                   # TRNG_PIYUSH
WORKSPACE_ROOT = PROJECT_ROOT.parent               # pythonproject

# Imports
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# =========================================================
# MODULE IMPORTS
# =========================================================

try:
    from trng.PUFDesign import main_HRS
except Exception as e:
    print(f"[WARN] Could not import main_HRS: {e}")
    main_HRS = None

try:
    from entropy.ShannonEntropy import ShannonEntropy
except Exception as e:
    print(f"[WARN] Could not import ShannonEntropy: {e}")
    ShannonEntropy = None

try:
    from nist.nist_tests import run_nist_tests
except Exception:
    run_nist_tests = None


# =========================================================
# STDOUT REDIRECTOR
# =========================================================

class StdoutRedirector:
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s):
        if s and not s.isspace():
            self.q.put(s)

    def flush(self):
        pass


# =========================================================
# GUI CLASS
# =========================================================

class HarnessGUI:
    def __init__(self, root):
        self.root = root
        root.geometry("1300x750")
        root.minsize(1200, 700)

        root.title("PUF Harness GUI")

        frame = ttk.Frame(root, padding=10)
        frame.grid(sticky="nsew")

        root.rowconfigure(0, weight=1)
        # root.columnconfigure(0, weight=1)

        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        # ---------------- Inputs ----------------
        ttk.Label(frame, text="Challenge size").grid(row=0, column=0)
        self.challenge_entry = ttk.Entry(frame)
        self.challenge_entry.insert(0, "4")
        self.challenge_entry.grid(row=0, column=1)

        ttk.Label(frame, text="CRP").grid(row=1, column=0)
        self.crp_entry = ttk.Entry(frame)
        self.crp_entry.insert(0, "4")
        self.crp_entry.grid(row=1, column=1)

        ttk.Label(frame, text="Runs").grid(row=2, column=0)
        self.runs_entry = ttk.Entry(frame)
        self.runs_entry.insert(0, "4")
        self.runs_entry.grid(row=2, column=1)

        ttk.Label(frame, text="Timestamp (optional)").grid(row=3, column=0)
        self.ts_entry = ttk.Entry(frame)
        self.ts_entry.grid(row=3, column=1)

        # ---------------- Checkboxes ----------------
        self.run_puf = tk.BooleanVar(value=True)
        self.run_entropy = tk.BooleanVar(value=True)
        self.run_nist = tk.BooleanVar(value=False)

        ttk.Checkbutton(frame, text="Run PUF", variable=self.run_puf).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(frame, text="Run Entropy", variable=self.run_entropy).grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(frame, text="Run NIST", variable=self.run_nist).grid(row=5, column=0, sticky="w")

        ttk.Button(frame, text="Run Tests", command=self.start).grid(row=6, column=0, columnspan=2, pady=6)

        # ---------------- Log box ----------------
        self.log = scrolledtext.ScrolledText(frame, width=155, height=25)
        self.log.grid(row=7, column=0, columnspan=2, sticky="nsew")

        frame.rowconfigure(7, weight=1)

        # Redirect stdout
        self.q = queue.Queue()
        sys.stdout = StdoutRedirector(self.q)
        self.root.after(200, self.poll_queue)

        self.worker = None

    def poll_queue(self):
        while not self.q.empty():
            self.log.insert("end", self.q.get())
            self.log.see("end")
        self.root.after(200, self.poll_queue)

    # =====================================================
    # MAIN RUNNER
    # =====================================================

    def start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Info", "Process already running")
            return

        try:
            challenge = int(self.challenge_entry.get())
            crp = int(self.crp_entry.get())
            runs = int(self.runs_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input")
            return

        timestamp = self.ts_entry.get().strip() or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        def task():
            print(f"[GUI] PROJECT_ROOT = {PROJECT_ROOT}")
            print(f"[GUI] Starting: C={challenge}, R={crp}, runs={runs}, ts={timestamp}\n")

            # ---------------- PUF ----------------
            if self.run_puf.get():
                print("[GUI] Running PUF...\n")
                # main_HRS(16, 8, save_csv=True, Binary=True, runs=1)
                main_HRS( challenge,
                    crp,
                    save_csv=True,
                    Binary=True,
                    runs=runs,
                    timestamp=timestamp,
                    project_root=PROJECT_ROOT,   # <<< IMPORTANT
                )

                print("\n\n\n" + "*" * 82 + "\n")
                print("                           [GUI] PUF completed")

            # ---------------- ENTROPY ----------------
            if self.run_entropy.get():


                print("\n"+"                           [GUI] Running Entropy...")
                print("\n" + "*" * 82 + "\n")

                # print("[GUI] Running Entropy...\n")

                csv_dir = PROJECT_ROOT / "Result" / "TRAN_result" / f"{timestamp}_C{challenge}R{crp}" / "CSV"
                if not csv_dir.exists():
                    print("[GUI] No CSV files found\n")
                else:
                    for csv_file in csv_dir.glob("*.csv"):
                        se = ShannonEntropy(base=2,
                                            timestamp=timestamp,
                                            challenge_size=challenge,
                                            crp=crp,
                                            total_runs=runs)

                        se.load_csv(csv_file, usecols=[1])
                        se.compute()
                        stats = se.summary(csv_save=True)
                        print(f"\n[Entropy] {csv_file.name}: Avg={stats['average']:.4f}")

            # ---------------- NIST ----------------
            if self.run_nist.get():
                print("[GUI] Running NIST...\n")
                run_nist_tests(timestamp=timestamp)

            print("\n\n\n******************************************************************************")
            print("\n                          [GUI] ALL TASKS COMPLETED")
            print("\n\n******************************************************************************")

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()


# =========================================================
# ENTRY POINT
# =========================================================

def main():
    root = tk.Tk()
    HarnessGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
