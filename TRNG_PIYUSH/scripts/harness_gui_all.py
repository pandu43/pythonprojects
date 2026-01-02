"""Dry-run copy of `harness_gui.py` into `scripts/`.
"""
import os
import sys
import threading
import queue
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from trng.PUFDesign import main_HRS
except Exception as e:
    print(f"[WARN] Could not import main_HRS: {e}")
    main_HRS = None

try:
    from trng.PUF_RNG import PUF_RNG
    from trng.puf_adapter import PUFAdapter
except Exception:
    PUF_RNG = None
    PUFAdapter = None

try:
    from entropy.ShannonEntropy import ShannonEntropy
except Exception as e:
    print(f"[WARN] Could not import ShannonEntropy: {e}")
    ShannonEntropy = None

try:
    from nist.nist_tests import run_nist_tests
except Exception:
    run_nist_tests = None

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox


class StdoutRedirector:
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s):
        if s and not s.isspace():
            self.q.put(s)

    def flush(self):
        pass


class HarnessGUI:
    def __init__(self, root):
        self.root = root
        root.title('PUF Harness GUI')

        mainframe = ttk.Frame(root, padding='8')
        mainframe.grid(row=0, column=0, sticky='nsew')
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        # Inputs
        ttk.Label(mainframe, text='Challenge size (bits)').grid(row=0, column=0)
        self.challenge_entry = ttk.Entry(mainframe)
        self.challenge_entry.insert(0, '32')
        self.challenge_entry.grid(row=0, column=1)

        ttk.Label(mainframe, text='CRP (crp)').grid(row=1, column=0)
        self.crp_entry = ttk.Entry(mainframe)
        self.crp_entry.insert(0, '4')
        self.crp_entry.grid(row=1, column=1)

        ttk.Label(mainframe, text='Runs').grid(row=2, column=0)
        self.runs_entry = ttk.Entry(mainframe)
        self.runs_entry.insert(0, '1')
        self.runs_entry.grid(row=2, column=1)

        ttk.Label(mainframe, text='Timestamp (optional)').grid(row=3, column=0)
        self.ts_entry = ttk.Entry(mainframe)
        self.ts_entry.grid(row=3, column=1)

        ttk.Label(mainframe, text='Select Tests to Run:').grid(row=4, column=0, columnspan=2, sticky='w')
        
        # Checkboxes for test selection
        self.run_puf_var = tk.BooleanVar(value=True)
        self.run_entropy_var = tk.BooleanVar(value=True)
        self.run_nist_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(mainframe, text='Run PUF', variable=self.run_puf_var).grid(row=5, column=0, sticky='w', padx=20)
        ttk.Checkbutton(mainframe, text='Run Entropy', variable=self.run_entropy_var).grid(row=5, column=1, sticky='w')
        ttk.Checkbutton(mainframe, text='Run NIST', variable=self.run_nist_var).grid(row=6, column=0, sticky='w', padx=20)

        ttk.Button(mainframe, text='Run Tests', command=self.start_harness).grid(row=7, column=0, columnspan=2, pady=6)

        ttk.Separator(mainframe, orient='horizontal').grid(row=8, column=0, columnspan=2, sticky='ew', pady=6)

        # Log box
        self.log_box = scrolledtext.ScrolledText(mainframe, width=80, height=20)
        self.log_box.grid(row=10, column=0, columnspan=2, sticky='nsew')
        mainframe.rowconfigure(10, weight=1)

        # Queue & redirect
        self.q = queue.Queue()
        self._orig_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.q)

        # Poll the queue
        self.root.after(200, self.poll_queue)

        # Thread handles
        self.harness_thread = None

    def poll_queue(self):
        while True:
            try:
                msg = self.q.get_nowait()
            except queue.Empty:
                break
            else:
                self.log_box.insert('end', msg)
                self.log_box.see('end')
        self.root.after(200, self.poll_queue)

    def start_harness(self):
        # Get checkbox states
        run_puf = self.run_puf_var.get()
        run_entropy = self.run_entropy_var.get()
        run_nist = self.run_nist_var.get()
        
        # Check if at least one test is selected
        if not (run_puf or run_entropy or run_nist):
            messagebox.showwarning('Warning', 'Please select at least one test to run')
            return
        
        # Validate based on selected tests
        if run_puf and main_HRS is None:
            messagebox.showerror('Error', 'PUFDesign harness (PUFDesign.main_HRS) not available')
            return
        
        if run_entropy and ShannonEntropy is None:
            messagebox.showerror('Error', 'Entropy module (entropy.ShannonEntropy) not available')
            return
        
        if run_nist and run_nist_tests is None:
            messagebox.showerror('Error', 'NIST module (NIST.nist_tests) not available')
            return
        
        if self.harness_thread and self.harness_thread.is_alive():
            messagebox.showinfo('Info', 'Tests already running')
            return

        try:
            challenge = int(self.challenge_entry.get())
            crp = int(self.crp_entry.get())
            runs = int(self.runs_entry.get())
        except ValueError:
            messagebox.showerror('Error', 'Invalid numeric input')
            return

        timestamp = self.ts_entry.get().strip() or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        def target():
            selected_tests = []
            if run_puf:
                selected_tests.append('PUF')
            if run_entropy:
                selected_tests.append('Entropy')
            if run_nist:
                selected_tests.append('NIST')
            
            print(f'[GUI] Starting tests: {" + ".join(selected_tests)}')
            print(f'[GUI] C={challenge}, R={crp}, runs={runs}, ts={timestamp}\n')
            
            try:
                # Run PUF if selected
                if run_puf:
                    print('[GUI] Running PUF...\n')
                    main_HRS(challenge, crp, save_csv=True, Binary=True, runs=runs, timestamp=timestamp)
                    print('[GUI] PUF completed\n')
                
                # Run Entropy if selected
                if run_entropy:
                    print('[GUI] Running Entropy analysis...\n')
                    try:
                        # Run entropy on generated CSV files
                        import glob
                        folder_name = f"{timestamp}_C{challenge}R{crp}"
                        csv_dir = os.path.join('Result', 'TRAN_result', folder_name, 'CSV')
                        
                        if os.path.exists(csv_dir):
                            csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
                            for csv_idx, csv_path in enumerate(csv_files):
                                # Extract run index from filename
                                filename = os.path.basename(csv_path)
                                run_idx = None
                                if '_run' in filename:
                                    try:
                                        run_idx = int(filename.split('_run')[-1].split('.')[0])
                                    except ValueError:
                                        pass
                                
                                se = ShannonEntropy(
                                    base=2,
                                    timestamp=timestamp,
                                    challenge_size=challenge,
                                    crp=crp,
                                    run_idx=run_idx
                                )
                                se.load_csv(csv_path, usecols=[1])
                                se.compute()
                                stats = se.summary(csv_save=True)
                                print(f'[GUI] Entropy: Max={stats["max"]:.4f}, Min={stats["min"]:.4f}, Avg={stats["average"]:.4f}')
                        else:
                            print(f'[GUI] No CSV files found for entropy analysis')
                            
                        print('[GUI] Entropy analysis completed\n')
                    except Exception as e:
                        print(f'[GUI] Entropy error: {e}\n')
                        import traceback
                        traceback.print_exc()
                
                # Run NIST if selected
                if run_nist:
                    print('[GUI] Running NIST tests...\n')
                    try:
                        run_nist_tests(timestamp=timestamp)
                        print('[GUI] NIST tests completed\n')
                    except Exception as e:
                        print(f'[GUI] NIST error: {e}\n')
                
                print('\n[GUI] All selected tests finished')
            except Exception as e:
                print(f'\n[GUI] Error: {e}')

        self.harness_thread = threading.Thread(target=target, daemon=True)
        self.harness_thread.start()


def main():
    root = tk.Tk()
    app = HarnessGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
