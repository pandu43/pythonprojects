"""Harness script to run PUF design and Shannon entropy analysis pipeline."""

import os
import sys
from datetime import datetime
import time
import glob

# Add src directory to path so we can import from trng and entropy modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from trng.PUFDesign import main_HRS
except Exception as e:
    print("[HARN] Could not import main_HRS from trng.PUFDesign:", e)
    sys.exit(1)

try:
    from entropy.ShannonEntropy import ShannonEntropy
except Exception as e:
    print("[HARN] Could not import ShannonEntropy:", e)
    sys.exit(1)

def generate_timestamp():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def main(challenge_size=16, crp=8, runs=1):
    print("\n================ HARNESS STARTED ================\n")
    timestamp = generate_timestamp()
    print(f"[HARN] Timestamp: {timestamp}")
    
    # Step 1: Run PUF Design
    print("\n[HARN] Running PUFDesign ...\n")
    try:
        main_HRS(challenge_size=challenge_size, crp=crp, save_csv=True, Binary=True, runs=runs, timestamp=timestamp)
        print("\n[HARN] PUF Design Completed.\n")
    except Exception as e:
        print(f"[HARN ERROR] PUF Design failed: {e}")
        return
    
    # Step 2: Run Shannon Entropy Analysis
    print("[HARN] Running Shannon Entropy Analysis ...\n")
    se_start = time.time()
    
    try:
        # Locate the generated CSV files
        folder_name = f"{timestamp}_C{challenge_size}R{crp}"
        csv_dir = os.path.join('Result', 'TRAN_result', folder_name, 'CSV')
        
        if not os.path.exists(csv_dir):
            print(f"[HARN ERROR] CSV directory not found: {csv_dir}")
            return
        
        # Find all CSV files for this run
        csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
        
        if not csv_files:
            print(f"[HARN ERROR] No CSV files found in {csv_dir}")
            return
        
        print(f"[HARN] Found {len(csv_files)} CSV file(s)")
        
        # Process each CSV file with ShannonEntropy
        for csv_idx, csv_path in enumerate(csv_files):
            print(f"\n[HARN] Processing CSV {csv_idx+1}/{len(csv_files)}: {os.path.basename(csv_path)}")
            
            # Extract run index from filename (e.g., "..._run0.csv" -> 0)
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
                challenge_size=challenge_size,
                crp=crp,
                run_idx=run_idx
            )
            se.load_csv(csv_path, usecols=[1])  # Load the ResponsesBinary column
            se.compute()
            stats = se.summary(csv_save=True)
            
            print(f"[HARN] Entropy Stats: Max={stats['max']:.4f}, Min={stats['min']:.4f}, Avg={stats['average']:.4f}")
        
        se_end = time.time()
        se_duration = se_end - se_start
        print(f"\n[HARN] Shannon Entropy execution time: {se_duration:.3f} seconds")
        print("\n================ HARNESS COMPLETED ================\n")
        
    except Exception as e:
        print(f"[HARN ERROR] Shannon Entropy analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run PUF Design and Shannon Entropy Analysis Pipeline')
    parser.add_argument('-c', '--challenge', type=int, default=16, help='Challenge size (default: 16)')
    parser.add_argument('-r', '--crp', type=int, default=10, help='CRP value (default: 8)')
    parser.add_argument('-n', '--runs', type=int, default=3, help='Number of runs (default: 1)')
    
    args = parser.parse_args()
    main(challenge_size=args.challenge, crp=args.crp, runs=args.runs)
