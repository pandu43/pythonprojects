import pandas as pd
import numpy as np

# ---------- load data ----------
df_std = pd.read_excel("/home/lavanya/Documents/4/filtered_std.xlsx")
df_cv  = pd.read_excel("/home/lavanya/Documents/4/filtered_cv.xlsx")

# merge and rename to consistent columns
df = df_std.merge(df_cv, on=["Amplitude(V)", "Duration(s)"])
df = df.rename(columns={'Filtered_Std(Ω)': 'STD', 'Filtered_CV': 'CV'})

# keep only CV > 10
df = df[df['CV'] > 10].reset_index(drop=True)

# helper columns
df['STD_order'] = np.floor(np.log10(df['STD'])).astype(int)
df['CV_floor'] = np.floor(df['CV']).astype(int)
df['idx'] = df.index

print("Total rows after CV>10 filter:", len(df))

# ------------------ Type 1: SAME CV (~), VERY DIFFERENT STD (>=10x) ------------------
type1_rows = []

# group by CV_floor to reduce search; within each group we'll consider exact CV distance < 1
for cv_floor, group in df.groupby('CV_floor'):
    # restrict to rows whose CV is within [cv_floor, cv_floor+1)
    g = group[np.abs(group['CV'] - cv_floor) < 1]
    if len(g) < 2:
        continue

    # find min STD row and max STD row (representatives)
    idx_min = g['STD'].idxmin()
    idx_max = g['STD'].idxmax()

    std_min = df.at[idx_min, 'STD']
    std_max = df.at[idx_max, 'STD']
    cv_min = df.at[idx_min, 'CV']
    cv_max = df.at[idx_max, 'CV']

    # ensure CV difference < 1 (they should be, because of the filter)
    if abs(cv_min - cv_max) >= 1:
        # skip (not same CV approx)
        continue

    # check extreme STD ratio
    if max(std_min, std_max) / min(std_min, std_max) >= 10.0:
        row = {
            "amp_A": df.at[idx_min, "Amplitude(V)"], "dur_A": df.at[idx_min, "Duration(s)"],
            "STD_A": std_min, "CV_A": cv_min, "idx_A": idx_min,
            "amp_B": df.at[idx_max, "Amplitude(V)"], "dur_B": df.at[idx_max, "Duration(s)"],
            "STD_B": std_max, "CV_B": cv_max, "idx_B": idx_max,
            "CV_floor": cv_floor
        }
        type1_rows.append(row)

print("Representative Type-1 pairs found:", len(type1_rows))

if type1_rows:
    df_type1 = pd.DataFrame(type1_rows)
    df_type1.to_excel("/home/lavanya/Documents/4/type1_rep_pairs.xlsx", index=False)
    print("Saved Type1 representative pairs to /home/lavanya/Documents/4/type1_rep_pairs.xlsx")
    print("\nSample Type1 pair:")
    print(df_type1.iloc[0].to_string())
else:
    print("No Type1 representative pairs found (no CV bucket has STD difference >=10x).")

# ------------------ Type 2: SAME STD ORDER, VERY DIFFERENT CV (>=5 units) ------------------
type2_rows = []

for order, group in df.groupby('STD_order'):
    if len(group) < 2:
        continue

    # pick min CV and max CV representatives (within the same STD order)
    idx_min_cv = group['CV'].idxmin()
    idx_max_cv = group['CV'].idxmax()

    cv_min = df.at[idx_min_cv, 'CV']
    cv_max = df.at[idx_max_cv, 'CV']

    if abs(cv_max - cv_min) >= 5.0:
        row = {
            "amp_A": df.at[idx_min_cv, "Amplitude(V)"], "dur_A": df.at[idx_min_cv, "Duration(s)"],
            "STD_A": df.at[idx_min_cv, "STD"], "CV_A": cv_min, "idx_A": idx_min_cv,
            "amp_B": df.at[idx_max_cv, "Amplitude(V)"], "dur_B": df.at[idx_max_cv, "Duration(s)"],
            "STD_B": df.at[idx_max_cv, "STD"], "CV_B": cv_max, "idx_B": idx_max_cv,
            "STD_order": order
        }
        type2_rows.append(row)

print("Representative Type-2 pairs found:", len(type2_rows))

if type2_rows:
    df_type2 = pd.DataFrame(type2_rows)
    df_type2.to_excel("/home/lavanya/Documents/4/type2_rep_pairs.xlsx", index=False)
    print("Saved Type2 representative pairs to /home/lavanya/Documents/4/type2_rep_pairs.xlsx")
    print("\nSample Type2 pair:")
    print(df_type2.iloc[0].to_string())
else:
    print("No Type2 representative pairs found (no STD order has CV span >=5).")

# ------------------ Final note ------------------
print("\nFinished. If you want ALL qualifying pairs instead of representative extremes, I can provide")
print("that too, but it will be slower — tell me and I'll add an optimized all-pairs routine.")