import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import pandas as pd

y0 = -1.66432
A  = 1.35558e3
xc = 6.57341
w  = 0.38959
a3 = 50.29355
a4 = 2050.27159

fac3 = 3*2*1
fac4 = 4*3*2*1
fac6 = 6*5*4*3*2*1
hotspots_value=[]
num_outputs = int(input( "Enter the number of required outputs : "))
# ---------------- Define hotspot PDF ----------------
def hotspot_pdf(x, y0, A, xc, w, a3, a4):
    u = (x - xc) / w
    term1 = np.exp(-0.5 * u**2)
    t_a3  = (a3 / fac3) * (u * (u**2 - 3))
    t_a4  = (a4 / fac4) * (u**4 - 6*u**3 + 3)   # literal from ECS equation
    t_a32 = (10 * a3**2 / fac6) * (u**6 - 15*u**4 + 45*u**2 - 15)
    term2 = 1 + t_a3 + t_a4 + t_a32
    return y0 + (A / (w * np.sqrt(2*np.pi))) * term1 * term2
# ---------------- Generate hotspot values ----------------
def generate_hotspots(num_outputs):
    x_values = np.linspace(4.4, 6.4, num_outputs)
    pdf_values = hotspot_pdf(x_values, y0, A, xc, w, a3, a4)
    if num_outputs == 1:
        return np.power(10, x_values)  # Return the single current value directly
    # Normalize the PDF values and convert to counts
    counts = (pdf_values - np.min(pdf_values))  # Shift to make minimum zero
    if np.max(counts) != 0:  #
       counts = (counts / np.max(counts)) * num_outputs # Scale counts for visualization (e.g., max value set to 1000)

    H_values = np.power(10, x_values)  # I = 10^(-log10(R))
    hotspots_value = np.random.choice(H_values, size=num_outputs, p=counts / np.sum(counts))

    return hotspots_value

i = 0
temp = []
i = i + 1
for i in range(num_outputs):
    hotspots = generate_hotspots(num_outputs)[i]
    hotspots_value.append(hotspots)

# Convert to numpy array for convenience
hotspots_value = np.array(hotspots_value)

# Convert to DataFrame
df = pd.DataFrame(hotspots_value, columns=["hotspots_value"])
df.to_excel("/home/lavanya/Documents/hotspots_values.xlsx", index=False)
print("✅ Hotspot values saved to Excel")

# ---------------- CDF ----------------
sorted_values = np.sort(hotspots_value)
cdf = np.arange(1, len(sorted_values)+1) / len(sorted_values)*100

plt.figure(figsize=(6,4))
plt.plot(sorted_values, cdf, color="green", linewidth=2, label="Simulated CDF")
plt.xscale("log")
plt.xlabel("Hotspot Value")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Hotspot Values")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
  # --- Read experimental data from ODS file ---
exp_file = "/home/lavanya/Documents/hotspots distribution.ods"
exp_df = pd.read_excel(exp_file, engine='odf')  # Requires odfpy
    # Assuming the file has 4 columns: x1, y1, x2, y2
#x1 = exp_df.iloc[:, 0]
#y1 = exp_df.iloc[:, 1]
x2 = exp_df.iloc[:, 2]
y2 = exp_df.iloc[:, 3]
    # --- Overlay experimental data in one color (e.g., black) ---
#plt.plot(x1, y1, '-', color='red', label='exp data')
plt.plot(x2, y2, '-', color='red', label='exp cdf data')
plt.xscale("log")  # currents span decades
plt.legend()
plt.show()


