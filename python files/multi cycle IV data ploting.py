import os
import csv
import matplotlib.pyplot as plt

# List of CSV file paths
file_paths = [
    '/home/lavanya/Documents/2/plot1.csv',
    '/home/lavanya/Documents/2/plot2.csv',
    '/home/lavanya/Documents/2/plot3.csv',
    '/home/lavanya/Documents/2/plot4.csv',
    '/home/lavanya/Documents/2/plot5.csv',
    '/home/lavanya/Documents/2/plot6.csv',
    '/home/lavanya/Documents/2/plot7.csv',
    '/home/lavanya/Documents/2/plot8.csv',
    '/home/lavanya/Documents/2/plot9.csv',
]

# Files to be plotted in red
red_files = ['plot9.csv']

# Create figure
plt.figure(figsize=(7.31, 5.89))

for file_path in file_paths:
    if os.path.isfile(file_path):
        file_name = os.path.basename(file_path)

        # Open and read CSV
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = rows[0]  # Header row
        data_rows = rows[1:]  # Numerical data

        num_cycles = len(header) // 2  # V1, I1, V2, I2, ...

        for cycle in range(num_cycles):
            Voltage = []
            Current = []

            for row in data_rows:
                try:
                    v = float(row[2 * cycle])
                    i = float(row[2 * cycle + 1])
                    Voltage.append(v)
                    Current.append(i)
                except:
                    continue  # Skip bad rows

            if Voltage and Current:
                color = 'red' if file_name == 'plot9.csv' else 'gray'
                plt.semilogy(Voltage, Current, color=color, linewidth=1.5)

    else:
        print(f"File not found: {file_path}")

# Label and show plot
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()