from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import os

# List of file paths
file_paths = [
    '/home/lavanya/Documents/abrupt reset muti cycles data(10k iv)/plot1.csv',
    '/home/lavanya/Documents/abrupt reset muti cycles data(10k iv)/plot2.csv',
    '/home/lavanya/Documents/abrupt reset muti cycles data(10k iv)/plot3.csv',
    '/home/lavanya/Documents/abrupt reset muti cycles data(10k iv)/plot4.csv',
    '/home/lavanya/Documents/abrupt reset muti cycles data(10k iv)/plot5.csv',
    '/home/lavanya/Documents/abupt reset muti cycles data(10k iv)/plot6.csv',
    '/home/lavanya/Documents/abrupt reset muti cycles data(10k iv)/plot7.csv',
    '/home/lavanya/Documents/abrupt reset muti cycles data(10k iv)/plot8.csv',
]

# List of files to plot in red
red_files = ['plot1.csv', 'plot3.csv', 'plot5.csv', 'plot7.csv']

# Create a plot with the specified figure size
plt.figure(figsize=(7.31, 5.89))

for file_path in file_paths:
    if os.path.isfile(file_path):
        with open(file_path, 'r') as csvfile:
            # Create a reader object
            csvreader = csv.reader(csvfile)

            # Read the first row to determine the number of cycles (columns)
            header = next(csvreader)
            num_cycles = (len(header) - 1) // 2  # Divide by 2 for voltage and current pairs

            # Reset the file pointer to the beginning for the first cycle
            csvfile.seek(0)
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header row

            # Process data for each cycle
            for cycle in range(num_cycles):
                Voltage = []
                Current = []

                for row in csvreader:
                    try:
                        voltage_value = float(row[cycle * 2])
                        current_value = float(row[cycle * 2 + 1])
                        if current_value != 0:
                            Voltage.append(voltage_value)
                            Current.append(current_value)
                    except (ValueError, IndexError):
                        pass  # Skip non-numeric values or index errors

                # Determine the color and z-order based on the file name
                file_name = os.path.basename(file_path)
                if cycle == 0 and file_name in red_files:
                    color = 'red'
                    zorder = 2  # Higher z-order for first cycle in red
                else:
                    color = 'gray'
                    zorder = 1  # Lower z-order for other cycles

                # Plot data for the current cycle
                if Voltage and Current:  # Only plot if there is valid data
                    plt.semilogy(Voltage, Current, color=color, linewidth=1.5, zorder=zorder,
                                 label=f"Cycle {cycle + 1} - {file_name}" if cycle == 0 else "")

                # Reset file pointer to beginning for the next cycle
                csvfile.seek(0)
                next(csvreader)  # Skip header row again for the next cycle
    else:
        print(f"File not found: {file_path}")

# Set limits
plt.xlim(-1.3, 4)
plt.ylim(1e-12, 5e-1)

# Configure the plot
ax = plt.gca()
ax.spines['top'].set_linewidth(2)  # Top spine
ax.spines['bottom'].set_linewidth(2)  # Bottom spine
ax.spines['left'].set_linewidth(2)  # Left spine
ax.spines['right'].set_linewidth(2)  # Right spine

# Adjust tick parameters and label sizes
plt.tick_params(axis='both', which='major', labelsize=16)  # Increased font size for tick labels
plt.ylabel("I (A)", fontsize=22, labelpad=5, rotation=90, color='black')  # Darker label
plt.xlabel("V$_{app}$ (V)", fontsize=22, labelpad=5, color='black')  # Darker label
plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.tick_params(direction='in', length=10, width=2, colors='k', grid_color='k', grid_alpha=0.5)

# Show the plot
plt.show()
