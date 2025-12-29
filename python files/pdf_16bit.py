# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import functions as f


# %% md
### LRS
# %%
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

################################ Data Loading & Processing for Resistance Values  ########################
bin_size = 21

df = pd.read_excel(r"C:\Users\Kartik\Desktop\Stochastic_Computing\cycle_data_1e5\LRS_1e5_data.xlsx")

combined_series = pd.concat([df[col] for col in df.columns], ignore_index=True)
combined_series = combined_series.dropna()
resistance_val = combined_series.values

print(resistance_val.shape)

# print("Below 2.7kohm: {}".format(np.sum(resistance_val < 2700)))
# print("Above 270kohms: {}".format(np.sum(resistance_val > 270000)))
resistance_val_log = np.log10(resistance_val)
resistance_val_log = np.round(resistance_val_log, 2)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
# Create histogram
counts, bin_edges, _ = plt.hist(resistance_val_log, label='Histogram', edgecolor='white', bins=bin_size)
plt.legend()

# Get bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

x_lower = bin_edges[0]
x_upper = bin_edges[-1]

x_arr = np.linspace(bin_centers[0], bin_centers[-1], 100000)
popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=[max(counts), np.mean(bin_centers), np.std(bin_centers)])
plt.plot(x_arr, gaussian(x_arr, *popt), color='r', label='Gaussian Fit')
plt.legend()
plt.title("1e5 cycles of LRS")

plt.subplot(1, 2, 2)
count, bins_count = np.histogram(resistance_val_log, bins=bin_size)

# ###############3############finding the PDF of the histogram using count values   ##########################3
# pdf = count/sum(count)
pdf = count

# print(count)
# print(bins_count)
plt.plot(x_arr, gaussian(x_arr, *popt), color='r', label='Gaussian Fit')
plt.plot(bin_centers, pdf, label="PDF")
plt.scatter(bin_centers, pdf)
plt.legend()
plt.title("pdf and gaussian fit")
plt.show()

print(bin_centers)
print(popt)
print(pcov)

# %% md
####################################   Finding thresholds   ######################################################
# %%
area = np.trapz(pdf, bin_centers)
print(area)

splits = 16

# Define the area per partition
partition_area = area / splits


# Function to find the x-value for a given target area
def find_x_for_area(target_area, x, y):
    cumulative_area = 0
    for i in range(1, len(x)):
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        trapezoid_area = (y0 + y1) * (x1 - x0) / 2

        if cumulative_area + trapezoid_area >= target_area:
            remaining_area = target_area - cumulative_area
            base = x1 - x0
            height_diff = y1 - y0
            if height_diff == 0:
                # If y0 == y1, the trapezoid is a rectangle
                return x0 + remaining_area / y0
            else:
                a = y0
                b = height_diff / base
                a_q = b
                b_q = 2 * a - 2 * x0 * b
                c_q = b * (x0 ** 2) - 2 * a * x0 - 2 * remaining_area
                discriminant = (b_q ** 2) - (4 * a_q * c_q)
                x_sol1 = (-b_q + np.sqrt(discriminant)) / (2 * a_q)
                x_sol2 = (-b_q - np.sqrt(discriminant)) / (2 * a_q)
                if ((x_sol1 >= x0) and (x_sol1 <= x1)):
                    x_solution = x_sol1
                else:
                    x_solution = x_sol2
                return x_solution
        cumulative_area += trapezoid_area
    return x[-1]


# Find the thresholds
thresholds = []
for i in range(1, splits):
    target_area = i * partition_area
    threshold_x = find_x_for_area(target_area, bin_centers, pdf)
    thresholds.append(threshold_x)

print("Thresholds for equal partitioning:", thresholds)

plt.scatter(bin_centers, pdf, color='k')
plt.plot(bin_centers, pdf, color='r', label='pdf')
counts, bin_edges, _ = plt.hist(resistance_val_log, label='Histogram', edgecolor='white', bins=bin_size, alpha=0.5)
# counts, bin_edges, _ = plt.hist(resistance_val_log, label='Histogram', alpha=0.5)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
          'cyan', 'magenta', 'yellow', 'lime', 'teal', 'navy', 'maroon']
for i in range(splits - 1):
    plt.axvline(thresholds[i], label=f'{(i + 1) * 6.25}%: {thresholds[i]:.2f}', color=f'{colors[i]}')
plt.title("16-bit precision: 1e5 cycles of LRS")
plt.legend()
plt.show()

thresholds_16bit = np.array([x_lower, *thresholds, x_upper])
print(thresholds_16bit)
# %% md
### 10, 20 & 50 MAC with 16 bit precision
# %%
precision = 16

# 10-input MAC
input_size = 10
input_length = 10
iterations_arr = np.array([1, 2, 5, 10, 25, 50])
x_labels = np.array(['1', '2', '5', '10', '25', '50'])

iter_acc_matrix = np.zeros((input_size, len(iterations_arr)))

for k in range(input_size):

    weights = 5 * np.random.random(input_length)
    inputs = np.random.randint(0, precision + 1, size=input_length)
    actual_res = np.sum(np.multiply(weights, inputs))
    accuracy_list = []

    for p in range(len(iterations_arr)):
        result = f.MAC_16bit(weights, inputs, resistance_val, thresholds_16bit, iterations_arr[p])
        acc = 100 * (1 - (np.abs(actual_res - result) / actual_res))
        accuracy_list.append(acc)

    accuracy_list = np.asarray(accuracy_list)
    iter_acc_matrix[k, :] = accuracy_list

y_arr_10_lrs = np.zeros(len(iterations_arr))
y_err_10_lrs = np.zeros(len(iterations_arr))

for k in range(len(iterations_arr)):
    y_arr_10_lrs[k] = np.mean(iter_acc_matrix[:, k])
    y_err_10_lrs[k] = np.std(iter_acc_matrix[:, k])

# 20-input MAC
input_size = 10
input_length = 20

iter_acc_matrix = np.zeros((input_size, len(iterations_arr)))

for k in range(input_size):

    weights = 5 * np.random.random(input_length)
    inputs = np.random.randint(0, precision + 1, size=input_length)
    actual_res = np.sum(np.multiply(weights, inputs))
    accuracy_list = []

    for p in range(len(iterations_arr)):
        result = f.MAC_16bit(weights, inputs, resistance_val, thresholds_16bit, iterations_arr[p])
        acc = 100 * (1 - (np.abs(actual_res - result) / actual_res))
        accuracy_list.append(acc)

    accuracy_list = np.asarray(accuracy_list)
    iter_acc_matrix[k, :] = accuracy_list

y_arr_20_lrs = np.zeros(len(iterations_arr))
y_err_20_lrs = np.zeros(len(iterations_arr))

for k in range(len(iterations_arr)):
    y_arr_20_lrs[k] = np.mean(iter_acc_matrix[:, k])
    y_err_20_lrs[k] = np.std(iter_acc_matrix[:, k])

# 50-input MAC
input_size = 10
input_length = 50

iter_acc_matrix = np.zeros((input_size, len(iterations_arr)))

for k in range(input_size):

    weights = 5 * np.random.random(input_length)
    inputs = np.random.randint(0, precision + 1, size=input_length)
    actual_res = np.sum(np.multiply(weights, inputs))
    accuracy_list = []

    for p in range(len(iterations_arr)):
        result = f.MAC_16bit(weights, inputs, resistance_val, thresholds_16bit, iterations_arr[p])
        acc = 100 * (1 - (np.abs(actual_res - result) / actual_res))
        accuracy_list.append(acc)

    accuracy_list = np.asarray(accuracy_list)
    iter_acc_matrix[k, :] = accuracy_list

y_arr_50_lrs = np.zeros(len(iterations_arr))
y_err_50_lrs = np.zeros(len(iterations_arr))

for k in range(len(iterations_arr)):
    y_arr_50_lrs[k] = np.mean(iter_acc_matrix[:, k])
    y_err_50_lrs[k] = np.std(iter_acc_matrix[:, k])

plt.figure(figsize=(8, 6))
plt.xscale("log")
plt.errorbar(iterations_arr, y_arr_10_lrs, yerr=y_err_10_lrs, fmt='o', color='green', label='10 MAC')
plt.errorbar(iterations_arr, y_arr_20_lrs, yerr=y_err_20_lrs, fmt='o', color='red', label='20 MAC')
plt.errorbar(iterations_arr, y_arr_50_lrs, yerr=y_err_50_lrs, fmt='o', color='k', label='50 MAC')
plt.xticks(iterations_arr, x_labels)
plt.legend()
plt.xlabel("Number of iterations over which the result is obtained")
plt.ylabel("Range of Accuracy for a set of 10 test inputs")
plt.title("10, 20, 50 input MAC with 16-bit precision using LRS data")
plt.show()
# %% md
### HRS
# %%
bin_count = 21

plt.figure(figsize=(15, 6))

df = pd.read_excel(r"C:\Users\Kartik\Desktop\Stochastic_Computing\cycle_data_1e5\HRS_1e5_data.xlsx")

combined_series = pd.concat([df[col] for col in df.columns], ignore_index=True)
combined_series = combined_series.dropna()
resistance_val = combined_series.values

print(resistance_val.shape)

print("Below 27 Mohm: {}".format(np.sum(resistance_val < 15000000)))
# print("Above 270kohms: {}".format(np.sum(resistance_val > 270000)))

resistance_val_log = np.log10(resistance_val)
resistance_val_round = np.round(resistance_val_log, 2)
print(resistance_val_round)
print(np.min(resistance_val_log))

plt.subplot(1, 2, 1)
# Create histogram
counts, bin_edges, _ = plt.hist(resistance_val_round, label='Histogram', edgecolor='white', bins=bin_count)
plt.legend()

# Get bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

x_lower = bin_edges[0]
x_upper = bin_edges[-1]

x_arr = np.linspace(bin_centers[0], bin_centers[-1], 100000)
popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=[max(counts), np.mean(bin_centers), np.std(bin_centers)])
plt.plot(x_arr, gaussian(x_arr, *popt), color='r', label='Gaussian Fit')
plt.legend()
plt.title("1e5 cycles of HRS")

plt.subplot(1, 2, 2)
count, bins_points = np.histogram(resistance_val_round, bins=bin_count)
print(bins_points)
# finding the PDF of the histogram using count values
# pdf = count/sum(count)
pdf = count

plt.plot(x_arr, gaussian(x_arr, *popt), color='r', label='Gaussian Fit')
plt.plot(bin_centers, pdf, label="PDF")
plt.scatter(bin_centers, pdf)
plt.legend()
plt.title("pdf and gaussian fit")

plt.show()
# %%
# # plt.figure(figsize=(15, 6))
# from scipy.stats import skewnorm

# df = pd.read_excel(r"C:\Users\Kartik\Desktop\Stochastic_Computing\cycle_data_1e5\HRS_1e5_data.xlsx")

# combined_series = pd.concat([df[col] for col in df.columns], ignore_index=True)
# combined_series = combined_series.dropna()
# resistance_val = combined_series.values

# print(resistance_val.shape)
# resistance_val_log = np.log10(resistance_val)
# resistance_val_round = np.round(resistance_val_log, 2)
# print(resistance_val_round)

# bin_size  = 0.2

# min_edge = np.min(resistance_val_round)
# max_edge = np.max(resistance_val_round)
# bins = np.arange(min_edge, max_edge + bin_size, bin_size)

# # Plot the histogram using the calculated bin edges
# plt.hist(resistance_val_round, bins=bins, edgecolor='white')

# shape, loc, scale = skewnorm.fit(resistance_val_round)

# # Create x values for plotting the fitted distribution
# x = np.linspace(min_edge, max_edge, 1000)

# # Compute the PDF of the fitted skew normal distribution
# pdf = skewnorm.pdf(x, shape, loc, scale)
# pdf = pdf * len(resistance_val_round)*bin_size


# # Plot the fitted skew normal distribution on top of the histogram
# plt.plot(x, pdf, 'r-', lw=2, label='Fitted Skew Normal PDF')
# plt.xlim(7, 9.5)

# plt.show()

# %% md
### Finding Thresholds
# %%
area = np.trapz(pdf, bin_centers)
print(area)

splits = 16

# Define the area per partition
partition_area = area / splits


# Function to find the x-value for a given target area
def find_x_for_area(target_area, x, y):
    cumulative_area = 0
    for i in range(1, len(x)):
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        trapezoid_area = (y0 + y1) * (x1 - x0) / 2

        if cumulative_area + trapezoid_area >= target_area:
            remaining_area = target_area - cumulative_area
            base = x1 - x0
            height_diff = y1 - y0
            if height_diff == 0:
                # If y0 == y1, the trapezoid is a rectangle
                return x0 + remaining_area / y0
            else:
                a = y0
                b = height_diff / base
                a_q = b
                b_q = 2 * a - 2 * x0 * b
                c_q = b * (x0 ** 2) - 2 * a * x0 - 2 * remaining_area
                discriminant = (b_q ** 2) - (4 * a_q * c_q)
                x_sol1 = (-b_q + np.sqrt(discriminant)) / (2 * a_q)
                x_sol2 = (-b_q - np.sqrt(discriminant)) / (2 * a_q)
                if ((x_sol1 >= x0) and (x_sol1 <= x1)):
                    x_solution = x_sol1
                else:
                    x_solution = x_sol2
                return x_solution
        cumulative_area += trapezoid_area
    return x[-1]


# Find the thresholds
thresholds = []
for i in range(1, splits):
    target_area = i * partition_area
    threshold_x = find_x_for_area(target_area, bin_centers, pdf)
    thresholds.append(threshold_x)

print("Thresholds for equal partitioning:", thresholds)

plt.scatter(bin_centers, pdf, color='k')
plt.plot(bin_centers, pdf, color='r', label='pdf')
counts, bin_edges, _ = plt.hist(resistance_val_log, label='Histogram', edgecolor='white', bins=bin_size, alpha=0.5)
# counts, bin_edges, _ = plt.hist(resistance_val_log, label='Histogram', alpha=0.5)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
          'cyan', 'magenta', 'yellow', 'lime', 'teal', 'navy', 'maroon']
for i in range(splits - 1):
    plt.axvline(thresholds[i], label=f'{(i + 1) * 6.25}%: {thresholds[i]:.2f}', color=f'{colors[i]}')
plt.title("16-bit precision: 1e5 cycles of HRS")
plt.legend()
plt.show()

thresholds_16bit = np.array([x_lower, *thresholds, x_upper])
print(thresholds_16bit)
# %% md
### 10, 20 & 50 MAC with 16 bit precision
# %%
precision = 16

# 10-input MAC
input_size = 10
input_length = 10
iterations_arr = np.array([1, 2, 5, 10, 25, 50])
x_labels = np.array(['1', '2', '5', '10', '25', '50'])

iter_acc_matrix = np.zeros((input_size, len(iterations_arr)))

for k in range(input_size):

    weights = 5 * np.random.random(input_length)
    inputs = np.random.randint(0, precision + 1, size=input_length)
    actual_res = np.sum(np.multiply(weights, inputs))
    accuracy_list = []

    for p in range(len(iterations_arr)):
        result = f.MAC_16bit(weights, inputs, resistance_val, thresholds_16bit, iterations_arr[p])
        acc = 100 * (1 - (np.abs(actual_res - result) / actual_res))
        accuracy_list.append(acc)

    accuracy_list = np.asarray(accuracy_list)
    iter_acc_matrix[k, :] = accuracy_list

y_arr_10_hrs = np.zeros(len(iterations_arr))
y_err_10_hrs = np.zeros(len(iterations_arr))

for k in range(len(iterations_arr)):
    y_arr_10_hrs[k] = np.mean(iter_acc_matrix[:, k])
    y_err_10_hrs[k] = np.std(iter_acc_matrix[:, k])

# 20-input MAC
input_size = 10
input_length = 20

iter_acc_matrix = np.zeros((input_size, len(iterations_arr)))

for k in range(input_size):

    weights = 5 * np.random.random(input_length)
    inputs = np.random.randint(0, precision + 1, size=input_length)
    actual_res = np.sum(np.multiply(weights, inputs))
    accuracy_list = []

    for p in range(len(iterations_arr)):
        result = f.MAC_16bit(weights, inputs, resistance_val, thresholds_16bit, iterations_arr[p])
        acc = 100 * (1 - (np.abs(actual_res - result) / actual_res))
        accuracy_list.append(acc)

    accuracy_list = np.asarray(accuracy_list)
    iter_acc_matrix[k, :] = accuracy_list

y_arr_20_hrs = np.zeros(len(iterations_arr))
y_err_20_hrs = np.zeros(len(iterations_arr))

for k in range(len(iterations_arr)):
    y_arr_20_hrs[k] = np.mean(iter_acc_matrix[:, k])
    y_err_20_hrs[k] = np.std(iter_acc_matrix[:, k])

# 50-input MAC
input_size = 10
input_length = 50

iter_acc_matrix = np.zeros((input_size, len(iterations_arr)))

for k in range(input_size):

    weights = 5 * np.random.random(input_length)
    inputs = np.random.randint(0, precision + 1, size=input_length)
    actual_res = np.sum(np.multiply(weights, inputs))
    accuracy_list = []

    for p in range(len(iterations_arr)):
        result = f.MAC_16bit(weights, inputs, resistance_val, thresholds_16bit, iterations_arr[p])
        acc = 100 * (1 - (np.abs(actual_res - result) / actual_res))
        accuracy_list.append(acc)

    accuracy_list = np.asarray(accuracy_list)
    iter_acc_matrix[k, :] = accuracy_list

y_arr_50_hrs = np.zeros(len(iterations_arr))
y_err_50_hrs = np.zeros(len(iterations_arr))

for k in range(len(iterations_arr)):
    y_arr_50_hrs[k] = np.mean(iter_acc_matrix[:, k])
    y_err_50_hrs[k] = np.std(iter_acc_matrix[:, k])

plt.figure(figsize=(8, 6))
plt.xscale("log")
plt.errorbar(iterations_arr, y_arr_10_hrs, yerr=y_err_10_hrs, fmt='o', color='green', label='10 MAC')
plt.errorbar(iterations_arr, y_arr_20_hrs, yerr=y_err_20_hrs, fmt='o', color='red', label='20 MAC')
plt.errorbar(iterations_arr, y_arr_50_hrs, yerr=y_err_50_hrs, fmt='o', color='k', label='50 MAC')
plt.xticks(iterations_arr, x_labels)
plt.legend()
plt.xlabel("Number of iterations over which the result is obtained")
plt.ylabel("Range of Accuracy for a set of 10 test inputs")
plt.title("10, 20, 50 input MAC with 16-bit precision for HRS")
plt.show()
# %%
iterations_arr = np.array([1, 2, 5, 10, 25, 50])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.set_xscale("log")
ax1.errorbar(iterations_arr, y_arr_10_lrs, yerr=y_err_10_lrs, fmt='o', color='green', label='10 MAC')
ax1.errorbar(iterations_arr, y_arr_20_lrs, yerr=y_err_20_lrs, fmt='o', color='red', label='20 MAC')
ax1.errorbar(iterations_arr, y_arr_50_lrs, yerr=y_err_50_lrs, fmt='o', color='k', label='50 MAC')
ax1.set_xticks(iterations_arr)
# ax1.legend()
ax1.title.set_text("(i) Using LRS data")

ax2.errorbar(iterations_arr, y_arr_10_hrs, yerr=y_err_10_hrs, fmt='o', color='green', label='10 MAC')
ax2.errorbar(iterations_arr, y_arr_20_hrs, yerr=y_err_20_hrs, fmt='o', color='red', label='20 MAC')
ax2.errorbar(iterations_arr, y_arr_50_hrs, yerr=y_err_50_hrs, fmt='o', color='k', label='50 MAC')
ax2.set_xticks(iterations_arr)
ax2.set_xticklabels(x_labels)
ax2.legend()
ax2.set_xlabel("Number of iterations over which the result is obtained")
# plt.ylabel("Range of Accuracy for a set of 10 test inputs")
fig.text(0.04, 0.5, 'Range of Accuracy for a set of 10 test inputs', va='center', rotation='vertical')
ax2.title.set_text("(ii) Using HRS data")
fig.suptitle("10, 20, 50 input MAC with 16-bit precision")
plt.subplots_adjust(left=0.11)
plt.show()

data = {
    'Iterations_arr': iterations_arr,
    'y_arr_10_lrs': y_arr_10_lrs,
    'y_err_10_lrs': y_err_10_lrs,
    'y_arr_20_lrs': y_arr_20_lrs,
    'y_err_20_lrs': y_err_20_lrs,
    'y_arr_50_lrs': y_arr_50_lrs,
    'y_err_50_lrs': y_err_50_lrs,
    'y_arr_10_hrs': y_arr_10_hrs,
    'y_err_10_hrs': y_err_10_hrs,
    'y_arr_20_hrs': y_arr_20_hrs,
    'y_err_20_hrs': y_err_20_hrs,
    'y_arr_50_hrs': y_arr_50_hrs,
    'y_err_50_hrs': y_err_50_hrs
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel('output_new.xlsx', index=False)