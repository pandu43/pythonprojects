import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import sys
import time as tm
start_time = tm.time()
from matplotlib.colors import LogNorm

DeltaT = 1e-4 # sec
DeltaV = 5e-5  # V
RampRate = DeltaV / DeltaT
MaxNoSteps = 1000
Eb_Cr = 3.2#  3.2 ev
Eb_An = 3# 3.0  ev

#Rth_Set = 5e7 #5e7
nu0_Set = 1e5   #1e5
nu01_Set = 1e3  #1e3
alpha_Cr_Set = 9.5e-10 #9.5e-10
alpha_An2_Set = 1e-9   # 1e-9
alpha_An1_Set = 1e14  # 1e14
ResetExpo_Set = 4.2   #4.2

#Rth_Reset = 5e6#5e6
nu0_Reset = 1e5  #1e5
nu01_Reset = 5e4 #1e3
alpha_Cr_Reset = 9.5e-10#9.5e-10
alpha_An2_Reset = 1.5e-9# 1.5e-9
alpha_An1_Reset = 1e14 # 1e14
ResetExpo_Reset = 4.2#4.2

Thickness = 1E-9
ComplianceCurrent = 2e-2
NoOfSulphurAtoms = int(70E6)  # in 5umx5um there are 25E7 Sulphur atoms
#NoOfSulphurAtoms = int(3.5E9)
print('total no of sulfur atoms',NoOfSulphurAtoms)
PercentageOfVacancies = 1.5 / 100
ExpectedNumberOfVacancies = int(PercentageOfVacancies * NoOfSulphurAtoms)
G0 = 5E-10# S
print("ExpectedNumberOfVacancies ", ExpectedNumberOfVacancies)
num_outputs = int(input( "Enter the number of required outputs : "))

event_flag_list = []
event_flag_list_Reset = []

Rth0= 3e6#3e6
A= 7e6#7e6

def calculate_piecewise_Rth(NoOfVacancies,NoOfHotspots,Rth0, A):
  Rth = Rth0 + A * (np.log10(NoOfVacancies) - np.log10(NoOfHotspots))
  return Rth

def random_reset():
    result = np.random.normal(0.5, 0.5)
    if result <= 0.5:
        return 0
    else:
        return 1

#def InitializeTheSulphurVacancies(NoOfSulphurAtoms, ExpectedNumberOfVacancies):
   # Prob = ExpectedNumberOfVacancies / NoOfSulphurAtoms
   # ActualNumberOfVacancies = 0
   # for i in range(0, NoOfSulphurAtoms):
  #      r = np.random.uniform(low=0.0, high=1.0, size=None)
    #    if r <= Prob:
    #        ActualNumberOfVacancies = ActualNumberOfVacancies + 1
  #  return (ActualNumberOfVacancies)

def CalculateTheHotspotFormationRate_Set(ElectricField, Eb_Cr, alpha_Cr_Set, AppliedBias, Current ,kBT_q_Set):
   # kBT_q = calculate_kBT_qSet(AppliedBias, Current, Rs)
    expo = (-(Eb_Cr - alpha_Cr_Set * ElectricField) / kBT_q_Set)
    if expo > 0:
        Rate = nu0_Set
    else:
        Rate = nu0_Set * np.exp(expo)
    return (Rate)

def CalculateTheHotspotAnnihilationRate_Set(ElectricField, AppliedBias, Current, alpha_An1_Set,
        ResetExpo_Set, alpha_An2_Set, Eb_An, kBT_q_Set):

    expo = (-(Eb_An - alpha_An1_Set * pow((AppliedBias * Current),
                                          ResetExpo_Set) - alpha_An2_Set * ElectricField) / kBT_q_Set)
    if expo > 0:
        Rate = nu01_Set
    else:
        Rate = nu01_Set * np.exp(expo)  # expo is already negative
    return Rate

def CalculateTheHotspotFormationRate_Reset(ElectricField, Eb_Cr, alpha_Cr_Reset, AppliedBias, Current ,kBT_q_Reset):
   # kBT_q = calculate_kBT_qSet(AppliedBias, Current, Rs)
    expo = (-(Eb_Cr - alpha_Cr_Reset * ElectricField) / kBT_q_Reset)
    if expo > 0:
        Rate = nu0_Reset
    else:
        Rate = nu0_Reset * np.exp(expo)
    return (Rate)

def CalculateTheHotspotAnnihilationRate_Reset(ElectricField, AppliedBias, Current, alpha_An1_Reset,
        ResetExpo_Reset, alpha_An2_Reset, Eb_An, kBT_q_Reset):

    expo = (-(Eb_An - alpha_An1_Reset * pow((AppliedBias * Current),
                                          ResetExpo_Reset) - alpha_An2_Reset * ElectricField) / kBT_q_Reset)
    if expo > 0:
        Rate = nu01_Reset
    else:
        Rate = nu01_Reset * np.exp(expo)  # expo is already negative
    return Rate

def CalculateTheTimeUpdate( Rate):
    r2 = random.random()
    Rtot = Rate
    if Rtot != 0:
        tau = -np.log(r2) / Rtot
    else:
        # Handle the case when Rtot is zero
        tau = float('inf')  # Set tau to infinity or handle it based on your specific requirements
    return tau

def CurrentCalculation(AppliedBias, NoOfHotspots):
    I = G0 * AppliedBias * NoOfHotspots
    return (I)

def calculate_kBT_qSet(AppliedBias, Current, Rth):
    temperature = 300 + AppliedBias * Current * Rth
    return (8.625E-5 * temperature)

def calculate_kBT_qReset(AppliedBias, Current, Rth):
    temperature = 300 + AppliedBias * Current * Rth
    return (8.625E-5 * temperature)

def generate_pulse(time, pulse_duration,pulse_amplitude ):
    if time < pulse_duration:
        return pulse_amplitude
    else:
        return 0

def KmcProcess_FWReset(RampRate, Thickness, NoOfColdspots , NoOfHotspots,NoOfVacancies,
                   Eb_Cr, alpha_Cr_Reset, pulse_amplitude,pulse_duration, alpha_An1_Reset,alpha_An2_Reset, ResetExpo_Reset, Eb_An):

    StepIndex = 0
    ElectricField = 0
    time = 0
    k = 0
    step = 0

    while (time < pulse_duration):

        AppliedBias = generate_pulse(time, pulse_duration, pulse_amplitude)
        ElectricField = AppliedBias / Thickness

        Rth = calculate_piecewise_Rth(NoOfVacancies,NoOfHotspots,Rth0, A)

        kBT_q_Reset = calculate_kBT_qReset(AppliedBias, CurrentCalculation(AppliedBias, NoOfHotspots), Rth)

        Rate_hotspots_Reset = CalculateTheHotspotFormationRate_Reset(ElectricField, Eb_Cr, alpha_Cr_Reset, AppliedBias,
                                                                   CurrentCalculation(AppliedBias, NoOfHotspots),
                                                                   kBT_q_Reset)

        Rate_coldspots_Reset = CalculateTheHotspotAnnihilationRate_Reset(ElectricField, AppliedBias,
                                                                         CurrentCalculation(AppliedBias, NoOfHotspots),
                                                                         alpha_An1_Reset,
                                                                         ResetExpo_Reset, alpha_An2_Reset, Eb_An,
                                                                         kBT_q_Reset)

        R1_Reset = Rate_hotspots_Reset * NoOfColdspots
        R2_Reset = Rate_coldspots_Reset * NoOfHotspots
        Rate_Reset = R1_Reset + R2_Reset
        tau = CalculateTheTimeUpdate(Rate_Reset)
        current = CurrentCalculation(AppliedBias, NoOfHotspots)
        r = np.random.rand()

        if (time + tau <= pulse_duration):

            if r < (R1_Reset / Rate_Reset):
                upper_bound = max(min(1e4, NoOfColdspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)

                if NoOfColdspots > random_numbers:

                   NoOfColdspots -= random_numbers
                   NoOfHotspots += random_numbers
                #   print(">> R_hot event: Hotspot formed")

                   if NoOfColdspots == 0:
                       print(
                           f"⚠️ Coldspots dropped to zero at Step {StepIndex}, Bias = {AppliedBias:.3f} V, Time = {time:.2e} s")
                       break
                else:
                    print("!! Skipped R_hot event: Not enough coldspots available")
                    # Force transition to RESET phase
            else:
                upper_bound = max(min(1e4, NoOfHotspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)

                if (NoOfHotspots > random_numbers):

                     NoOfColdspots += random_numbers
                     NoOfHotspots -= random_numbers
                  #   print(">> R_cold event: Hotspot removed")

                     if NoOfHotspots == 0:
                         print(
                             f"⚠️ Coldspots dropped to zero at Step {StepIndex}, Bias = {AppliedBias:.3f} V, Time = {time:.2e} s")
                         break
                else:
                     print("!! Skipped R_cold event: No hotspots available")
                     break

            time += tau

        else:
            k += 1
            time = pulse_duration

        #print(
         #f"time={time:.2e}, Hotspots={NoOfHotspots}")

        CurrentSeries_FWRESET.append(CurrentCalculation(AppliedBias, NoOfHotspots))
        TimeSeries_FWRESET.append(time)
        NoOfHotspotsSeries_FWRESET.append(NoOfHotspots)
        AppliedBiasSeries_FWRESET.append(AppliedBias)
        NoOfColdspotsSeries_FWRESET.append(NoOfColdspots)

    print(
        f"time={time:.2e}, Hotspots={NoOfHotspots}")
    print("Simulation completed: AppliedBias reached 4V after RESET.")
    print("Reset Process Summary:")
    print("\ttime = ", time)
    print("\tNumber of Coldspots ", NoOfColdspots)
    print("\tNumber of Hotspots ", NoOfHotspots)
    print("appliedbias value", AppliedBias)

    return time,NoOfColdspots , NoOfHotspots , AppliedBias


i = 0
temp = []
i = i + 1
#InitialNoOfColdspots = InitializeTheSulphurVacancies(NoOfSulphurAtoms, ExpectedNumberOfVacancies)
NoOfVacancies = ExpectedNumberOfVacancies
print("Initial No Of Coldspots: ", NoOfVacancies)
Rth_series =[]

output_folder = "contour plot data mean,vp,tp"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_folder = os.path.join(output_folder, timestamp)
os.makedirs(output_folder, exist_ok=True)
# ✅ Define pulse durations you want to test
NoOfColdspots = 10
NoOfHotspots = 1050000
#pulse_amplitudes = np.arange(0.9,2.5,0.1)  # Step size 0.1V
#pulse_durations = np.array([5E-9,5E-8,5E-7,5E-6,5E-5,5e-4,5E-3])
pulse_amplitudes = np.linspace(0.9, 2.4, 100)
# Choose 10 durations between 1e-9 and 1e-3 (log scale)
pulse_durations = np.logspace(-9, -3, 100)
# Initialize storage for all cycles
all_cycles_data = np.zeros((len(pulse_amplitudes), len(pulse_durations), num_outputs))

for i, pulse_amp in enumerate(pulse_amplitudes):
  print(f"Processing pulse amplitude {pulse_amp:.1f}V")
  current_row = []
  df_combined_current = pd.DataFrame()
  df_combined_hotspots = pd.DataFrame()
      # Loop through each pulse duration
  for j, pulse_dur in enumerate(pulse_durations):
    print(f"Processing pulse duration {pulse_dur:.1e}s")
    for cycle in range(num_outputs):
        print("ComplianceCurrent", ComplianceCurrent)
        print("cycle number: ", cycle)
        result_reset = random_reset()
        CurrentSeries_FWRESET = []
        TimeSeries_FWRESET = []
        NoOfHotspotsSeries_FWRESET = []
        NoOfColdspotsSeries_FWRESET = []
        AppliedBiasSeries_FWRESET = []
        formationrate_Reset = []
        annhilationrate_Reset = []
        Voltage_series_Set = []
        Voltage_series_Reset = []
        tau_set_series = []
        Rth_series_reset = []
        Rth_series = []

        timeReset,NoOfColdspotsFR,NoOfHotspotsFR,AppliedBiasFR = KmcProcess_FWReset (RampRate,
                                                                                Thickness, NoOfColdspots,
                                                                                NoOfHotspots,NoOfVacancies,
                                                                                Eb_Cr,alpha_Cr_Reset,pulse_amp,pulse_dur,
                                                                                alpha_An1_Reset,alpha_An2_Reset,
                                                                                ResetExpo_Reset,Eb_An)

    # ⚡ One HRS value per cycle → using last current in the series
        HRS_value = pulse_amp / CurrentSeries_FWRESET[-1]
        print(f"Cycle {cycle + 1}:  HRS = {HRS_value}")
        all_cycles_data[i, j, cycle] = HRS_value  # Store for later statistics

# Compute statistics across all cycles
mean_values = np.mean(all_cycles_data, axis=2)
std_values = np.std(all_cycles_data, axis=2)
cov_values = std_values / mean_values  # Coefficient of Variation
# Create meshgrid for contour plots
X, Y = np.meshgrid(pulse_amplitudes, pulse_durations, indexing='ij')

# Plot Contour for Mean
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, mean_values, cmap='rainbow', norm=LogNorm())
cbar = plt.colorbar(contour)
cbar.set_label('Mean (log scale)')
plt.yscale('log')
plt.xlabel('Pulse Amplitude (V)')
plt.ylabel('Pulse Duration (s)')
plt.title('Mean vs Pulse Amplitude and Duration')
plt.show()

# Plot Contour for Standard Deviation
plt.figure(figsize=(8, 6))
#contour = plt.contourf(X, Y, std_values, cmap='viridis', norm=LogNorm())
contour = plt.contourf(X, Y, std_values, cmap='rainbow', norm=LogNorm())
cbar = plt.colorbar(contour)
cbar.set_label('Standard Deviation (log scale)')
plt.yscale('log')
plt.xlabel('Pulse Amplitude (V)')
plt.ylabel('Pulse Duration (s)')
plt.title('Standard Deviation vs Pulse Amplitude and Duration')
plt.show()

# Plot Contour for Median
plt.figure(figsize=(8, 6))
#contour = plt.contourf(X, Y, median_values, cmap='plasma', norm=LogNorm())
contour = plt.contourf(X, Y, cov_values, cmap='rainbow', norm=LogNorm(vmin=1e-2, vmax=1e1))
cbar = plt.colorbar(contour)
cbar.set_label('CV  (log scale)')
plt.yscale('log')
plt.xlabel('Pulse Amplitude (V)')
plt.ylabel('Pulse Duration (s)')
plt.title('CV vs Pulse Amplitude and Duration')
plt.show()

# Save data to an Excel file
df_mean = pd.DataFrame(mean_values, index=[f"Amplitude_{amp}" for amp in pulse_amplitudes],
                        columns=[f"Duration_{dur}" for dur in pulse_durations])
df_std = pd.DataFrame(std_values, index=[f"Amplitude_{amp}" for amp in pulse_amplitudes],
                       columns=[f"Duration_{dur}" for dur in pulse_durations])
df_CV = pd.DataFrame(cov_values, index=[f"Amplitude_{amp}" for amp in pulse_amplitudes],
                          columns=[f"Duration_{dur}" for dur in pulse_durations])

# Define file paths for three separate Excel files
mean_filepath = os.path.join(output_folder, 'mean.xlsx')
std_filepath = os.path.join(output_folder, 'std.xlsx')
CV_filepath = os.path.join(output_folder, 'CV.xlsx')

# Save each DataFrame as a separate Excel file
df_mean.to_excel(mean_filepath, sheet_name='Mean')
df_std.to_excel(std_filepath, sheet_name='Std Dev ')
df_CV.to_excel(CV_filepath, sheet_name='CV ')

print(f"🎯 All files saved in the '{output_folder}' folder:")
print(f"- {mean_filepath}")
print(f"- {std_filepath}")
print(f"- {CV_filepath}")




