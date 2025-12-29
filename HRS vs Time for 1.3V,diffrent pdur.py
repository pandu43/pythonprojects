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
import matplotlib.cm as cm

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
alpha_An2_Reset = 2.2e-9# 1.5e-9
alpha_An1_Reset = 0 # 1e14
ResetExpo_Reset = 0#4.2

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

Rth0= 10#3e6
A= 0#7e6

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

output_folder = "Time vs HRS data"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_folder = os.path.join(output_folder, timestamp)
os.makedirs(output_folder, exist_ok=True)
# Initialize DataFrames

fwreset_df = pd.DataFrame()
fwreset_resistance_values = []

NoOfColdspots = 100
NoOfHotspots = 1050000
pulse_amplitude = 1.3
pulse_durations = [1e-8,1e-7,1e-6,1e-5]
all_hrs = {}

offset_factor = 50  # vertical offset between layers
# Reverse the order (so tpw[-1] goes on top)
ordered_pulse_durations = pulse_durations[::-1]
#colors = cm.viridis(np.linspace(0.2, 0.8, len(pulse_durations)))  # rainbow colormap
# Custom dark colors for each layer
colors = ['black', 'red', 'green', 'blue']
# ✅ Each subplot has separate y-scale (IMPORTANT)
fig, axes = plt.subplots(len(ordered_pulse_durations), 1,
                         figsize=(8, 10), sharex=True, sharey=False)
axes = np.array(axes).flatten()

# DEFINE SEPARATE Y-LIMITS (edit these for your case)
y_limits = [
    (1e3,1e4),
    (1e3, 1e4),   # for tpw = 1e-6
    (1e3, 1e4),   # for tpw = 1e-7
    (1e3, 1e9)    # for tpw = 1e-8
]
# === Loop over durations ===
for idx, pdur in enumerate(pulse_durations):
    resistance_dfs = []   # store resistance traces for all cycles
    HRS_values = []
    ax = axes[idx]
    for i in range(num_outputs):
        print("ComplianceCurrent",ComplianceCurrent)
        print ("cycle number: ", i)
        result_reset = random_reset()
        CurrentSeries_FWRESET = []
        TimeSeries_FWRESET = []
        NoOfHotspotsSeries_FWRESET = []
        NoOfColdspotsSeries_FWRESET = []
        AppliedBiasSeries_FWRESET =[]

        formationrate_Reset = []
        annhilationrate_Reset = []
        Voltage_series_Set = []
        Voltage_series_Reset = []
        tau_set_series = []
        Rth_series_reset = []

        timeReset,NoOfColdspotsFR,NoOfHotspotsFR,AppliedBiasFR = KmcProcess_FWReset (RampRate,
                                                                                    Thickness, NoOfColdspots,
                                                                                    NoOfHotspots,NoOfVacancies,
                                                                                    Eb_Cr,alpha_Cr_Reset,pulse_amplitude,
                                                                                    pdur,
                                                                                    alpha_An1_Reset,alpha_An2_Reset,
                                                                                    ResetExpo_Reset,Eb_An)

        # Convert to numpy arrays for plotting
        time_array = np.array(TimeSeries_FWRESET)
        resistance = pulse_amplitude / np.array(CurrentSeries_FWRESET)
        ax.loglog(time_array, resistance, '-', color=colors[idx])

        # APPLY DIFFERENT Y-LIMIT TO EACH PANEL
    ax.set_ylim(y_limits[idx])

    def format_pdur(pdur):
        if pdur >= 1e-3:
            return f"{pdur * 1e3:.0f}\\,\\mathrm{{ms}}"
        elif pdur >= 1e-6:
            return f"{pdur * 1e6:.0f}\\,\\mathrm{{\mu s}}"
        elif pdur >= 1e-9:
            return f"{pdur * 1e9:.0f}\\,\\mathrm{{ns}}"
        else:
            return f"{pdur * 1e12:.0f}\\,\\mathrm{{ps}}"

    # In the plot
    ax.text(0.05, 0.85, rf"$t_{{pw}} = {format_pdur(pdur)}$",
            transform=ax.transAxes, fontsize=14, fontweight="bold", color=colors[idx])

    #ax.grid(True, which='both', axis='y', ls='--', alpha=0.5)

plt.xlim(1e-9, 1.2e-5)   # ✅ set x-axis range
# Common x-label
#axes[-1].set_xlabel("Time (s)", fontsize=14, fontweight="bold")
# Adjust spacing so y-label is outside plot
# Adjust spacing: move plots right to create space for y-label
#fig.subplots_adjust(left=0.15, right=0.95, hspace=0.3)
# Add one common y-axis label
#fig.text(0.03, 0.5, "Resistance (Ω)", va='center', rotation='vertical',
       #  fontsize=24, fontweight='bold')
# ✅ Apply bold ticks to all layers
for ax in axes:
    ax.tick_params(axis='both', labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

plt.tight_layout()
plt.show()
