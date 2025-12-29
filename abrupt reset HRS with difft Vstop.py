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
from scipy.special import erf

DeltaT = 1e-4 # sec
DeltaV = 5e-5  # V
RampRate = DeltaV / DeltaT
MaxNoSteps = 1000
Eb_Cr = 3.4#  3.2 ev,3.2
Eb_An = 2.6# 3.0  ev,2.8

#Rth_Set = 5e7 #5e7
nu0_Set = 1e5 #1e5
nu01_Set = 1e3 #1e3
alpha_Cr_Set = 8e-10 #9.5e-10
alpha_An2_Set = 1e-10   # 1e-9
alpha_An1_Set = 1e2 # 1e14
ResetExpo_Set = 4.2#4.2

#Rth_Reset = 5e6#5e6
nu0_Reset = 1e5  #1e5
nu01_Reset = 1e3#1e3
alpha_Cr_Reset = 8e-10#9.5e-10
alpha_An2_Reset = 2.3e-9# 1.5e-9,2.5e-9
alpha_An1_Reset = 1e2# 1e14,1e10
ResetExpo_Reset = 4.2#4.2,5.7
vstop_values = [ 1,1.05,1.1]

Thickness = 1E-9
#ComplianceCurrent = 2e-3
#NoOfSulphurAtoms = int(70E6)  # in 5umx5um there are 25E7 Sulphur atoms
#ComplianceCurrent = 5e-1
NoOfSulphurAtoms = int(3.5e9)  # in 5umx5um there are 25E7 Sulphur atoms ,3.5e9
print('total no of sulfur atoms',NoOfSulphurAtoms)
PercentageOfVacancies = 1.5 / 100
ExpectedNumberOfVacancies = int(PercentageOfVacancies * NoOfSulphurAtoms)
G0 = 5E-10# S
print("ExpectedNumberOfVacancies ", ExpectedNumberOfVacancies)
num_outputs = int(input( "Enter the number of required outputs : "))
event_flag_list = []
event_flag_list_Reset = []
Rth0=3e3#3e6,10
A= 7e3#7e6 ,1e1

y0 = -1.66432
A  = 1.35558e3
xc = 6.57341
w  = 0.38959
a3 = 50.29355
a4 = 2050.27159

fac3 = 3*2*1
fac4 = 4*3*2*1
fac6 = 6*5*4*3*2*1

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
    Prob = ExpectedNumberOfVacancies / NoOfSulphurAtoms
    ActualNumberOfVacancies = 0
    for i in range(0, NoOfSulphurAtoms):
        r = np.random.uniform(low=0.0, high=1.0, size=None)
        if r <= Prob:
          ActualNumberOfVacancies = ActualNumberOfVacancies + 1
    return (ActualNumberOfVacancies)

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

def calculate_Rset(dataframe):
    # Find the index where X-axis value is closest to 0.5
    idx = (np.abs(dataframe['X-axis'] - 0.1)).idxmin()
    # Retrieve the corresponding Y-axis value
    y_value = dataframe.at[idx, 'Y-axis']
    # Calculate R
    Rset = 0.1 / y_value
    return Rset

def calculate_Rrst(dataframe):
    # Find the index where X-axis value is closest to -0.5
    idx = (np.abs(dataframe['X-axis'] + 0.1)).idxmin()
    # Retrieve the corresponding Y-axis value
    y_value = dataframe.at[idx, 'Y-axis']
    # Calculate R
    Rrst = 0.1 / y_value
    return Rrst
# Sulphur moves into the Molybdenum plane
#sulphur moves into the Molybdenum plane
def KmcProcess_FWReset(RampRate, Thickness, NoOfColdspots , NoOfHotspots,NoOfVacancies,
                   Eb_Cr, alpha_Cr_Reset, StartBias, alpha_An1_Reset,alpha_An2_Reset, ResetExpo_Reset, Eb_An,vstop_values):

    StepIndex = 0
    AppliedBias = StartBias
    AppliedBiasOld = StartBias
    ElectricField = 0
    time = 0
    k = 0
    step = 0


    while (AppliedBias <= vstop_values):

        StepIndex = StepIndex + 1
        CurrentTimeStep = k * DeltaT
        NextIncrementTime = (k + 1) * DeltaT
        AppliedBias = CurrentTimeStep * RampRate

       # if StepIndex > 1e7:
        #    print("Terminating due to excessive iterations")
           # break
        if ((AppliedBias - AppliedBiasOld) > 1E-5):
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

        formationrate_Reset.append(R1_Reset)
        annhilationrate_Reset.append(R2_Reset)
        Voltage_series_Reset.append(-AppliedBias)
        Rth_series_reset.append(Rth)
        r = np.random.rand()


        if (time + tau <= CurrentTimeStep + DeltaT):

            event_flag_Reset =1

            if r < (R1_Reset / Rate_Reset):
                upper_bound = max(min(1e4, NoOfColdspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)
                #random_numbers=1
                if NoOfColdspots > random_numbers:

                   NoOfColdspots -= random_numbers
                   NoOfHotspots += random_numbers
                   #print(">> R_hot event: Hotspot formed")

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
                #random_numbers =1
                # print(random_numbers)
                if (NoOfHotspots > random_numbers):

                     NoOfColdspots += random_numbers
                     NoOfHotspots -= random_numbers
                     #print(">> R_cold event: Hotspot removed")

                     if NoOfHotspots == 0:
                         print(
                             f"⚠️ Coldspots dropped to zero at Step {StepIndex}, Bias = {AppliedBias:.3f} V, Time = {time:.2e} s")
                         break

                else:
                     print("!! Skipped R_cold event: No hotspots available")
                     break

            time += tau
        else:
            event_flag_Reset = 0

            k += 1
            time = NextIncrementTime

        AppliedBiasOld = AppliedBias

        event_flag_list_Reset.append(event_flag_Reset)
        CurrentSeries_FWRESET.append(CurrentCalculation(AppliedBias, NoOfHotspots))
        TimeSeries_FWRESET.append(-time)
        NoOfHotspotsSeries_FWRESET.append(NoOfHotspots)
        AppliedBiasSeries_FWRESET.append(AppliedBias)
        NoOfColdspotsSeries_FWRESET.append(NoOfColdspots)

    print("Simulation completed: AppliedBias reached 4V after RESET.")
    print("Reset Process Summary:")
    print("\ttime = ", time)
    print("\tNumber of Coldspots ", NoOfColdspots)
    print("\tNumber of Hotspots ", NoOfHotspots)
    print("appliedbias value", AppliedBias)

    return time,NoOfColdspots , NoOfHotspots , AppliedBias

def KmcProcess_BWReset (RampRate, MaxNoSteps, Thickness, NoOfColdspots , NoOfHotspots , StartBias):

    StepIndex = 0
    AppliedBias = StartBias
    AppliedBiasOld = StartBias
    ElectricField = 0
    time = 0
    k = 0
    while (AppliedBias > 0):
        StepIndex = StepIndex + 1
        CurrentTimeStep = k * DeltaT
        NextIncrementTime = (k + 1) * DeltaT
        AppliedBias = StartBias - CurrentTimeStep * RampRate

        k = k + 1
        time = NextIncrementTime
        AppliedBiasOld = AppliedBias

        CurrentSeries_BWRESET.append(CurrentCalculation(AppliedBias, NoOfHotspots))
        TimeSeries_BWRESET.append(-time)
        NoOfHotspotsSeries_BWRESET.append(NoOfHotspots)
        NoOfColdspotsSeries_BWRESET.append(NoOfColdspots)
        AppliedBiasSeries_BWRESET.append(AppliedBias)

    print("Backward Reset Process Summary:")
    print("\ttime = ", time)
    print("\tNumber of Coldspots ", NoOfColdspots )
    print("\tNumber of Hotspots ",  NoOfHotspots)
    print('Debug - End of Loop: AppliedBias =', AppliedBias)
    print('Debug - End of Loop: AppliedBiasOld =', AppliedBiasOld)

    return time,NoOfColdspots , NoOfHotspots,AppliedBias

i = 0
temp = []
i = i + 1
InitialNoOfColdspots = ExpectedNumberOfVacancies
NoOfVacancies = InitialNoOfColdspots

print("NoOfVacancies: ", NoOfVacancies)
output_folder = "HRS data for difft vstop"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_folder = os.path.join(output_folder, timestamp)
os.makedirs(output_folder, exist_ok=True)
# Initialize DataFrames

fwreset_df = pd.DataFrame()
bwreset_df = pd.DataFrame()
fwreset_resistance_dict = {}
bwreset_resistance_dict = {}

Rth_series =[]
NoOfColdspots_value=[]
hotspots_value=[]
for vstop in vstop_values:
    fwreset_resistance_values = []
    bwreset_resistance_values = []

    for i in range(num_outputs):
        NoOfVacancies = InitialNoOfColdspots
        NoOfHotspots = generate_hotspots(num_outputs)[i]
        NoOfColdspots = NoOfVacancies - NoOfHotspots
        hotspots_value.append(NoOfHotspots)
        NoOfColdspots_value.append(NoOfColdspots)
        # Convert to numpy arrays
        print("NoOfHotspots",NoOfHotspots)
        print("NoOfColdspots", NoOfColdspots)
        print ("cycle number: ", i)

        result_reset = random_reset()
        CurrentSeries_FWSET = []
        TimeSeries_FWSET = []
        NoOfHotspotsSeries_FWSET = []
        NoOfColdspotsSeries_FWSET = []
        AppliedBiasSeries_FWSET = []

        CurrentSeries_BWSET = []
        TimeSeries_BWSET = []
        NoOfHotspotsSeries_BWSET = []
        NoOfColdspotsSeries_BWSET = []
        AppliedBiasSeries_BWSET = []

        NoOfHotspotsSeries_FWRESET = []
        NoOfColdspotsSeries_FWRESET = []
        AppliedBiasSeries_FWRESET =[]
        TimeSeries_FWRESET = []
        CurrentSeries_FWRESET=[]

        NoOfHotspotsSeries_BWRESET = []
        NoOfColdspotsSeries_BWRESET = []
        AppliedBiasSeries_BWRESET = []
        TimeSeries_BWRESET = []
        CurrentSeries_BWRESET = []

        formationrate_Set = []
        annhilationrate_Set = []
        formationrate_Reset = []
        annhilationrate_Reset = []
        Voltage_series_Set = []
        Voltage_series_Reset = []

        event_flag_list = []
        tau_set_series = []
        Rth_series_reset = []

       # NoOfHotspots = 3e7
        AppliedBias = 0

        timeReset,NoOfColdspotsFR,NoOfHotspotsFR,AppliedBiasFR = KmcProcess_FWReset (RampRate,
                                                                                    Thickness, NoOfColdspots,
                                                                                    NoOfHotspots,NoOfVacancies,
                                                                                    Eb_Cr,alpha_Cr_Reset,
                                                                                    AppliedBias,
                                                                                    alpha_An1_Reset,alpha_An2_Reset,
                                                                                    ResetExpo_Reset,Eb_An,vstop)

        timeReset,NoOfColdspotsBR,NoOfHotspotsBR,AppliedBiasBR = KmcProcess_BWReset (RampRate,
                                                                                     MaxNoSteps,
                                                                                     Thickness,
                                                                                     NoOfColdspotsFR,
                                                                                     NoOfHotspotsFR,
                                                                                     AppliedBiasFR)


        plt.semilogy(RampRate * np.array(TimeSeries_FWRESET), CurrentSeries_FWRESET, '-', color='blue')
        plt.semilogy(-AppliedBiasFR - RampRate * np.array(TimeSeries_BWRESET), CurrentSeries_BWRESET, '--',
                     color='blue')

        # Extract resistance
        fwreset_output_df = pd.DataFrame({'X-axis': RampRate * np.array(TimeSeries_FWRESET),
                                          'Y-axis': CurrentSeries_FWRESET})
        bwreset_output_df = pd.DataFrame({'X-axis': -AppliedBiasFR - RampRate * np.array(TimeSeries_BWRESET),
                                          'Y-axis': CurrentSeries_BWRESET})

        R_fwreset = calculate_Rrst(fwreset_output_df)
        R_bwreset = calculate_Rrst(bwreset_output_df)

        fwreset_resistance_values.append(R_fwreset)
        bwreset_resistance_values.append(R_bwreset)

        # Save list under this vstop
    fwreset_resistance_dict[f"Vstop={vstop}"] = fwreset_resistance_values
    bwreset_resistance_dict[f"Vstop={vstop}"] = bwreset_resistance_values

# 🚀 Build DataFrames: each column = one Vstop
fwreset_resistance_df = pd.DataFrame(fwreset_resistance_dict)
bwreset_resistance_df = pd.DataFrame(bwreset_resistance_dict)

# Save CSVs
fwreset_resistance_df.to_csv(os.path.join(output_folder, "fwreset_resistance.csv"), index=False)
bwreset_resistance_df.to_csv(os.path.join(output_folder, "bwreset_resistance.csv"), index=False)

print("FW Reset resistances:\n", fwreset_resistance_df.head())
print("BW Reset resistances:\n", bwreset_resistance_df.head())

# Sort values
sorted_hotspot_values = np.sort(hotspots_value)
sorted_coldspot_values = np.sort(NoOfColdspots_value)
# Compute individual CDFs
cdf_hotspot = np.arange(1, len(sorted_hotspot_values) + 1) / len(sorted_hotspot_values) * 100
cdf_coldspot = np.arange(1, len(sorted_coldspot_values) + 1) / len(sorted_coldspot_values) * 100
# Plot
plt.figure(figsize=(6, 4))
plt.plot(sorted_hotspot_values, cdf_hotspot, color="green", linewidth=2, label="Simulated Hotspot CDF")
plt.xscale("log")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability (%)")
plt.title("CDF of Hotspots ")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.figure(figsize=(6, 4))
plt.plot(sorted_coldspot_values, cdf_coldspot, color="red", linewidth=2, label="Simulated Coldspot CDF")
plt.xscale("log")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability (%)")
plt.title("CDF of Coldspots")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()



#plt.legend()
plt.show()
end_time = tm.time()
print(f"✅ Simulation completed in {end_time - start_time:.2f} seconds")


