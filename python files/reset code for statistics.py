import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import sys
import time as tm
from scipy.special import erf
start_time = tm.time()

DeltaT = 1e-4 # sec
DeltaV = 5e-5  # V
RampRate = DeltaV / DeltaT
MaxNoSteps = 1000
Eb_Cr = 3.2 #  3.2 ev
Eb_An = 3.0# 3.4  ev

#Rth_Set = 5e7 #5e7
nu0_Set = 1e5   #1e5
nu01_Set = 1e3  #1e3
alpha_Cr_Set = 1e-9  #9.5e-10
alpha_An2_Set = 1e-9   # 1e-9
alpha_An1_Set = 1e14  # 1e14
ResetExpo_Set = 4.2   #4.2

#Rth_Reset = 5e6#5e6
nu0_Reset = 1e5  #1e5
nu01_Reset = 1e3 #1e3
alpha_Cr_Reset = 9.5e-10#9.5e-10
alpha_An2_Reset = 4e-9  # 1.5e-9
alpha_An1_Reset = 1e14  # 1e14
ResetExpo_Reset = 4.2 #4.2

Thickness = 1E-9
ComplianceCurrent = 1e-2
#NoOfSulphurAtoms = int(70E6)  # in 5umx5um there are 25E7 Sulphur atoms
NoOfSulphurAtoms = int(3.5E8)  # in 5umx5um there are 25E7 Sulphur atoms
print('total no of sulfur atoms',NoOfSulphurAtoms)
PercentageOfVacancies = 1.5 / 100
ExpectedNumberOfVacancies = int(PercentageOfVacancies * NoOfSulphurAtoms)
G0 = 5E-10# S
print("ExpectedNumberOfVacancies ", ExpectedNumberOfVacancies)
num_outputs = int(input( "Enter the number of required outputs : "))

event_flag_list = []
event_flag_list_Reset = []

Rth0= 3e6
A= 1e7

def calculate_piecewise_Rth(NoOfVacancies,NoOfHotspots,Rth0, A):
  Rth = Rth0 + A * (np.log10(NoOfVacancies) - np.log10(NoOfHotspots))
  return Rth

# Parameters for the GaussMod model
y0 = -149.58917
A = 564.27985
xc = 3.09437
w = 0.11387
t0 = 0.62164

# Define the GaussMod function for resistance values
def gauss_mod(x, y0, A, xc, w, t0):
    z = (x - xc) / w - w / t0
    y = y0 + A / t0 * np.exp(0.5 * (w / t0) ** 2 - (x - xc) / t0) * (erf(z / np.sqrt(2)) + 1) / 2
    return y

def generate_compliance_current(num_outputs):
    # Define range of resistance values in log10(R)
    x_values = np.linspace(3.0, 4.4,num_outputs )
    # Calculate the PDF values using the GaussMod function
    pdf_values = gauss_mod(x_values, y0, A, xc, w, t0)
    # Handle single output case separately
    if num_outputs == 1:
        return np.power(10, -x_values)  # Return the single current value directly
    # Normalize the PDF values and convert to counts
    counts = (pdf_values - np.min(pdf_values))  # Shift to make minimum zero (to avoid negative values)
    if np.max(counts) != 0:  # Avoid division by zero
        counts = (counts / np.max(counts))*num_outputs # Scale counts for visualization (e.g., max value set to 1000)
    # Convert x (resistance log10(R)) values to current (I) values in linear scale
    I_values = np.power(10, -x_values)  # I = 10^(-log10(R))
    compliance_current = np.random.choice(I_values, size=num_outputs, p=counts / np.sum(counts))
    #print("compliance_current", compliance_current)
    return compliance_current

def random_reset():
    result = np.random.normal(0.5, 0.5)
    if result <= 0.5:
        return 0
    else:
        return 1

def InitializeTheSulphurVacancies(NoOfSulphurAtoms, ExpectedNumberOfVacancies):
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


def KmcProcess_FWReset(RampRate, Thickness, NoOfColdspots , NoOfHotspots,NoOfVacancies,
                   Eb_Cr, alpha_Cr_Reset, StartBias, alpha_An1_Reset,alpha_An2_Reset, ResetExpo_Reset, Eb_An):

    StepIndex = 0
    AppliedBias = StartBias
    AppliedBiasOld = StartBias
    ElectricField = 0
    time = 0
    k = 0
    step = 0


    while (AppliedBias <= 1.95):

        StepIndex = StepIndex + 1
        CurrentTimeStep = k * DeltaT
        NextIncrementTime = (k + 1) * DeltaT
        AppliedBias = CurrentTimeStep * RampRate

       # if StepIndex > 1e0:
           # print("Terminating due to excessive iterations")
            #break
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
NoOfVacancies = InitializeTheSulphurVacancies(NoOfSulphurAtoms, ExpectedNumberOfVacancies)
#InitialNoOfColdspots = ExpectedNumberOfVacancies
#print("Initial No Of Coldspots: ", InitialNoOfColdspots)

output_folder = "HRS data for difft vstop"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_folder = os.path.join(output_folder, timestamp)
os.makedirs(output_folder, exist_ok=True)
# Initialize DataFrames
fwset_df = pd.DataFrame()
Bwset_df = pd.DataFrame()
fwreset_df = pd.DataFrame()
bwreset_df = pd.DataFrame()

fwset_resistance_values = []
Bwset_resistance_values = []
fwreset_resistance_values = []
bwreset_resistance_values = []

Rth_series =[]


for i in range(num_outputs):

    ComplianceCurrent = generate_compliance_current(num_outputs)[i]
    print("ComplianceCurrent",ComplianceCurrent)
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

    CurrentSeries_FWRESET = []
    TimeSeries_FWRESET = []
    NoOfHotspotsSeries_FWRESET = []
    NoOfColdspotsSeries_FWRESET = []
    AppliedBiasSeries_FWRESET =[]


    CurrentSeries_BWRESET = []
    TimeSeries_BWRESET = []
    NoOfHotspotsSeries_BWRESET = []
    NoOfColdspotsSeries_BWRESET = []
    AppliedBiasSeries_BWRESET = []


    formationrate_Set = []
    annhilationrate_Set = []
    formationrate_Reset = []
    annhilationrate_Reset = []
    Voltage_series_Set = []
    Voltage_series_Reset = []

    event_flag_list = []
    tau_set_series = []
    Rth_series_reset = []

    NoOfColdspots = 10
   # NoOfHotspots = 1050000
    NoOfHotspots = 5250000
    AppliedBias = 0

    timeReset,NoOfColdspotsFR,NoOfHotspotsFR,AppliedBiasFR = KmcProcess_FWReset (RampRate,
                                                                                Thickness, NoOfColdspots,
                                                                                NoOfHotspots,NoOfVacancies,
                                                                                Eb_Cr,alpha_Cr_Reset,
                                                                                AppliedBias,
                                                                                alpha_An1_Reset,alpha_An2_Reset,
                                                                                ResetExpo_Reset,Eb_An)

    timeReset,NoOfColdspots,NoOfHotspots,AppliedBias = KmcProcess_BWReset (RampRate,
                                                                                 MaxNoSteps,
                                                                                 Thickness,
                                                                                 NoOfColdspotsFR,
                                                                                 NoOfHotspotsFR,
                                                                                 AppliedBiasFR)

    fwreset_x_axis = RampRate * np.array(TimeSeries_FWRESET)
    bwreset_x_axis = -AppliedBiasFR - RampRate * np.array(TimeSeries_BWRESET)

    fwreset_data = {'X-axis': fwreset_x_axis, 'Y-axis':CurrentSeries_FWRESET }
    fwreset_output_df = pd.DataFrame(fwreset_data)

    bwreset_data = {'X-axis': bwreset_x_axis , 'Y-axis': CurrentSeries_BWRESET }
    bwreset_output_df = pd.DataFrame(bwreset_data)

    # Calculate resistances

    R_fwreset = calculate_Rrst(fwreset_output_df)
    R_bwreset = calculate_Rrst(bwreset_output_df)

    # Print resistance values
    # print("R for fwset_output_df:", R_fwset)
    # print("R for Bwset_output_df:", R_Bwset)
    # print("R for forward_output_df:", R_forward)
    # print("R for backward_output_df:", R_backward)

    # Append resistance values to lists
    fwreset_resistance_values.append(R_fwreset)
    bwreset_resistance_values.append(R_bwreset)

    # Reduce the size of DataFrames by selecting every 700th row
    fwreset_output_df = fwreset_output_df.iloc[::2, :]
    bwreset_output_df = bwreset_output_df.iloc[::2, :]

    # Concatenate DataFrames
    fwreset_df = pd.concat([fwreset_df, fwreset_output_df], axis=1)
    bwreset_df = pd.concat([bwreset_df, bwreset_output_df], axis=1)

    plt.figure(1)
    plt.semilogy(RampRate * np.array(TimeSeries_FWRESET), CurrentSeries_FWRESET, '-', color='blue')
    plt.semilogy(-AppliedBiasFR - RampRate * np.array(TimeSeries_BWRESET), CurrentSeries_BWRESET, '--', color='blue')

    plt.xlabel('Voltage (V)')
    plt.ylabel('  Current(A)')

fwreset_file_path = os.path.join(output_folder, f"fwreset_data_{timestamp}.csv")
fwreset_df.to_csv(fwreset_file_path, index=False)

bwreset_file_path = os.path.join(output_folder, f"bwreset_data_{timestamp}.csv")
bwreset_df.to_csv(bwreset_file_path, index=False)

    # Create DataFrames for resistance values and save them to CSV files
fwreset_resistance_df = pd.DataFrame({'Resistance': fwreset_resistance_values})
bwreset_resistance_df = pd.DataFrame({'Resistance': bwreset_resistance_values})

fwreset_resistance_df.to_csv(os.path.join(output_folder, 'fwreset_resistance.csv'), index=False)
bwreset_resistance_df.to_csv(os.path.join(output_folder, 'bwreset_resistance.csv'), index=False)

print(f"{num_outputs} files have been generated and saved in the folder: {output_folder}")
print(f"Output folder location: {os.path.abspath(output_folder)}")

#plt.legend()
plt.show()
end_time = tm.time()
print(f"✅ Simulation completed in {end_time - start_time:.2f} seconds")


