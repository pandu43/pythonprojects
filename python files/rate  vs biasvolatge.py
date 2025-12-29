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

DeltaT = 1e-4 # sec
DeltaV = 5e-5  # V
RampRate = DeltaV / DeltaT
MaxNoSteps = 1000
Eb_Cr = 3.2 #  3.2 ev
Eb_An = 3.0# 3.0  ev

#Rth_Set = 5e7 #5e7
nu0_Set = 1e5   #1e5
nu01_Set = 1e3  #1e3
alpha_Cr_Set = 9.5e-10 #9.5e-10
alpha_An2_Set = 1e-9   # 1e-9
alpha_An1_Set = 1e14  # 1e14
ResetExpo_Set = 4.2   #4.2

#Rth_Reset = 5e6#5e6
nu0_Reset = 1e5  #1e5
nu01_Reset = 1e3 #1e3
alpha_Cr_Reset = 9.5e-10#9.5e-10
alpha_An2_Reset = 1.5e-9# 1.5e-9
alpha_An1_Reset = 1e14 # 1e14
ResetExpo_Reset = 4.2 #4.2

Thickness = 1E-9
ComplianceCurrent = 2e-5
NoOfSulphurAtoms = int(70E6)  # in 5umx5um there are 25E7 Sulphur atoms
#NoOfSulphurAtoms = int(3.5E8)
print('total no of sulfur atoms',NoOfSulphurAtoms)
PercentageOfVacancies = 1.5 / 100
ExpectedNumberOfVacancies = int(PercentageOfVacancies * NoOfSulphurAtoms)
G0 = 5E-10# S
print("ExpectedNumberOfVacancies ", ExpectedNumberOfVacancies)
num_outputs = int(input( "Enter the number of required outputs : "))

NoOfHotspots = 10
AppliedBias = 0

event_flag_list = []
event_flag_list_Reset = []

Rth0= 3e6 #3e6
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
def KmcProcess_FWSet(RampRate, Thickness, NoOfColdspots , NoOfHotspots,NoOfVacancies,
                   Eb_Cr, alpha_Cr_Set, StartBias, alpha_An1_Set,alpha_An2_Set, ResetExpo_Set, Eb_An):

    StepIndex = 0
    AppliedBias = StartBias
    AppliedBiasOld = StartBias
    time = 0
    k = 0

    ElectricField = 0
    print("Initial No Of Coldspots before kMC_FWSet  ", NoOfColdspots)
    print("Initial No Of Hotspots before kMC_FWSet ", NoOfHotspots)

    while (CurrentCalculation(AppliedBias, NoOfHotspots) <= ComplianceCurrent):

        StepIndex = StepIndex + 1
        CurrentTimeStep = k * DeltaT
        NextIncrementTime = (k + 1) * DeltaT
        AppliedBias = CurrentTimeStep * RampRate

        #if StepIndex > 1e6:
           # print("Terminating due to excessive iterations")
           # break

        if ((AppliedBias - AppliedBiasOld) > 1E-5):

            ElectricField = AppliedBias / Thickness


        Rth = calculate_piecewise_Rth(NoOfVacancies,NoOfHotspots,Rth0, A)

        kBT_q_Set = calculate_kBT_qSet(AppliedBias, CurrentCalculation(AppliedBias,NoOfHotspots), Rth)

        Rate_hotspots_Set = CalculateTheHotspotFormationRate_Set(ElectricField, Eb_Cr, alpha_Cr_Set,AppliedBias,
                                                         CurrentCalculation(AppliedBias,NoOfHotspots),
                                                       kBT_q_Set )

        Rate_coldspots_Set = CalculateTheHotspotAnnihilationRate_Set(ElectricField, AppliedBias,
                                                CurrentCalculation(AppliedBias, NoOfHotspots), alpha_An1_Set,
                                                ResetExpo_Set,  alpha_An2_Set, Eb_An , kBT_q_Set)


        R1_Set = Rate_hotspots_Set * NoOfColdspots
        R2_Set = Rate_coldspots_Set * NoOfHotspots
        Rate_Set = R1_Set + R2_Set

        Rth_series.append(Rth)
        tau = CalculateTheTimeUpdate(Rate_Set)

        Current = CurrentCalculation(AppliedBias,NoOfHotspots)

      #  print(
       #     f"\nStep {StepIndex} | Bias = {AppliedBias:.4f} V |Current = {Current:.2e} Amp | Hotspots = {NoOfHotspots} | Coldspots = {NoOfColdspots}")

       # print(
      #      f"\ntau {tau} | time = {time:.4f} V |NextIncrementTime = {NextIncrementTime:.2e}")

        formationrate_Set.append(R1_Set)
        annhilationrate_Set.append(R2_Set)
        Voltage_series_Set.append(AppliedBias)
        tau_set_series.append(tau)


        r = np.random.rand()

        if (time + tau <= CurrentTimeStep + DeltaT):

            event_flag = 1  # Event occurred
           # print(f"Event occurred at Bias = {AppliedBias:.3f} V, Time = {time:.2e} ")

            if r < (R1_Set / Rate_Set):

                upper_bound = max(min(1e4, NoOfColdspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)
                #random_numbers=1

                if NoOfColdspots > random_numbers:

                   NoOfColdspots -= random_numbers
                   NoOfHotspots += random_numbers
                  # print(">> R_hot event: Hotspot formed")

                   if NoOfColdspots == 0:
                       print(
                           f"⚠️ Coldspots dropped to zero at Step {StepIndex}, Bias = {AppliedBias:.3f} V, Time = {time:.2e} s")
                       break
                else:
                   print("!! Skipped R_hot event: Not enough coldspots available")
                   break
                    # Force transition to RESET phase
            else:
                upper_bound = max(min(1e4, NoOfHotspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)
                #random_numbers=1
                # print(random_numbers)
                if ( NoOfHotspots > random_numbers ):

                   NoOfColdspots += random_numbers
                   NoOfHotspots -= random_numbers
                  # print(">> R_cold event: Hotspot removed")
                else:
                   print("!! Skipped R_cold event: No hotspots available")

            time += tau
        else:
            event_flag = 0  # Event skipped


            k += 1
            time = NextIncrementTime


        AppliedBiasOld = AppliedBias
        event_flag_list.append(event_flag)
        CurrentSeries_FWSET.append(CurrentCalculation(AppliedBias, NoOfHotspots))
        TimeSeries_FWSET.append(time)
        NoOfHotspotsSeries_FWSET.append(NoOfHotspots)
        NoOfColdspotsSeries_FWSET.append(NoOfColdspots)
        AppliedBiasSeries_FWSET.append(AppliedBias)

    print("\033[91m⚡ Compliance current reached! Resetting Bias to 0V for RESET phase.\033[0m")
    print("Forward Set Process Summary: ")
    print("\tNo Of Steps = ", StepIndex)
    print("\ttime = ", time)
    print("\tNumber of Coldspots   ", NoOfColdspots)
    print("\tNumber of Hotspots ", NoOfHotspots)
    print("electric field value", ElectricField)
    print("Applied bias value",AppliedBias)

    return time, NoOfColdspots, NoOfHotspots, AppliedBias

def KmcProcess_BwSet(RampRate, MaxNoSteps, Thickness, NoOfColdspots , NoOfHotspots , StartBias):

    time = 0
    StepIndex = 0
    AppliedBias = StartBias
    print("Applied Bias after Backward Set :", AppliedBias)
    AppliedBiasOld = StartBias
    ElectricField = 0
    k = 0
    print("Initial No Of Coldspots before kMC_BwSet  ", NoOfColdspots)
    print("Initial No Of Hotspots before kMC_BwSet ", NoOfHotspots)

    while( AppliedBias > 0 ):

        StepIndex = StepIndex + 1
        CurrentTimeStep = k * DeltaT
        NextIncrementTime = (k + 1) * DeltaT
        AppliedBias = StartBias - CurrentTimeStep * RampRate

        k = k + 1
        time = NextIncrementTime
        AppliedBiasOld = AppliedBias

        CurrentSeries_BWSET.append(CurrentCalculation(AppliedBias, NoOfHotspots))
        TimeSeries_BWSET.append(time)
        NoOfHotspotsSeries_BWSET.append(NoOfHotspots)
        NoOfColdspotsSeries_BWSET.append(NoOfColdspots)
        AppliedBiasSeries_BWSET.append(AppliedBias)

    print("Backward Set Process Summary: ")
    print("\tNo Of Steps = ", StepIndex)
    print("\ttime = ", time)
    print("\tNumber of Coldspots ",  NoOfColdspots )
    print("\tNumber of Hotspots",  NoOfHotspots)
    print('Debug - End of Loop: AppliedBias =', AppliedBias)
    print('Debug - End of Loop: AppliedBiasOld =', AppliedBiasOld)

    return time, NoOfColdspots, NoOfHotspots, AppliedBias

def KmcProcess_FWReset(RampRate, Thickness, NoOfColdspots , NoOfHotspots,NoOfVacancies,
                   Eb_Cr, alpha_Cr_Reset, StartBias, alpha_An1_Reset,alpha_An2_Reset, ResetExpo_Reset, Eb_An):

    StepIndex = 0
    AppliedBias = StartBias
    AppliedBiasOld = StartBias
    ElectricField = 0
    time = 0
    k = 0
    step = 0


    while (AppliedBias <= 2):

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

                #random_numbers=1

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

               # random_numbers =1

                # print(random_numbers)
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
#InitialNoOfColdspots = InitializeTheSulphurVacancies(NoOfSulphurAtoms, ExpectedNumberOfVacancies)
NoOfVacancies = ExpectedNumberOfVacancies
print("Initial No Of Coldspots: ", NoOfVacancies)
output_folder = "Formation-Annhilation_data"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_folder = os.path.join(output_folder, timestamp)
os.makedirs(output_folder, exist_ok=True)

# Initialize DataFrames
crset_df = pd.DataFrame()
annset_df = pd.DataFrame()
crreset_df = pd.DataFrame()
annreset_df = pd.DataFrame()

fwset_resistance_values = []
Bwset_resistance_values = []
fwreset_resistance_values = []
bwreset_resistance_values = []

Rth_series =[]

for i in range(num_outputs):
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

    NoOfColdspots = NoOfVacancies

    timeSet, NoOfColdspotsFS, NoOfHotspotsFS, AppliedBiasFS = KmcProcess_FWSet (RampRate,
                                                                               Thickness,
                                                                               NoOfColdspots,
                                                                               NoOfHotspots, NoOfVacancies,
                                                                               Eb_Cr, alpha_Cr_Set,
                                                                               AppliedBias,
                                                                               alpha_An1_Set,alpha_An2_Set, ResetExpo_Set,Eb_An)


    timeSet,NoOfColdspotsFS,NoOfHotspotsFS,AppliedBiasBS = KmcProcess_BwSet    (RampRate,
                                                                               MaxNoSteps,
                                                                               Thickness,
                                                                               NoOfColdspotsFS,
                                                                               NoOfHotspotsFS,
                                                                               AppliedBiasFS )

    timeReset,NoOfColdspotsFR,NoOfHotspotsFR,AppliedBiasFR = KmcProcess_FWReset (RampRate,
                                                                                Thickness, NoOfColdspotsFS,
                                                                                NoOfHotspotsFS,NoOfVacancies,
                                                                                Eb_Cr,alpha_Cr_Reset,
                                                                                AppliedBiasBS,
                                                                                alpha_An1_Reset,alpha_An2_Reset,
                                                                                ResetExpo_Reset,Eb_An)

    timeReset,NoOfColdspots,NoOfHotspots,AppliedBias = KmcProcess_BWReset (RampRate,
                                                                                 MaxNoSteps,
                                                                                 Thickness,
                                                                                 NoOfColdspotsFR,
                                                                                 NoOfHotspotsFR,
                                                                                 AppliedBiasFR)


    set_x_axis = Voltage_series_Set
    reset_x_axis = Voltage_series_Reset

    crset_data = {'X-axis': set_x_axis, 'Y-axis': formationrate_Set }
    crset_output_df = pd.DataFrame(crset_data)

    annset_data = {'X-axis': set_x_axis, 'Y-axis': annhilationrate_Set }
    annset_output_df = pd.DataFrame(annset_data)

    crreset_data = {'X-axis': reset_x_axis, 'Y-axis':formationrate_Reset }
    crreset_output_df = pd.DataFrame(crreset_data)

    annreset_data = {'X-axis': reset_x_axis , 'Y-axis': annhilationrate_Reset }
    annreset_output_df = pd.DataFrame(annreset_data)


    # Reduce the size of DataFrames by selecting every 700th row
    crset_output_df = crset_output_df.iloc[::700, :]
    annset_output_df = annset_output_df.iloc[::700, :]
    crreset_output_df = crreset_output_df.iloc[::700, :]
    annreset_output_df = annreset_output_df.iloc[::700, :]

    # Concatenate DataFrames
    crset_df = pd.concat([crset_df, crset_output_df], axis=1)
    annset_df = pd.concat([annset_df, annset_output_df], axis=1)
    crreset_df = pd.concat([crreset_df, crreset_output_df], axis=1)
    annreset_df = pd.concat([annreset_df, annreset_output_df], axis=1)


    plt.figure(1)
    plt.semilogy(Voltage_series_Set, formationrate_Set, '-', color='blue', label='formationrate_Set')
    plt.semilogy(Voltage_series_Set, annhilationrate_Set, '-', color='red', label='annhilationrate_Set')


    plt.semilogy(Voltage_series_Reset, formationrate_Reset, '-', color='blue',label='formationrate_Set')
    plt.semilogy(Voltage_series_Reset, annhilationrate_Reset, '-', color='red',label= 'annhilationrate_Reset')

    plt.xlabel('Voltage (V)')
    plt.ylabel('rate')

# Save full DataFrames
try:
    crset_df.to_csv(os.path.join(output_folder, f"crset_data_{timestamp}.csv"), index=False)
    annset_df.to_csv(os.path.join(output_folder, f"annset_data_{timestamp}.csv"), index=False)
    crreset_df.to_csv(os.path.join(output_folder, f"crreset_data_{timestamp}.csv"), index=False)
    annreset_df.to_csv(os.path.join(output_folder, f"annreset_data_{timestamp}.csv"), index=False)
except Exception as e:
    print("Error saving waveform data files:", e)
plt.legend()
plt.show()

