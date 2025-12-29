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
Eb_An = 3# 3.4  ev

Rth_Set = 5e7 #5e7
nu0_Set = 1e5   #1e5
nu01_Set = 1e3  #1e3
alpha_Cr_Set = 1e-9  #1e-9
alpha_An2_Set = 1e-9   # 2e-9
alpha_An1_Set = 1e14   # 1e14
ResetExpo_Set = 4.2   #4.2


Rth_Reset = 5e6#5e6
nu0_Reset = 1e5  #1e5
nu01_Reset = 1e3 #1e3
alpha_Cr_Reset = 1e-15#1e-9
alpha_An2_Reset = 1.5e-9  # 2e-9
alpha_An1_Reset = 1e11  # 1e14
ResetExpo_Reset = 3.2 #4.2


Thickness = 1E-9
ComplianceCurrent = 2e-3
NoOfSulphurAtoms = int(70E6)  # in 5umx5um there are 25E7 Sulphur atoms
print('total no of sulfur atoms',NoOfSulphurAtoms)
PercentageOfVacancies = 1.5 / 100
ExpectedNumberOfVacancies = int(PercentageOfVacancies * NoOfSulphurAtoms)
G0 = 5E-10# S
print("ExpectedNumberOfVacancies ", ExpectedNumberOfVacancies)
num_outputs = int(input( "Enter the number of required outputs : "))

NoOfHotspots = 50
AppliedBias = 0

event_flag_list = []
event_flag_list_Reset = []



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

def CalculateTheHotspotFormationRate_Set(ElectricField, Eb_Cr, alpha_Cr_Set, Rth_Set, AppliedBias, Current ,kBT_q_Set):
   # kBT_q = calculate_kBT_qSet(AppliedBias, Current, Rs)
    expo = (-(Eb_Cr - alpha_Cr_Set * ElectricField) / kBT_q_Set)
    if expo > 0:
        Rate = nu0_Set
    else:
        Rate = nu0_Set * np.exp(expo)
    return (Rate)

def CalculateTheHotspotAnnihilationRate_Set(ElectricField, AppliedBias, Current, alpha_An1_Set,
        ResetExpo_Set, Rth_Set, alpha_An2_Set, Eb_An, kBT_q_Set):

    expo = (-(Eb_An - alpha_An1_Set * pow((AppliedBias * Current),
                                          ResetExpo_Set) - alpha_An2_Set * ElectricField) / kBT_q_Set)
    if expo > 0:
        Rate = nu01_Set
    else:
        Rate = nu01_Set * np.exp(expo)  # expo is already negative
    return Rate

def CalculateTheHotspotFormationRate_Reset(ElectricField, Eb_Cr, alpha_Cr_Reset, Rth_Reset, AppliedBias, Current ,kBT_q_Reset):
   # kBT_q = calculate_kBT_qSet(AppliedBias, Current, Rs)
    expo = (-(Eb_Cr - alpha_Cr_Reset * ElectricField) / kBT_q_Reset)
    if expo > 0:
        Rate = nu0_Reset
    else:
        Rate = nu0_Reset * np.exp(expo)
    return (Rate)

def CalculateTheHotspotAnnihilationRate_Reset(ElectricField, AppliedBias, Current, alpha_An1_Reset,
        ResetExpo_Reset, Rth_Reset, alpha_An2_Reset, Eb_An, kBT_q_Reset):

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

def calculate_kBT_qSet(AppliedBias, Current, Rth_Set):
    temperature = 300 + AppliedBias * Current * Rth_Set
    return (8.625E-5 * temperature)

def calculate_kBT_qReset(AppliedBias, Current, Rth_Reset):
    temperature = 300 + AppliedBias * Current * Rth_Reset
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
def KmcProcess_FWSet(RampRate, Thickness, NoOfColdspots , NoOfHotspots,
                   Eb_Cr, alpha_Cr_Set, StartBias, Rth_Set, alpha_An1_Set,alpha_An2_Set, ResetExpo_Set, Eb_An):

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

        #if StepIndex > 1e7:
           # print("Terminating due to excessive iterations")
           # break

        if ((AppliedBias - AppliedBiasOld) > 1E-5):

            ElectricField = AppliedBias / Thickness


        kBT_q_Set = calculate_kBT_qSet(AppliedBias, CurrentCalculation(AppliedBias,NoOfHotspots), Rth_Set)

        Rate_hotspots_Set = CalculateTheHotspotFormationRate_Set(ElectricField, Eb_Cr, alpha_Cr_Set,Rth_Set,AppliedBias,
                                                         CurrentCalculation(AppliedBias,NoOfHotspots),
                                                       kBT_q_Set )

        Rate_coldspots_Set = CalculateTheHotspotAnnihilationRate_Set(ElectricField, AppliedBias,
                                                CurrentCalculation(AppliedBias, NoOfHotspots), alpha_An1_Set,
                                                ResetExpo_Set, Rth_Set, alpha_An2_Set, Eb_An , kBT_q_Set)


        R1_Set = Rate_hotspots_Set * NoOfColdspots
        R2_Set = Rate_coldspots_Set * NoOfHotspots
        Rate_Set = R1_Set + R2_Set


        tau = CalculateTheTimeUpdate(Rate_Set)

        Current = CurrentCalculation(AppliedBias,NoOfHotspots)

        print(
            f"\nStep {StepIndex} | Bias = {AppliedBias:.4f} V |Current = {Current:.2e} Amp | Hotspots = {NoOfHotspots} | Coldspots = {NoOfColdspots}")

        print(
            f"\ntau {tau} | time = {time:.4f} V |NextIncrementTime = {NextIncrementTime:.2e}")

        formationrate_Set.append(R1_Set)
        annhilationrate_Set.append(R2_Set)
        Voltage_series_Set.append(AppliedBias)
        tau_set_series.append(tau)


        r = np.random.rand()

        if (time + tau <= CurrentTimeStep + DeltaT):

            event_flag = 1  # Event occurred
            print(f"Event occurred at Bias = {AppliedBias:.3f} V, Time = {time:.2e} ")

            if r < (R1_Set / Rate_Set):

                upper_bound = max(min(1e4, NoOfColdspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)

                if NoOfColdspots > random_numbers:

                   NoOfColdspots -= random_numbers
                   NoOfHotspots += random_numbers
                   print(">> R_hot event: Hotspot formed")

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

                # print(random_numbers)
                if ( NoOfHotspots > random_numbers ):

                   NoOfColdspots += random_numbers
                   NoOfHotspots -= random_numbers
                   print(">> R_cold event: Hotspot removed")
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

def KmcProcess_FWReset(RampRate, Thickness, NoOfColdspots , NoOfHotspots,
                   Eb_Cr, alpha_Cr_Reset, StartBias, Rth_Reset, alpha_An1_Reset,alpha_An2_Reset, ResetExpo_Reset, Eb_An):

    StepIndex = 0
    AppliedBias = StartBias
    AppliedBiasOld = StartBias
    ElectricField = 0
    time = 0
    k = 0
    step = 0

    while (AppliedBias <= 1.9):

        StepIndex = StepIndex + 1
        CurrentTimeStep = k * DeltaT
        NextIncrementTime = (k + 1) * DeltaT
        AppliedBias = CurrentTimeStep * RampRate

       # if StepIndex > 1e0:
           # print("Terminating due to excessive iterations")
            #break


        if ((AppliedBias - AppliedBiasOld) > 1E-5):
            ElectricField = AppliedBias / Thickness

        kBT_q_Reset = calculate_kBT_qReset(AppliedBias, CurrentCalculation(AppliedBias, NoOfHotspots), Rth_Reset)

        Rate_hotspots_Reset = CalculateTheHotspotFormationRate_Reset(ElectricField, Eb_Cr, alpha_Cr_Reset, Rth_Reset, AppliedBias,
                                                                   CurrentCalculation(AppliedBias, NoOfHotspots),
                                                                   kBT_q_Reset)

        Rate_coldspots_Reset = CalculateTheHotspotAnnihilationRate_Reset(ElectricField, AppliedBias,
                                                                         CurrentCalculation(AppliedBias, NoOfHotspots),
                                                                         alpha_An1_Reset,
                                                                         ResetExpo_Reset, Rth_Reset, alpha_An2_Reset, Eb_An,
                                                                         kBT_q_Reset)


        R1_Reset = Rate_hotspots_Reset * NoOfColdspots
        R2_Reset = Rate_coldspots_Reset * NoOfHotspots
        Rate_Reset = R1_Reset + R2_Reset
        tau = CalculateTheTimeUpdate(Rate_Reset)

        formationrate_Reset.append(R1_Reset)
        annhilationrate_Reset.append(R2_Reset)
        Voltage_series_Reset.append(-AppliedBias)

        r = np.random.rand()


        if (time + tau <= CurrentTimeStep + DeltaT):

            event_flag_Reset =1


            if r < (R1_Reset / Rate_Reset):
                upper_bound = max(min(1e4, NoOfColdspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)

                if NoOfColdspots > random_numbers:

                   NoOfColdspots -= random_numbers
                   NoOfHotspots += random_numbers
                   print(">> R_hot event: Hotspot formed")

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
                     print(">> R_cold event: Hotspot removed")

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

        if ((AppliedBias - AppliedBiasOld) > 1E-5):

            ElectricField = AppliedBias / Thickness

        kBT_q_Reset = calculate_kBT_qReset(AppliedBias, CurrentCalculation(AppliedBias, NoOfHotspots), Rth_Reset)

        Rate_hotspots_Reset = CalculateTheHotspotFormationRate_Reset(ElectricField, Eb_Cr, alpha_Cr_Reset, Rth_Reset, AppliedBias,
                                                                   CurrentCalculation(AppliedBias, NoOfHotspots),
                                                                   kBT_q_Reset)

        Rate_coldspots_Reset = CalculateTheHotspotAnnihilationRate_Reset(ElectricField, AppliedBias,
                                                                         CurrentCalculation(AppliedBias, NoOfHotspots),
                                                                         alpha_An1_Reset,
                                                                         ResetExpo_Reset, Rth_Reset, alpha_An2_Reset, Eb_An,
                                                                         kBT_q_Reset)


        R1_Reset = Rate_hotspots_Reset * NoOfColdspots
        R2_Reset = Rate_coldspots_Reset * NoOfHotspots
        Rate_Reset = R1_Reset + R2_Reset
        tau = CalculateTheTimeUpdate(Rate_Reset)

        r = np.random.rand()


        if (time + tau <= CurrentTimeStep + DeltaT):


            if r < (R1_Reset / Rate_Reset):
                upper_bound = max(min(1e4, NoOfColdspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)

                if NoOfColdspots > random_numbers:

                   NoOfColdspots -= random_numbers
                   NoOfHotspots += random_numbers
                   print(">> R_hot event: Hotspot formed")

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
                     print(">> R_cold event: Hotspot removed")

                     if NoOfHotspots == 0:
                         print(
                             f"⚠️ Coldspots dropped to zero at Step {StepIndex}, Bias = {AppliedBias:.3f} V, Time = {time:.2e} s")
                         break

                else:
                     print("!! Skipped R_cold event: No hotspots available")
                     break

            time += tau
        else:
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
InitialNoOfColdspots = ExpectedNumberOfVacancies
print("Initial No Of Coldspots: ", InitialNoOfColdspots)

output_folder = "Formation-Annhilation_data"
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


    timeSet, NoOfColdspotsFS, NoOfHotspotsFS, AppliedBiasFS = KmcProcess_FWSet (RampRate,
                                                                               Thickness,
                                                                               InitialNoOfColdspots,
                                                                               NoOfHotspots,
                                                                               Eb_Cr, alpha_Cr_Set,
                                                                               AppliedBias,Rth_Set,
                                                                               alpha_An1_Set,alpha_An2_Set, ResetExpo_Set,Eb_An)


    timeSet,NoOfColdspotsFS,NoOfHotspotsFS,AppliedBiasBS = KmcProcess_BwSet    (RampRate,
                                                                               MaxNoSteps,
                                                                               Thickness,
                                                                               NoOfColdspotsFS,
                                                                               NoOfHotspotsFS,
                                                                               AppliedBiasFS )

    timeReset,NoOfColdspotsFR,NoOfHotspotsFR,AppliedBiasFR = KmcProcess_FWReset (RampRate,
                                                                                Thickness, NoOfColdspotsFS,
                                                                                NoOfHotspotsFS,
                                                                                Eb_Cr,alpha_Cr_Reset,
                                                                                AppliedBiasBS,Rth_Reset,
                                                                                alpha_An1_Reset,alpha_An2_Reset,
                                                                                ResetExpo_Reset,Eb_An)

    timeReset,InitialNoOfColdspots,NoOfHotspots,AppliedBias = KmcProcess_BWReset (RampRate,
                                                                                 MaxNoSteps,
                                                                                 Thickness,
                                                                                 NoOfColdspotsFR,
                                                                                 NoOfHotspotsFR,
                                                                                 AppliedBiasFR)


    fwset_x_axis = RampRate * np.array(TimeSeries_FWSET )
    Bwset_x_axis = AppliedBiasFS - RampRate * np.array(TimeSeries_BWSET)
    fwreset_x_axis = RampRate * np.array(TimeSeries_FWRESET)
    bwreset_x_axis = -AppliedBiasFR - RampRate * np.array(TimeSeries_BWRESET)

    fwset_data = {'X-axis': fwset_x_axis, 'Y-axis': CurrentSeries_FWSET }
    fwset_output_df = pd.DataFrame(fwset_data)

    Bwset_data = {'X-axis': Bwset_x_axis, 'Y-axis': CurrentSeries_BWSET }
    Bwset_output_df = pd.DataFrame(Bwset_data)

    fwreset_data = {'X-axis': fwreset_x_axis, 'Y-axis':CurrentSeries_FWRESET }
    fwreset_output_df = pd.DataFrame(fwreset_data)

    bwreset_data = {'X-axis': bwreset_x_axis , 'Y-axis': CurrentSeries_BWRESET }
    bwreset_output_df = pd.DataFrame(bwreset_data)

    # Calculate resistances
    R_fwset = calculate_Rset(fwset_output_df)
    R_Bwset = calculate_Rset(Bwset_output_df)
    R_fwreset = calculate_Rrst(fwreset_output_df)
    R_bwreset = calculate_Rrst(bwreset_output_df)

    # Print resistance values
    # print("R for fwset_output_df:", R_fwset)
    # print("R for Bwset_output_df:", R_Bwset)
    # print("R for forward_output_df:", R_forward)
    # print("R for backward_output_df:", R_backward)

    # Append resistance values to lists
    fwset_resistance_values.append(R_fwset)
    Bwset_resistance_values.append(R_Bwset)
    fwreset_resistance_values.append(R_fwreset)
    bwreset_resistance_values.append(R_bwreset)

    # Reduce the size of DataFrames by selecting every 700th row
    fwset_output_df = fwset_output_df.iloc[::2, :]
    Bwset_output_df = Bwset_output_df.iloc[::2, :]
    fwreset_output_df = fwreset_output_df.iloc[::2, :]
    bwreset_output_df = bwreset_output_df.iloc[::2, :]

    # Concatenate DataFrames
    fwset_df = pd.concat([fwset_df, fwset_output_df], axis=1)
    Bwset_df = pd.concat([Bwset_df, Bwset_output_df], axis=1)
    fwreset_df = pd.concat([fwreset_df, fwreset_output_df], axis=1)
    bwreset_df = pd.concat([bwreset_df, bwreset_output_df], axis=1)


    plt.figure(1)
    plt.semilogy(RampRate * np.array(TimeSeries_FWSET), CurrentSeries_FWSET, '-', color='blue', label='New sim data')
    plt.semilogy(AppliedBiasFS - RampRate * np.array(TimeSeries_BWSET), CurrentSeries_BWSET, '--', color='blue')
    plt.semilogy(RampRate * np.array(TimeSeries_FWRESET), CurrentSeries_FWRESET, '-', color='blue')
    plt.semilogy(-AppliedBiasFR - RampRate * np.array(TimeSeries_BWRESET), CurrentSeries_BWRESET, '--', color='blue')

    plt.xlabel('Voltage (V)')
    plt.ylabel('  Current(A)')
   # plt.legend()
    # --- Read experimental data from ODS file ---
    exp_file = "/home/lavanya/Desktop/sim calibratn data.ods"
    exp_df = pd.read_excel(exp_file, engine='odf')  # Requires odfpy

    # Assuming the file has 4 columns: x1, y1, x2, y2
    x1 = exp_df.iloc[:, 0]
    y1 = exp_df.iloc[:, 1]
    x2 = exp_df.iloc[:, 2]
    y2 = exp_df.iloc[:, 3]
   # x3 = exp_df.iloc[:, 4]
   # y3 = exp_df.iloc[:, 5]
  #  x4 = exp_df.iloc[:, 6]
   # y4 = exp_df.iloc[:, 7]

    # --- Overlay experimental data in one color (e.g., black) ---
    plt.semilogy(x1, y1, '-', color='red', label='Old sim data 1e-5')
    plt.semilogy(x2, y2, '-', color='red')
    #plt.semilogy(x3, y3, '-', color='red', label='Old sim data 2e-3')
   # plt.semilogy(x4, y4, '-', color='red')


   # plt.figure(2)

    #plt.semilogy(RampRate * np.array(TimeSeries_FWSET), NoOfHotspotsSeries_FWSET, 'o', color='blue', label='Hotsopts')
    #plt.semilogy(RampRate * np.array(TimeSeries_FWSET), NoOfColdspotsSeries_FWSET, 'o', color='red', label='coldspots')
    #plt.xlabel('Voltage (V)')
    #plt.ylabel(' NoOfHotspots/NoOfColdspots  ')
    #plt.legend()

   # plt.figure(3)
    #plt.semilogy( Voltage_series_Set,  formationrate_Set, '-', color='blue', label='FormationRate_Set')
   # plt.semilogy(Voltage_series_Set, annhilationrate_Set, '--', color='red',label='AnnhilationRate_Set')
   # plt.semilogy(Voltage_series_Reset,formationrate_Reset, '-', color='blue',label='FormationRate_Reset')
   # plt.semilogy(Voltage_series_Reset,annhilationrate_Reset , '--', color='red',label='AnnhilationRate_Reset')

    #plt.xlabel('Voltage (V)')
    #plt.ylabel('  Rate ')
    #plt.legend()

   # plt.figure(4)
    #plt.plot(RampRate * np.array(TimeSeries_FWSET), event_flag_list, '-', color='blue', label=' event-flag_Set ')
   # plt.plot(RampRate * np.array(TimeSeries_FWRESET), event_flag_list_Reset, '-', color='blue', label=' event-flag_Reset ')
    #plt.xlabel('voltage (V)')
    #plt.ylabel('  event flog list ')
    #plt.legend()

    #plt.figure(5)
    #plt.semilogy(Voltage_series_Set, tau_set_series, '-', color='blue', label=' tau_set ')
    #plt.xlabel('voltage (V)')
   # plt.ylabel('  tau_set ')
    #plt.legend()

fwset_file_path = os.path.join(output_folder, f"fwset_data_{timestamp}.csv")
fwset_df.to_csv(fwset_file_path, index=False)

Bwset_file_path = os.path.join(output_folder, f"Bwset_data_{timestamp}.csv")
Bwset_df.to_csv(Bwset_file_path, index=False)

fwreset_file_path = os.path.join(output_folder, f"fwreset_data_{timestamp}.csv")
fwreset_df.to_csv(fwreset_file_path, index=False)

bwreset_file_path = os.path.join(output_folder, f"bwreset_data_{timestamp}.csv")
bwreset_df.to_csv(bwreset_file_path, index=False)

    # Create DataFrames for resistance values and save them to CSV files
fwset_resistance_df = pd.DataFrame({'Resistance': fwset_resistance_values})
Bwset_resistance_df = pd.DataFrame({'Resistance': Bwset_resistance_values})
fwreset_resistance_df = pd.DataFrame({'Resistance': fwreset_resistance_values})
bwreset_resistance_df = pd.DataFrame({'Resistance': bwreset_resistance_values})

fwset_resistance_df.to_csv(os.path.join(output_folder, 'fwset_resistance.csv'), index=False)
Bwset_resistance_df.to_csv(os.path.join(output_folder, 'Bwset_resistance.csv'), index=False)
fwreset_resistance_df.to_csv(os.path.join(output_folder, 'fwreset_resistance.csv'), index=False)
bwreset_resistance_df.to_csv(os.path.join(output_folder, 'bwreset_resistance.csv'), index=False)

print(f"{num_outputs} files have been generated and saved in the folder: {output_folder}")
print(f"Output folder location: {os.path.abspath(output_folder)}")

#plt.legend()
plt.show()
end_time = tm.time()
print(f"✅ Simulation completed in {end_time - start_time:.2f} seconds")


