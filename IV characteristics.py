import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from datetime import datetime

Rs = 1e8
Rr = 1e5
nu0 = 1e5# 1
nu01 = 1e11
DeltaT = 1e-4 # sec
DeltaV = 5e-5  # V
RampRate = DeltaV / DeltaT
MaxNoSteps = 1000
EbSet = 3.2#ev
EbReset = 3.4#ev
alphaSet = 7e-10
ResetExpo =  4.2
alphaReset1 = 1e20
alphaReset2 = 2.7e-10
Thickness = 1E-9
ComplianceCurrent = 1e-5
NoOfSulphurAtoms = int(70E6)  # in 5umx5um there are 25E7 Sulphur atoms
print('total no of sulfur atoms',NoOfSulphurAtoms)
PercentageOfVacancies = 1.5 / 100
ExpectedNumberOfVacancies = int(PercentageOfVacancies * NoOfSulphurAtoms)
G0 = 5E-10# S
print("ExpectedNumberOfVacancies ", ExpectedNumberOfVacancies)
CurrentSeries_SET = []
TimeSeries_SET = []
NoOfHotspotsSeries_SET = []
NoOfColdspotsSeries_SET = []
AppliedBiasSeries_SET = []

CurrentSeries_RESET = []
TimeSeries_RESET = []
NoOfHotspotsSeries_RESET = []
NoOfColdspotsSeries_RESET = []
AppliedBiasSeries_RESET = []

R1_series_SET = []
R2_series_SET = []
Voltage_series_SET = []

R1_series_RESET = []
R2_series_RESET = []
Voltage_series_RESET = []

num_outputs = int(input( "Enter the number of required outputs : "))
NoOfHotspots = 10
AppliedBias = 0
# Append to rate series
R1_series = []
R2_series = []
Voltage_series = []


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

def CalculateTheHotspotFormationRate(ElectricField, EbSet, alphaSet, Rs, AppliedBias, Current ,kBT_q_hotspots):
   # kBT_q = calculate_kBT_qSet(AppliedBias, Current, Rs)
    expo = (-(EbSet - alphaSet * ElectricField) / kBT_q_hotspots)
    if expo > 0:
        Rate = nu0
    else:
        Rate = nu0 * np.exp(expo)
    return (Rate)


def CalculateTheHotspotAnnihilationRate(ElectricField, AppliedBias, Current, alphaReset1,
        ResetExpo, Rr, alphaReset2, EbReset, kBT_q_coldspots):

    expo = (-(EbReset - alphaReset1 * pow((AppliedBias * Current),
                                          ResetExpo) - alphaReset2 * ElectricField) / kBT_q_coldspots)
    if expo > 0:
        Rate = nu01
    else:
        Rate = nu01 * np.exp(expo)  # expo is already negative
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

def CurrentCalculation(AppliedBias, NoOfPoppedSulphurAtoms):
    I = G0 * AppliedBias * NoOfPoppedSulphurAtoms
    return (I)

def calculate_kBT_qSet(AppliedBias, Current, Rs):
    temperature = 300 + AppliedBias * Current * Rs
    return (8.625E-5 * temperature)

def calculate_kBT_qReset(AppliedBias, Current, Rr):
    temperature = 300 + AppliedBias * Current * Rr
    return (8.625E-5 * temperature)

# Sulphur moves into the Molybdenum plane
def KmcProcess(RampRate, Thickness, NoOfColdspots , NoOfHotspots,
                   EbSet, alphaSet, StartBias, Rs, Rr, alphaReset1,alphaReset2, ResetExpo, EbReset):

    BiasWasReset = False
    StepIndex = 0
    AppliedBias = StartBias
    AppliedBiasOld = StartBias
    Time = 0
    k = 0
    ElectricField = 0
    print("Initial No Of Coldspots before kMC  ", NoOfColdspots)
    print("Initial No Of Hotspots before kMC ", NoOfHotspots)

    while True:
        ...
        # Terminate only after RESET phase and reaching 4V
        if BiasWasReset and AppliedBias >= 4.0:
            print("Terminating due to excessive iterations")
            break

        StepIndex = StepIndex + 1
        CurrentTimeStep = k * DeltaT
        NextIncrementTime = (k + 1) * DeltaT
        AppliedBias = CurrentTimeStep * RampRate

        #if StepIndex > 1e8:
           # print("Terminating due to excessive iterations")
          #  break

        Current = CurrentCalculation(AppliedBias, NoOfHotspots)
        print (f"Current = {Current:.2e}")

        if not BiasWasReset and Current >= ComplianceCurrent:
            print("\033[91m⚡ Compliance current reached! Resetting Bias to 0V for RESET phase.\033[0m")
            AppliedBias = 0
            AppliedBiasOld = 0
            Time = 0
            ElectricField = 0
            k = 0
            BiasWasReset = True
            continue# Start RESET phase from zero bias

        if ((AppliedBias - AppliedBiasOld) > 1E-5):

           ElectricField = AppliedBias / Thickness

        kBT_q_hotspots = calculate_kBT_qSet(AppliedBias, CurrentCalculation(AppliedBias,NoOfHotspots), Rs)
        kBT_q_coldspots = calculate_kBT_qReset(AppliedBias, CurrentCalculation(AppliedBias, NoOfHotspots), Rr)

        expo_hotspots = (EbSet - alphaSet * ElectricField)
        expo_coldspots = ((EbReset - alphaReset1 * pow((AppliedBias * CurrentCalculation(AppliedBias, NoOfHotspots)), ResetExpo) - alphaReset2 * ElectricField) )

        J = alphaReset1 * pow((AppliedBias * CurrentCalculation(AppliedBias, NoOfHotspots)), ResetExpo)
        E = alphaSet * ElectricField

        Rate_hotspots = CalculateTheHotspotFormationRate(ElectricField, EbSet, alphaSet,Rs,AppliedBias,
                                                         CurrentCalculation(AppliedBias,NoOfHotspots),
                                                       kBT_q_hotspots )

        Rate_coldspots = CalculateTheHotspotAnnihilationRate(ElectricField, AppliedBias,
                                                CurrentCalculation(AppliedBias, NoOfHotspots), alphaReset1,
                                                ResetExpo, Rr, alphaReset2,EbReset , kBT_q_coldspots)

        R1 = Rate_hotspots * NoOfColdspots
        R2 = Rate_coldspots * NoOfHotspots
        Rate =  R1 + R2
        tau = CalculateTheTimeUpdate(Rate)
        # Append to rate series
        if not BiasWasReset:
            # Append to SET (Formation) phase
            R1_series_SET.append(R1)
            R2_series_SET.append(R2)
            Voltage_series_SET.append(AppliedBias)
        else:
            # Append to RESET (Annihilation) phase
            R1_series_RESET.append(R1)
            R2_series_RESET.append(R2)
            Voltage_series_RESET.append(-AppliedBias)


        print(f"\nStep {StepIndex} | Bias = {AppliedBias:.4f} V |ElectricField = {ElectricField:.2e} V/m | Hotspots = {NoOfHotspots} | Coldspots = {NoOfColdspots}")
        print(f"R1 (hotspot) = {R1:.2e} | R2 (coldspot) = {R2:.2e} | Rate = {Rate:.2e} | tau = {tau:.2e} | time = {Time:.2e}")
        print( f"kBT_q_hotspots = {kBT_q_hotspots:.2e} | kBT_q_coldspots = {kBT_q_coldspots:.2e} | expo_hotspots = {expo_hotspots:.2e} | expo_coldspots = {expo_coldspots:.2e} ")
        print( f"J = {J:.2e} | E = {E:.2e} | Rate_coldspots = {Rate_coldspots:.2e} | Rate_hotspots  = {Rate_hotspots:.2e} ")
        # Generate a random number for event type selection
        r = np.random.rand()
        print(f"Random number: {r}")
        print(f"R1/Rate (probability of coldspot → hotspot): {R1 / Rate:.4f}")

        if (Time + tau <= CurrentTimeStep + DeltaT):

            if r < (R1 / Rate):
                upper_bound = max(min(10000,NoOfColdspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)
                if NoOfColdspots > random_numbers:
                    NoOfColdspots -= random_numbers
                    NoOfHotspots += random_numbers
                    Time += tau
                    print(">> R_hot event: Hotspot formed")

                    if NoOfColdspots == 0:
                        print(
                            f"⚠️ Coldspots dropped to zero at Step {StepIndex}, Bias = {AppliedBias:.3f} V, Time = {Time:.2e} s")
                        break
                else:
                    print("!! Skipped R_hot event: Not enough coldspots available")
                    # Force transition to RESET phase

            else:
                upper_bound = max(min(10000, NoOfHotspots), 1)
                random_numbers = np.random.randint(1, upper_bound + 1)
                # print(random_numbers)
                if ( NoOfHotspots <= random_numbers):
                    print("!! Skipped R_cold event: No hotspots available")
                    break
                NoOfColdspots += random_numbers
                NoOfHotspots -= random_numbers
                Time += tau
                print(">> R_cold event: Hotspot removed")

        else:
            k += 1
            Time = NextIncrementTime
            print("Event skipped, advancing time step. k =", k)
            #print("Number of Coldspots   ", NoOfColdspots)
           # print("Number of Hotspots ", NoOfHotspots)
        AppliedBiasOld = AppliedBias

        if not BiasWasReset:
              CurrentSeries_SET.append(CurrentCalculation(AppliedBias, NoOfHotspots))
              TimeSeries_SET.append(Time)
              NoOfHotspotsSeries_SET.append(NoOfHotspots)
              NoOfColdspotsSeries_SET.append(NoOfColdspots)
              AppliedBiasSeries_SET.append(AppliedBias)
        else:
              CurrentSeries_RESET.append(CurrentCalculation(AppliedBias, NoOfHotspots))
              TimeSeries_RESET.append(-Time)
              NoOfHotspotsSeries_RESET.append(NoOfHotspots)
              NoOfColdspotsSeries_RESET.append(NoOfColdspots)
              AppliedBiasSeries_RESET.append(AppliedBias)

    print("Simulation completed: AppliedBias reached 4V after RESET.")
    print(f"Final Hotspots: {NoOfHotspots}")
    print(f"Final Coldspots: {NoOfColdspots}")
    print("kMC Process Summary: ")
    print("\tNo Of Steps = ", StepIndex)
    print("\ttime = ", Time)
    print("\tNumber of Coldspots   ", NoOfColdspots)
    print("\tNumber of Hotspots ", NoOfHotspots)
    print("electric field value", ElectricField)
    print("Applied bias value",AppliedBias)

    return Time, NoOfColdspots, NoOfHotspots, AppliedBias

i = 0
temp = []
i = i + 1
#InitialNoOfColdspots = InitializeTheSulphurVacancies(NoOfSulphurAtoms, ExpectedNumberOfVacancies)
InitialNoOfColdspots = ExpectedNumberOfVacancies
print("Initial No Of Coldspots: ", InitialNoOfColdspots)

base_output_dir = "IV_output_data"
# Create timestamp folder name like '2025-05-23_15-40-10'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Full path to timestamp folder
output_dir = os.path.join(base_output_dir, timestamp)
# Create the folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for i in range(num_outputs):
    print("ComplianceCurrent",ComplianceCurrent)
    print ("cycle number: ", i)
    result_reset = random_reset()
    CurrentSeries_SET = []
    TimeSeries_SET = []
    NoOfHotspotsSeries_SET = []
    NoOfColdspotsSeries_SET = []
    AppliedBiasSeries_SET = []

    CurrentSeries_RESET = []
    TimeSeries_RESET = []
    NoOfHotspotsSeries_RESET = []
    NoOfColdspotsSeries_RESET = []
    AppliedBiasSeries_RESET =[]

    R1_series_SET = []
    R2_series_SET = []
    Voltage_series_SET = []

    R1_series_RESET = []
    R2_series_RESET = []
    Voltage_series_RESET = []


    timeSet, NoOfColdspotsFS, NoOfHotspotsFS, AppliedBiasFS = KmcProcess(RampRate, Thickness, InitialNoOfColdspots, NoOfHotspots,
                                                      EbSet, alphaSet, AppliedBias, Rs, Rr, alphaReset1, alphaReset2, ResetExpo, EbReset)


    Formation_Voltage  = RampRate * np.array(TimeSeries_SET)
    Annihilation_Voltage = RampRate * np.array(TimeSeries_RESET)
    # Create a dict of DataFrames
    excel_data = {
        "Formation_Current": pd.DataFrame({

            "Current_SET": np.array(CurrentSeries_SET),
            "Voltage_SET": Formation_Voltage,
            "Hotspots_SET": np.array(NoOfHotspotsSeries_SET),
            "Coldspots_SET": np.array(NoOfColdspotsSeries_SET)
        }),

        "Annihilation_Current": pd.DataFrame({

            "Current_RESET": np.array(CurrentSeries_RESET),
            "Voltage_RESET": Annihilation_Voltage,
            "Hotspots_RESET": np.array(NoOfHotspotsSeries_RESET),
            "Coldspots_RESET": np.array(NoOfColdspotsSeries_RESET)
        }),

        "Rates_SET": pd.DataFrame({
            "Voltage-SET": np.array(Voltage_series_SET),
            "R1_SET": np.array(R1_series_SET),
            "R2_SET": np.array(R2_series_SET),
        }),

        "Rates_RESET": pd.DataFrame({
            "Voltage-RESET": np.array(Voltage_series_RESET),
            "R1_RESET": np.array(R1_series_RESET),
            "R2_RESET": np.array(R2_series_RESET),
        }),
    }

    for name, df in excel_data.items():
        if not df.empty:
            file_path = os.path.join(output_dir, f"{name}.xlsx")
            df.to_excel(file_path, index=False)
            print(f"Saved {name} to {file_path}")
        else:
            print(f"⚠️ Warning: DataFrame {name} is empty. Skipping saving.")

  #  plt.title('50 cycles ')
    plt.figure(1)
    plt.semilogy(RampRate * np.array(TimeSeries_SET), CurrentSeries_SET, '-')
    plt.semilogy(RampRate * np.array(TimeSeries_RESET), CurrentSeries_RESET, '-')
    plt.xlabel('Voltage (V)')
    plt.ylabel('  Current(A)')
    plt.figure(2)
    plt.semilogy(RampRate * np.array(TimeSeries_SET), NoOfHotspotsSeries_SET, '-' ,label='Hotspots_SET')
    plt.semilogy(RampRate * np.array(TimeSeries_SET), NoOfColdspotsSeries_SET, '-' ,label='Coldspots_SET')
    plt.semilogy(RampRate * np.array(TimeSeries_RESET), NoOfHotspotsSeries_RESET, '-', label='Hotspots_RESET')
    plt.semilogy(RampRate * np.array(TimeSeries_RESET), NoOfColdspotsSeries_RESET, '-', label='Coldspots_RESET')
    plt.xlabel('Voltage (V)')
    plt.ylabel('  Hotspots')
    plt.legend()
    plt.figure(3)
    plt.semilogy(Voltage_series_SET, R1_series_SET, label='R1: Hotspot Formation Rate_SET')
    plt.semilogy(Voltage_series_SET, R2_series_SET, label='R2: Hotspot Annihilation Rate_SET')
    plt.semilogy(Voltage_series_RESET, R1_series_RESET, label='R1: Hotspot Formation Rate_RESET')
    plt.semilogy(Voltage_series_RESET, R2_series_RESET, label='R2: Hotspot Annihilation Rate_RESET')

    plt.xlabel("Voltage (V)")
    plt.ylabel("Rate (1/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


plt.show()
