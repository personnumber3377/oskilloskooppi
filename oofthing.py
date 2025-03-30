import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import *



def load_data():
    fh = open("data.txt", "r")
    lines = fh.readlines()
    lines = lines[3:] # Skip header
    fh.close()

    times = []
    voltages = []

    for line in lines:
        assert ";" in line
        t, U = line.split(";")
        t = float(t.replace(",", "."))
        U = float(U.replace(",", "."))
        U /= 1000 # in millivolts so convert to volts
        t /= 1000 # in milliseconds so convert to seconds
        times.append(t)
        voltages.append(U)

    
    return times, voltages

def fit_function(U_initial, U_final, t, tau): # tau is the variable we need to basically solve by fitting...
    # V(t) = V_final - (V_final - V_initial)*e**(-t/tau)
    return U_final - (U_final - U_initial)*e**(-t/tau)

def cut_data(times, voltages):
    # start = 775+5 # 600
    start = 775+6 # 600
    end = 795
    times = times[start:end]
    voltages = voltages[start:end]
    return times, voltages

def sol():
    # Example: Replace these with actual measurement data

    # kaistanleveys
    # rise_time = x / kaistanleveys                #  * t

    times, voltages = load_data()
    
    times, voltages = cut_data(times, voltages)

    # Make such that the first time is zero...
    t_i = times[0]
    times = [t - t_i for t in times]
    assert times[0] == 0.0

    U_initial = voltages[0]
    U_final = voltages[-1]

    print("Voltages: "+str(voltages))
    print("Times: "+str(times))
    # PLOT HERE PLEASE


    # tau_guess = (times[-1] - times[0]) / 5  # Rough estimate for tau

    tau_guess = 0.01

    # Fit curve
    popt, _ = curve_fit(lambda t, tau: fit_function(U_initial, U_final, t, tau), times, voltages, p0=[tau_guess])
    tau_fit = popt[0]

    print(f"Estimated tau: {tau_fit:.6e} seconds")

    # Generate fitted curve
    # fitted_voltages = fit_function(times, tau_fit, U_initial, U_final)

    # tau_fit = 0.001
    # 0.000002 is pretty close

    tau_fit = 0.0000015 # 0.000002 # 0.00001
    # print("Initial voltage: "+str(voltages[0]))

    # Generate fitted curve using linspace
    t_fine = np.linspace(min(times), max(times), 1000)
    fitted_voltages_fine = fit_function(U_initial, U_final, t_fine, tau_fit)
    
    # Plot data and fit
    plt.figure(figsize=(8, 5))
    plt.scatter(times, voltages, label="Measured Data", color="blue", marker="o")
    plt.plot(t_fine, fitted_voltages_fine, label=f"Fitted Curve (τ = {tau_fit:.2e} s)", color="red")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (volts)")
    plt.title("Exponential Fit of Rising Edge")
    plt.legend()
    plt.grid()

    plt.savefig("plot.eps", format="eps", dpi=1000)
    plt.show()

    return

if __name__=="__main__":
    sol()
    exit()