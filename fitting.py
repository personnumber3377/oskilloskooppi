import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)

# Magnetic field function from Equation (7)
def B_z(z, M, L, R):
    """Magnetic field B(z) along the axis of a cylindrical magnet."""
    term1 = z / np.sqrt(z**2 + R**2)
    term2 = (z - L) / np.sqrt((z - L)**2 + R**2)
    return (mu0 * M / 2) * (term1 - term2)

'''
Here is the raw measured data:
0,0
1,4.703
2,4.550
3,2.990 
4,2.720
5,2.590
6,2.550
7,2.530
8,2.520
9,2.516
10,2.512
11,2.509
12,2.507
13,2.505
14,2.504
15,2.503
16,2.503
'''


def load_data():
    fh = open("data.txt", "r")
    lines = fh.readlines()
    fh.close()
    S = 50 # V/T
    baseline_voltage = 2.5 # Volts
    E_values = []
    distances = []
    for line in lines:
        # 7mm per square
        assert "," in line
        d, v = line.split(",")
        d = float(d)
        v = float(v)
        d = d * 0.007 # 7mm in meters
        E_values.append(abs(v - baseline_voltage) / S) # dV / S = B
        distances.append(d)
    print(distances)
    skip_amount = 3
    # Skip bad data :D
    E_values = E_values[skip_amount:]
    distances = distances[skip_amount:]
    return E_values, distances

def sol():
    # Example: Replace these with actual measurement data
    # z_meas = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040])  # Measured z values in meters
    # B_meas = np.array([0.12, 0.11, 0.095, 0.080, 0.068, 0.055, 0.043, 0.030])  # Measured B values in Tesla
    B_meas, z_meas = load_data()
    print("z_meas: "+str(z_meas))
    # Known dimensions (Replace with actual values)
    # L=5mm R=4,5mm
    L = 0.005  # Magnet length in meters
    R = 0.0045  # Magnet radius in meters

    # Fit data to extract best-fit M
    popt, pcov = curve_fit(lambda z, M: B_z(z, M, L, R), z_meas, B_meas, p0=[1.0])  # Initial guess for M

    M_fit = popt[0]  # Extract fitted M
    M_fit_T = mu0 * M_fit  # Convert to Tesla

    print(f"Fitted magnetization M: {M_fit:.4f} A/m")
    print(f"Equivalent B = μ₀M: {M_fit_T:.4f} T")

    # Generate smooth curve for plotting
    z_fit = np.linspace(min(z_meas), max(z_meas), 1000)
    B_fit = B_z(z_fit, M_fit, L, R)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(z_meas, B_meas, color='red', label="Measured Data")  # Measurement points
    plt.plot(z_fit, B_fit, label=f"Fit: M = {M_fit:.2f} A/m", color='blue')  # Fit curve
    plt.xlabel("Distance z (m)")
    plt.ylabel("Magnetic Field B (T)")
    plt.title("Magnetic Field vs. Distance")
    plt.legend()
    plt.grid()
    plt.savefig("plot.eps", format="eps", dpi=1000)
    plt.show()

    return

if __name__=="__main__":
    sol()
    exit()