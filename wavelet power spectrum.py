#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:29:38 2025

@author: vipeenkumar
"""

import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet  # Install with: pip install pycwt

# Simulated Time Series Data
np.random.seed(42)
n = 256  # Number of data points
t = np.arange(n)
signal = np.sin(2 * np.pi * t / 32) + 0.5 * np.sin(2 * np.pi * t / 8) + np.random.normal(0, 0.3, n)

# Normalize the data
signal = (signal - np.mean(signal)) / np.std(signal)

# Define time step (assuming uniform sampling)
dt = 1  # Adjust according to your data (e.g., time interval between points)

# Define the Morlet wavelet
mother_wavelet = wavelet.Morlet(6)  # Morlet wavelet with ω₀ = 6

# Compute the Fourier factor to get scales from periods
fourier_factor = wavelet.Morlet().flambda  # Get Fourier factor for Morlet
min_period, max_period = 1, 64  # Define period range
scales = (min_period + np.arange(max_period - min_period)) / fourier_factor  # Compute scales

# Compute the Continuous Wavelet Transform (CWT)
coefficients, frequencies = wavelet.cwt(signal, dt, scales, mother_wavelet)
power = np.abs(coefficients) ** 2  # Compute power spectrum

# Plot Wavelet Power Spectrum
plt.figure(figsize=(10, 6))
plt.contourf(t, frequencies, power, levels=100, cmap="jet")  # Power spectrum visualization
plt.yscale("log")  # Log scale for better visualization
plt.colorbar(label="Power")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Wavelet Power Spectrum")
plt.show()
