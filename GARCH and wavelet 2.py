#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 15:14:35 2025

@author: vipeenkumar
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
from statsmodels.tsa.ardl import ARDL
from sklearn.preprocessing import StandardScaler
import pywt

# Load a CSV file
exchange_rate_data = pd.read_csv('/Users/vipeenkumar/Downloads/my own research/exchange rate volatility and capitatl formation india/ exchange rate dataset/usdinr 1992 to 2023 .csv', parse_dates=['Month'], index_col='Month')

# Display the first few rows of the DataFrame
print(exchange_rate_data.head())
capital_formation_data = pd.read_csv('/Users/vipeenkumar/Downloads/my own research/exchange rate volatility and capitatl formation india/ exchange rate dataset/gcf 1992 to 2023.csv', parse_dates=['Year'], index_col='Year')
print(capital_formation_data.head())

# Convert monthly exchange rate to annual
exchange_rate_annual = exchange_rate_data['End month USD'].resample('YE').std() #use standard deviation as volatility
exchange_rate_annual.index = exchange_rate_annual.index.year
capital_formation_data.index = capital_formation_data.index.year

# Align Data (same as before)
combined_data = pd.concat([exchange_rate_annual, capital_formation_data], axis=1).dropna()
combined_data.columns = ['ExchangeRateVolatility', 'CapitalFormation']
print(combined_data.head)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_data)
scaled_df = pd.DataFrame(scaled_data, index=combined_data.index, columns=combined_data.columns)

# Wavelet Analysis (Example using 'db4' wavelet)
wavelet = 'db4'  # Daubechies 4 wavelet
coeffs_exchange = pywt.wavedec(scaled_df['ExchangeRateVolatility'], wavelet)
coeffs_capital = pywt.wavedec(scaled_df['CapitalFormation'], wavelet)

# Visualization of Wavelet Coefficients
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
for i, coeff in enumerate(coeffs_exchange):
    plt.plot(coeff, label=f'Exchange Rate Coeff {i}')
plt.title('Wavelet Coefficients - Exchange Rate Volatility')
plt.legend()

plt.subplot(2, 1, 2)
for i, coeff in enumerate(coeffs_capital):
    plt.plot(coeff, label=f'Capital Formation Coeff {i}')
plt.title('Wavelet Coefficients - Capital Formation')
plt.legend()

plt.tight_layout()
plt.show()

# Detailed Wavelet Analysis and Visualization of Approximation and Detail Coefficients.

level = 3 # choose the level of decomposition
#UserWarning: Level value of 3 is too high: all coefficients will experience boundary effects.
coeffs_exchange_detailed = pywt.wavedec(scaled_df['ExchangeRateVolatility'], wavelet, level=level)
coeffs_capital_detailed = pywt.wavedec(scaled_df['CapitalFormation'], wavelet, level=level)

approx_exchange = coeffs_exchange_detailed[0]
detail_exchange = coeffs_exchange_detailed[1:]

approx_capital = coeffs_capital_detailed[0]
detail_capital = coeffs_capital_detailed[1:]

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(approx_exchange, label='Approximation Coeffs')
plt.title('Exchange Rate Volatility - Approximation Coefficients')
plt.legend()

plt.subplot(2, 2, 2)
for i, detail in enumerate(detail_exchange):
  plt.plot(detail, label=f'Detail Coeff {i+1}')
plt.title('Exchange Rate Volatility - Detail Coefficients')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(approx_capital, label='Approximation Coeffs')
plt.title('Capital Formation - Approximation Coefficients')
plt.legend()

plt.subplot(2, 2, 4)
for i, detail in enumerate(detail_capital):
  plt.plot(detail, label=f'Detail Coeff {i+1}')
plt.title('Capital Formation - Detail Coefficients')
plt.legend()

plt.tight_layout()
plt.show()
