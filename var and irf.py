#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:34:50 2025

@author: vipeenkumar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Load a CSV file
exchange_rate_data = pd.read_csv('/Users/vipeenkumar/Downloads/my own research/exchange rate volatility and capitatl formation india/ exchange rate dataset/usdinr 1992 to 2023 .csv', parse_dates=['Month'], index_col='Month')
print(exchange_rate_data.head()) # Display the first few rows of the DataFrame
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

# Load your time series data (ensure it's in a pandas DataFrame)
df = combined_data[['ExchangeRateVolatility', 'CapitalFormation']]

# Ensure it's in the correct date-time format
df.index = pd.to_datetime(df.index)

# Function to perform ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is NOT stationary. Consider differencing.")

# Run ADF test on both series
print("ADF Test for Exchange Rate Volatility:")
adf_test(df['ExchangeRateVolatility'])
print("\nADF Test for Capital Formation:")
adf_test(df['CapitalFormation'])

#if the series is non stationary, apply first differencing
#df_diff = df.diff().dropna()

# Fit VAR model to determine optimal lag length
model = VAR(df)
lag_selection = model.select_order(maxlags=5)  # Test up to 10 lags but 10 lags is too large for this observation
print(lag_selection.summary())

selected_lag = lag_selection.aic  # Or use lag_selection.bic
var_model = model.fit(selected_lag)

# Display model summary
print(var_model.summary())

# Plot impulse response functions for 10 periods ahead
irf = var_model.irf(10)  
irf.plot(orth=True)  # orth=True may be due to data is not standardize
plt.show()



