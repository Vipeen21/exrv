#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:30:37 2025

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
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad, acorr_ljungbox
import pywt
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch

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


#descriptive statistics
descriptive_stats = combined_data.describe()
print(descriptive_stats)



# ADF Test (Stationarity)
def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller test
    """
    print(f'Results of ADF Test: {title}')
    result = sm.tsa.stattools.adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_string())
    
adf_test(combined_data['ExchangeRateVolatility'], 'Exchange Rate Volatility')
adf_test(combined_data['CapitalFormation'], 'Capital Formation')

print("Checking NANs Values before scaling") 
print(combined_data.isna().sum())  # Before scaling
print("Checking Infinite Values before scaling")
print(np.isinf(combined_data).sum())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_data)
scaled_df = pd.DataFrame(scaled_data, index=combined_data.index, columns=combined_data.columns)
print("After scaling")
print(scaled_df)

# Check for missing or infinite values
print("Checking NANs and Infinite Values after scaling")
print(scaled_df.info())  # Check for NaNs and data types
print(scaled_df.describe())  # Summary statistics
print("checking for NANs")
print(scaled_df.isna().sum())  # Count NaNs
print("checking for Infinite Values")
print(np.isinf(scaled_df).sum())  # Count Infs

# Replace inf values with NaN
scaled_df.replace([np.inf, -np.inf], np.nan, inplace=True)
print("infinite values replaced with Nan")
# Drop rows with NaN values
scaled_df.dropna(inplace=True)
print("after replacting infinite values with NAN, NAN values dropped ")

# Print after cleaning
print("after cleaning")
print(scaled_df.isna().sum())  # Check again


# Ensure index alignment
print("checking final dataset")
print(scaled_df.head())  # Check final dataset

# VAR Model (for impulse response and variance decomposition, if needed)

# Fit VAR model to determine optimal lag length
model_var = VAR(scaled_df)
lag_selection = model_var.select_order(maxlags=5)  # Test up to 10 lags but 10 lags is too large for this observation
print(lag_selection.summary())

selected_lag = lag_selection.aic  # Or use lag_selection.bic
var_model = model_var.fit(selected_lag)

# Display model summary
print(var_model.summary())


irf = var_model.irf(10) #Impulse Response Function
irf.plot(orth=False) #plot irf
fevd = var_model.fevd(10) #Forecast Error Variance Decomposition
fevd.plot() #plot fevd

# Granger Causality Test
granger_test = grangercausalitytests(scaled_df[['CapitalFormation', 'ExchangeRateVolatility']], maxlag=5, verbose=True) #maxlag may need adjustment



# Perform ARCH test
# If p-value < 0.05, reject H₀: No ARCH effect (no volatility clustering). → Volatility clustering exists.
arch_test = het_arch(scaled_df['ExchangeRateVolatility'])
print(f"ARCH Test p-value: {arch_test[1]}")

# Interpretation
if arch_test[1] < 0.05:
    print("Significant ARCH effect detected (Volatility clustering present).")
else:
    print("No significant ARCH effect detected.")


# GARCH Model (Volatility Modeling)
garch_model = arch_model(scaled_df['ExchangeRateVolatility'], vol='Garch', p=1, q=1) #p and q values may need adjustments
results_garch = garch_model.fit(disp='off')
print(results_garch.summary())





# Visualization




plt.figure(figsize=(12, 6))
plt.plot(combined_data['ExchangeRateVolatility'], label='Exchange Rate Volatility')
plt.plot(combined_data['CapitalFormation'], label='Capital Formation')
plt.title('Exchange Rate Volatility and Capital Formation')
plt.xlabel('Year')
plt.legend()
plt.show()

#checking volatility clustering visually
#If volatility clustering exists, you should see periods of high volatility followed by high volatility and low volatility followed by low volatility.
# Assuming 'Volatility' column contains the exchange rate volatility data
plt.figure(figsize=(12,6))
plt.plot(scaled_df['ExchangeRateVolatility'], label="Exchange Rate Volatility")
plt.title("Exchange Rate Volatility Over Time")
plt.xlabel("Year")
plt.ylabel("Volatility")
plt.legend()
plt.show()

# If the ACF of squared volatility is significantly positive for several lags, volatility clustering exists.
# ACF
# Squared Volatility
scaled_df['Squared_Volatility'] = scaled_df['ExchangeRateVolatility'] ** 2  

plt.figure(figsize=(12,6))
plot_acf(scaled_df['Squared_Volatility'], lags=20)
plt.title("ACF of Squared Exchange Rate Volatility")
plt.show()



#GARCH conditional volatility
plt.figure(figsize=(12, 6))
plt.plot(results_garch.conditional_volatility, label='GARCH Volatility')
plt.title('GARCH Conditional Volatility')
plt.xlabel('Year')
plt.legend()
plt.show()



# Residualfor  GARCH models.

residuals_garch = results_garch.resid


plt.figure(figsize=(12,6))
plt.plot(residuals_garch, label='GARCH Residuals')
plt.title('GARCH Residuals')
plt.xlabel('Year')
plt.legend()
plt.show()

