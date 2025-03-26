#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 00:49:52 2025

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
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# ... (Load data and perform initial analysis as before) ...

# ----------------------------------------------------------------------
# Robustness Checks:
# ----------------------------------------------------------------------

# 1. Alternative Volatility Measures (Example: Realized Volatility)
# (Requires daily exchange rate data)

daily_exchange_rate = pd.read_csv('/Users/vipeenkumar/Downloads/my own research/exchange rate volatility and capitatl formation india/ exchange rate dataset/USDINR Daily.csv', parse_dates=['Date'], index_col='Date')
# Check the data types
print(daily_exchange_rate.dtypes)
# Convert 'USDINR' to numeric, forcing errors to NaN
daily_exchange_rate['USDINR'] = pd.to_numeric(daily_exchange_rate['USDINR'], errors='coerce')

# Check for any NaN values
print(daily_exchange_rate['USDINR'].isna().sum())

# Option 1: Drop NaN values
daily_exchange_rate = daily_exchange_rate.dropna(subset=['USDINR'])

# Option 2: Fill NaN values (e.g., forward fill)
# daily_exchange_rate['USDINR'].fillna(method='ffill', inplace=True)

# Calculate realized volatility
realized_volatility = daily_exchange_rate['USDINR'].rolling(window=22).std().resample('YE').mean()  # Annual mean
print(realized_volatility)


try:
    daily_exchange_rate = pd.read_csv("USDINR_daily.csv", parse_dates=['Date'], index_col='Date')
    realized_volatility = daily_exchange_rate['USDINR'].rolling(window=22).std().resample('Y').mean() #approx 1 month rolling window, then annual mean
    realized_volatility.index = realized_volatility.index.year
    combined_data_realized = pd.concat([realized_volatility, capital_formation_data['CapitalFormation']], axis=1).dropna()
    combined_data_realized.columns = ['RealizedVolatility', 'CapitalFormation']

    scaler_realized = StandardScaler()
    scaled_realized = scaler_realized.fit_transform(combined_data_realized)
    scaled_realized_df = pd.DataFrame(scaled_realized, index=combined_data_realized.index, columns=combined_data_realized.columns)

    model_ardl_realized = ARDL(scaled_realized_df['CapitalFormation'], 1, scaled_realized_df['RealizedVolatility'], 1)
    results_ardl_realized = model_ardl_realized.fit()
    print("\nARDL with Realized Volatility:")
    print(results_ardl_realized.summary())
except FileNotFoundError:
    print("\nUSDINR_daily.csv not found. Skipping Realized Volatility Check.")

# 2. Alternative Model Specification (Varying ARDL Lags)
for p in range(1, 3):
    for q in range(1, 3):
        model_ardl_robust = ARDL(scaled_df['CapitalFormation'], p, scaled_df['ExchangeRateVolatility'], q)
        results_ardl_robust = model_ardl_robust.fit()
        print(f"\nARDL(p={p}, q={q}) Summary:")
        print(results_ardl_robust.summary())

# 3. Residual Diagnostics (Heteroskedasticity, Autocorrelation, Normality)
print("\nResidual Diagnostics (ARDL):")
bp_test = het_breuschpagan(results_ardl.resid, results_ardl.model.exog)
print(f"Breusch-Pagan Test: LM={bp_test[0]}, p-value={bp_test[1]}")

lb_test = acorr_ljungbox(results_ardl.resid, lags=[10], return_df=True)
print(f"Ljung-Box Test: Q={lb_test['lb'].values[0]}, p-value={lb_test['lb_pvalue'].values[0]}")

norm_test = normal_ad(results_ardl.resid)
print(f"Normality Test (AD): Statistic={norm_test[0]}, p-value={norm_test[1]}")

print("\nResidual Diagnostics (GARCH):")
bp_test_garch = het_breuschpagan(results_garch.resid, results_garch.conditional_volatility)
print(f"Breusch-Pagan Test (GARCH): LM={bp_test_garch[0]}, p-value={bp_test_garch[1]}")

lb_test_garch = acorr_ljungbox(results_garch.resid, lags=[10], return_df=True)
print(f"Ljung-Box Test (GARCH): Q={lb_test_garch['lb'].values[0]}, p-value={lb_test_garch['lb_pvalue'].values[0]}")

norm_test_garch = normal_ad(results_garch.resid)
print(f"Normality Test (GARCH): Statistic={norm_test_garch[0]}, p-value={norm_test_garch[1]}")

# 4. CUSUM Test for Model Stability
cusum_test = breaks_cusumolsresid(results_ardl.resid)
plt.figure(figsize=(10, 5))
plt.plot(cusum_test[1][0], label='CUSUM')
plt.plot(cusum_test[1][2][:, 0], '--', label='5% Critical Values')
plt.plot(cusum_test[1][2][:, 1], '--')
plt.title('CUSUM Test for ARDL Model Stability')
plt.legend()
plt.show()

# 5. ADF test of residuals.
adf_res_ardl = adfuller(results_ardl.resid)
print(f"\nADF test of ARDL residuals: ADF statistic: {adf_res_ardl[0]}, p-value: {adf_res_ardl[1]}")

adf_res_garch = adfuller(results_garch.resid)
print(f"\nADF test of GARCH residuals: ADF statistic: {adf_res_garch[0]}, p-value: {adf_res_garch[1]}")

# 6. Wavelet analysis using a different wavelet.
wavelet_robust = 'sym8'
coeffs_exchange_robust = pywt.wavedec(scaled_df['ExchangeRateVolatility'], wavelet_robust)
coeffs_capital_robust = pywt.wavedec(scaled_df['CapitalFormation'], wavelet_robust)

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
for i, coeff in enumerate(coeffs_exchange_robust):
    plt.plot(coeff, label=f'Exchange Rate Coeff {i}')
plt.title(f'Wavelet Coefficients ({wavelet_robust}) - Exchange Rate Volatility')
plt.legend()

plt.subplot(2, 1, 2)
for i, coeff in enumerate(coeffs_capital_robust):
    plt.plot(coeff, label=f'Capital Formation Coeff {i}')
plt.title(f'Wavelet Coefficients ({wavelet_robust}) - Capital Formation')
plt.legend()

plt.tight_layout()
plt.show()

# ... (Continue with other analyses and visualizations) ...