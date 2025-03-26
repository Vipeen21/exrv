#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:00:05 2025

@author: vipeenkumar
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron  # Correct PP test import

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


# Function to perform stationarity tests
def test_stationarity(series, name="Time Series"):
    print(f"\n{'='*40}\n📊 Stationarity Tests for {name}\n{'='*40}")

    # 1️⃣ Augmented Dickey-Fuller (ADF) Test
    adf_result = adfuller(series, autolag='AIC')
    print("\n🔹 Augmented Dickey-Fuller (ADF) Test")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"Critical Values: {adf_result[4]}")
    print("✅ Stationary" if adf_result[1] < 0.05 else "❌ Non-Stationary")

    # 2️⃣ KPSS Test
    kpss_result = kpss(series, regression='c', nlags="auto")
    print("\n🔹 KPSS Test")
    print(f"KPSS Statistic: {kpss_result[0]:.4f}")
    print(f"p-value: {kpss_result[1]:.4f}")
    print(f"Critical Values: {kpss_result[3]}")
    print("❌ Non-Stationary" if kpss_result[1] < 0.05 else "✅ Stationary")

    # 3️⃣ Phillips-Perron (PP) Test (from arch.unitroot)
    pp_test = PhillipsPerron(series)
    print("\n🔹 Phillips-Perron (PP) Test")
    print(f"PP Statistic: {pp_test.stat:.4f}")
    print(f"p-value: {pp_test.pvalue:.4f}")
    print(f"Critical Values: {pp_test.critical_values}")
    print("✅ Stationary" if pp_test.pvalue < 0.05 else "❌ Non-Stationary")

# Example usage
test_stationarity(combined_data["ExchangeRateVolatility"], "Exchange Rate Volatility")
test_stationarity(combined_data["CapitalFormation"], "Capital Formation")
