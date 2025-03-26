#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:37:55 2025

@author: vipeenkumar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:35:20 2025

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


# ARDL Model
# Define ARDL Model
model_ardl = ARDL(
    endog=scaled_df['CapitalFormation'],  
    exog=scaled_df[['ExchangeRateVolatility']],  
    lags=3,  
    order={'ExchangeRateVolatility': 3}  #  Use column name, not index
)

# Fit the model
results_ardl = model_ardl.fit()
print(results_ardl.summary())

# GARCH Model (Volatility Modeling)
garch_model = arch_model(scaled_df['ExchangeRateVolatility'], vol='Garch', p=1, q=1) #p and q values may need adjustments
results_garch = garch_model.fit(disp='off')
print(results_garch.summary())

# Granger Causality Test
granger_test = grangercausalitytests(scaled_df[['CapitalFormation', 'ExchangeRateVolatility']], maxlag=5, verbose=True) #maxlag may need adjustment

# VAR Model (for impulse response and variance decomposition, if needed)
model_var = VAR(scaled_df)
results_var = model_var.fit()
print(results_var.summary())
irf = results_var.irf(10) #Impulse Response Function
irf.plot(orth=False) #plot irf
fevd = results_var.fevd(10) #Forecast Error Variance Decomposition
fevd.plot() #plot fevd



# Visualization
plt.figure(figsize=(12, 6))
plt.plot(combined_data['ExchangeRateVolatility'], label='Exchange Rate Volatility')
plt.plot(combined_data['CapitalFormation'], label='Capital Formation')
plt.title('Exchange Rate Volatility and Capital Formation')
plt.xlabel('Year')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(results_garch.conditional_volatility, label='GARCH Volatility')
plt.title('GARCH Conditional Volatility')
plt.xlabel('Year')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(results_ardl.fittedvalues, label='Fitted Values ARDL')
plt.plot(scaled_df['CapitalFormation'], label='Actual Capital Formation')
plt.title('ARDL Fitted vs. Actual Capital Formation')
plt.xlabel('Year')
plt.legend()
plt.show()

# Residual diagnostics for ARDL and GARCH models.
residuals_ardl = results_ardl.resid
residuals_garch = results_garch.resid

plt.figure(figsize=(12,6))
plt.plot(residuals_ardl, label='ARDL Residuals')
plt.title('ARDL Residuals')
plt.xlabel('Year')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(residuals_garch, label='GARCH Residuals')
plt.title('GARCH Residuals')
plt.xlabel('Year')
plt.legend()
plt.show()

# Residual Diagnostics (Heteroskedasticity, Autocorrelation, Normality)
print("\nResidual Diagnostics (ARDL):")
# Convert exog to DataFrame before adding constant
exog_ardl = pd.DataFrame(results_ardl.model.exog, index=combined_data.index)
exog_ardl.columns = [f'Lag{i}' for i in range(exog_ardl.shape[1])]  # Automatically names columns
exog_ardl = sm.add_constant(exog_ardl)  # Adding constant term as bp test requires two exo term including constant
exog_ardl = exog_ardl.loc[results_ardl.resid.index]  # Ensure matching index

bp_test = het_breuschpagan(results_ardl.resid, exog_ardl)
print(f"ARDL Exog Shape: {exog_ardl.shape}")




#bp_test = het_breuschpagan(results_ardl.resid, results_ardl.model.exog)
print(f"Breusch-Pagan Test: LM={bp_test[0]}, p-value={bp_test[1]}")

lb_test = acorr_ljungbox(results_ardl.resid, lags=[10], return_df=True)
print(f"Ljung-Box Test: Q={lb_test['lb_stat'].values[0]}, p-value={lb_test['lb_pvalue'].values[0]}")


norm_test = normal_ad(results_ardl.resid)
print(f"Normality Test (AD): Statistic={norm_test[0]}, p-value={norm_test[1]}")

print("\nResidual Diagnostics (GARCH):")

print("Checking GARCH residuals for NaN or Inf values:")
print("NaNs", results_garch.resid.isna().sum())  # Check NaNs
print("Infinite values", np.isinf(results_garch.resid).sum())  # Check Inf

print("Checking conditional volatility for NaN or Inf values:")
print("NANs", results_garch.conditional_volatility.isna().sum())
print("Infinite values", np.isinf(results_garch.conditional_volatility).sum())

print("Residuals index:", results_garch.resid.index)
print("Conditional Volatility index:", results_garch.conditional_volatility.index)


exog_garch = pd.DataFrame(results_garch.conditional_volatility, index=results_garch.resid.index, columns=['Volatility'])

# Add a constant term (required for BP test)
exog_garch = sm.add_constant(exog_garch)
print("ensuring matching index")
exog_garch = exog_garch.loc[results_garch.resid.index]  # Ensure matching index
print(f"Shape after alignment: {exog_garch.shape}")
exog_garch = exog_garch.apply(pd.to_numeric, errors='coerce') #forcing exog_garch to be numeric

print("Data types in exog_garch:")
print(exog_garch.dtypes)

print("Before cleaning exog_garch:")
print("NANs")
print(exog_garch.isna().sum())  # Count NaNs
print("infinite values")
print(np.isinf(exog_garch).sum())  # Count Infs

print("after cleaning exog_garch")
# Ensure no NaN or Inf
exog_garch.replace([np.inf, -np.inf], np.nan, inplace=True)
exog_garch.fillna(exog_garch.mean(), inplace=True)

print("Final GARCH exogenous data shape:", exog_garch.shape)
print("Checking NaN/Inf after filling NaNs with column mean:")
print("NAN") 
print(exog_garch.isna().sum())
print("Infinite Values") 
print(np.isinf(exog_garch).sum())

if exog_garch.empty:
    print("exog_garch is empty after conversion!")
else:
    print("exog_garch contains data.")

#ensuring residuals and conditional volatility are aligned
print("Residuals index:", results_garch.resid.index)
print("Conditional Volatility index:", results_garch.conditional_volatility.index)

exog_garch = exog_garch.reindex(results_garch.resid.index)
print(exog_garch)



bp_test_garch = het_breuschpagan(results_garch.resid, exog_garch)
print(f"GARCH Volatility Shape: {exog_garch.shape}")

bp_test_garch = het_breuschpagan(results_garch.resid, results_garch.conditional_volatility)
print(f"Breusch-Pagan Test (GARCH): LM={bp_test_garch[0]}, p-value={bp_test_garch[1]}")

lb_test_garch = acorr_ljungbox(results_garch.resid, lags=[10], return_df=True)
print(f"Ljung-Box Test (GARCH): Q={lb_test_garch['lb_stat'].values[0]}, p-value={lb_test_garch['lb_pvalue'].values[0]}")

norm_test_garch = normal_ad(results_garch.resid)
print(f"Normality Test (GARCH): Statistic={norm_test_garch[0]}, p-value={norm_test_garch[1]}")

