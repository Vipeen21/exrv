#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:28:11 2025

@author: vipeenkumar
"""

import pandas as pd
from statsmodels.tsa.ardl import ARDL, ardl_select_order
# Load data from a CSV file. Ensure your data is clean and properly formatted.
data = pd.read_csv('/Users/vipeenkumar/Downloads/my own research/exchange rate volatility and capitatl formation india/ exchange rate dataset/exrcfdata.csv')

#Define your dependent variable (endogenous) and independent variables (exogenous).
endog = data['Capital Formation']  # Replace with your dependent variable
exog = data[['Exchange Rate Volatility']]  # Replace with your independent variables, you can add more variables using comma inbetween

#Use the ardl_select_order function to automatically select the optimal lag lengths based on criteria like AIC or BIC.
sel_res = ardl_select_order(endog, maxlag=2, exog=exog, ic='aic', maxorder=2)
print(f"The optimal order is: {sel_res.model.ardl_order}")

#Fit the ARDL model using the selected lag order.
model = ARDL(endog, lags=sel_res.model.ardl_order[0], exog=exog, order=sel_res.model.ardl_order[1:])
ardl_fit = model.fit()

#View the Results
print(ardl_fit.summary())

#General Model Summary:
#Dependent Variable: Capital Formation (suggesting you are analyzing factors influencing capital formation).

#Observations: 32 (which may be on the lower side for robust inference).

#Log Likelihood: -68.423 (a measure of model fit; higher (less negative) values indicate better fit).

#Method: Conditional Maximum Likelihood Estimation.

#AIC (Akaike Information Criterion): 148.845 (used to compare models; lower values suggest a better fit).

#BIC (Bayesian Information Criterion): 157.049 (penalizes model complexity more than AIC).

#HQIC (Hannan-Quinn Information Criterion): 151.414.

#Interpretation of Coefficients:
#Variable	Coefficient	Std. Error	z-value	p-value	95% Confidence Interval	Interpretation
#Constant	5.4961	3.547	1.550	0.134	(-1.824, 12.816)	Not statistically significant (p > 0.05). Suggests that when all other variables are zero, capital formation is around 5.5.
#Capital Formation (L1)	0.5640	0.209	2.694	0.013	(0.132, 0.996)	Statistically significant at 5% level. A 1-unit increase in past capital formation (1 lag) leads to a 0.564 increase in current capital formation.
#Capital Formation (L2)	0.2918	0.215	1.355	0.188	(-0.153, 0.736)	Not statistically significant. Past values of capital formation at lag 2 have a weaker effect.
#Capital Formation (L3)	0.0036	0.199	0.018	0.986	(-0.406, 0.414)	Not statistically significant. No meaningful effect at lag 3.
#Exchange Rate Volatility (L3)	-0.4090	0.580	-0.705	0.488	(-1.607, 0.789)	Not statistically significant. Suggests that exchange rate volatility at lag 3 has no strong effect on capital formation.
#Key Takeaways:
#Capital Formation has strong autocorrelation, with a significant impact from its first lag (L1) but not from longer lags (L2 and L3).

#Exchange Rate Volatility (L3) does not significantly impact Capital Formation, based on this model.

#The model fit (AIC, BIC, HQIC) suggests moderate performance, but the small sample size (32 observations) might limit its reliability.

#Only Capital Formation (L1) is statistically significant—meaning past capital formation is a strong predictor of future capital formation.

