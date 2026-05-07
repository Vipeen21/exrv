
---

# 📉 Exchange Rate Volatility & Stock Market Analysis (EXRV)

This repository contains a comprehensive suite of econometric and statistical tools for analyzing the relationship between **Exchange Rate Volatility** and **Stock Market Performance**. Using advanced time-series techniques, this project aims to uncover lead-lag relationships, volatility clustering, and long-term co-integration.

---

## 🔬 Econometric Methodologies

The project is structured around several high-level analytical frameworks:

### 1. Volatility Modeling (GARCH)
* **`GARCH and wavelet 2.py`**: Implementation of Generalized Autoregressive Conditional Heteroskedasticity models to capture volatility clustering.
* **`GARCH residual diagnostic.py`**: Statistical testing (ARCH-LM, Ljung-Box) to ensure model fitness and white noise residuals.

### 2. Time-Frequency Analysis (Wavelets)
* **`wavelet power spectrum.py`**: Decomposing time series into different frequency components to observe how the exchange-stock relationship changes across different time horizons (short-term vs. long-term).
* **`fourier and wavelet practice.py`**: Comparative spectral analysis.

### 3. Structural Modeling & Co-integration
* **`ardl model.py`**: Autoregressive Distributed Lag (ARDL) models for identifying long-run and short-run dynamics between variables.
* **`var and irf.py`**: Vector Autoregression (VAR) and Impulse Response Functions (IRF) to simulate how a shock in exchange rates affects the stock market.

---

## 📂 Repository Breakdown

| Script | Purpose |
| :--- | :--- |
| **`stationarity test...py`** | Unit root testing via ADF, KPSS, and Phillips-Perron (PP) tests. |
| **`robustness checks.py`** | Sensitivity analysis to validate the consistency of the econometric results. |
| **`final excrf.py`** | The primary execution script for the final exchange rate framework. |

---

## 🛠️ Prerequisites & Installation

To run these scripts, you will need **Python 3.x** and the following quantitative libraries:

```bash
pip install pandas numpy statsmodels arch pywavelets matplotlib seaborn
```

### Usage
1.  **Stationarity First**: Run the ADF/KPSS scripts to ensure your data is integrated of the same order.
2.  **Model Selection**: Use the ARDL or GARCH scripts based on your research objectives.
3.  **Visualization**: Use the Wavelet Power Spectrum to generate heatmaps of volatility across time scales.

---

## ⚖️ License

This project is licensed under the **[MIT License](https://github.com/Vipeen21/exrv/blob/main/LICENSE)** - see the file for details.

---

## 🤝 Contribution & Research
This repository is part of a broader research project into financial market interdependencies. If you find these scripts useful for your own academic work, feel free to fork the repo or submit a Pull Request.

*Maintained by [Vipeen Kumar](https://github.com/Vipeen21)*

---


**Do you have a specific dataset or country (like India) that this analysis focuses on? I can add a "Data" section if you do!**
