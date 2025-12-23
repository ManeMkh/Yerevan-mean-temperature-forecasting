# Yerevan Weather Forecasting

Forecast **weekly mean temperatures** for Yerevan using historical weather data from **01.11.2017 to 21.03.2023**. The target variable is `temp`.

The project uses **two original CSV files** containing different weather observations. These files were **concatenated and preprocessed**, resulting in the final dataset: `yerevan_weather.csv`, which is used for modeling.

---
## Folder structure
```
├─ data/ 
│   ├─ yere1.csv  # original raw data
│   ├─ yere2.csv  # original raw data
│   └─ yerevan_weather.csv        # concatenated and preprocessed
├─ modeling.py                     # main forecasting code
├─ requirements.txt                # Python libraries
└─ README.md                        
```

## Features

The final dataset includes:

`name`, `datetime`, `tempmax`, `tempmin`, `temp`, `feelslikemax`, `feelslikemin`, `feelslike`, `dew`, `humidity`, `precip`, `precipprob`, `precipcover`, `preciptype`, `snow`, `snowdepth`, `windgust`, `windspeed`, `winddir`, `sealevelpressure`, `cloudcover`, `visibility`, `solarradiation`.

---

## Installation

Requires **Python 3.9 or higher**. Recommended steps for Windows:

```bat
python -m venv env_name
env_name\Scripts\activate
pip install -r requirements.txt
```
## Data Preprocessing

- The **two original CSV files were concatenated**.  

- Features like `tempmax`, `tempmin`, `feelslikemax`, `feelslikemin` were **dropped** because they directly influence the mean temperature.  

- Features like `severerisk`, `preciptype`, `solarenergy`, `snowdepth`, `snow`, `windgust`, and `moonphase` were **dropped** due to missing values or no explanatory power for `temp`.  

- **Daily data** was aggregated to **weekly averages** for the forecasting task.  

- **Stationarity tests** (ADF and KPSS) confirmed that the target series is stationary.  

- **ACF and PACF plots** of the target were used to identify ARIMA parameters:  
  - **ACF**: sharp decline until lag 26, then increase → seasonal AR component of 26.  
  - **PACF**: first two lags are significant → AR(2) component.
## Modeling

Three forecasting methods were applied:

1. **SARIMA (Seasonal ARIMA)**
2. **Auto-ARIMA**
3. **Holt-Winters / Exponential Smoothing**

**Evaluation metric:** Root Mean Squared Error (RMSE) on both **train and test sets**.  

The **best-performing model** was **Auto ARIMA**, providing the most accurate predictions for weekly mean temperatures.

---

## Evaluation & Results

- Models were evaluated using **RMSE** on both train and test sets.  
- **Auto ARIMA** achieved the lowest RMSE and provided the most reliable weekly forecasts.  

