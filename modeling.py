import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import pickle
import os

class WeatherForecaster:
    def __init__(self, path, target_col, train_size=0.7):
        self.df = pd.read_csv(path)
        self.target = target_col
        self.train_size = train_size
        self.train_data = None
        self.test_data = None
        self.y_train = None
        self.y_test = None
        self.sarima_model = None
        self.arima_model = None
        self.exp_smooth_model = None
    
    def split_data(self):
        n_train = int(len(self.df) * self.train_size)
        self.train_data = self.df.iloc[:n_train]
        self.test_data = self.df.iloc[n_train:]
        self.y_train = self.train_data[self.target]
        self.y_test = self.test_data[self.target]

    def fit_sarima(self):
        self.sarima_model = SARIMAX(self.y_train, order=(26,0,2), seasonal_order = (2,0,3,52)).fit()
        return self.sarima_model
    

    def fit_arima(self, start_p=1, start_q=1, max_p=7, max_q=7, m=26):
        self.arima_model = pm.auto_arima(self.y_train, start_p=start_p, start_q=start_q,
                                  test="adf", max_p=max_p, max_q=max_q, m=m,
                                  seasonal=True, start_P=0,
                                  D=None, trace = True,
                                  error_action="ignore",
                                  suppress_warnings=True, stepwise=True)
        return self.arima_model
    
    def fit_exp_smooth(self):
        self.exp_smooth_model = ExponentialSmoothing(self.y_train, trend=None, seasonal="add", seasonal_periods=52).fit()
        return self.exp_smooth_model
    

    def forecast(self, model_type="sarima"):
        n_test = len(self.y_test)

        if model_type == "sarima":
            train_preds = self.sarima_model.fittedvalues
            test_preds = self.sarima_model.forecast(steps=n_test)
        
        elif model_type == "arima":
            train_preds = self.arima_model.predict_in_sample()
            test_preds = self.arima_model.predict(n_periods=n_test)
           
        elif model_type == "exp_smooth":
            train_preds = self.exp_smooth_model.fittedvalues
            test_preds = self.exp_smooth_model.forecast(steps=n_test)

        else:
            raise ValueError("The model should be sarima, arima or exp_smooth")
        
        return train_preds, test_preds


    def evaluate(self, preds):
        train_preds, test_preds = preds
        rmse_train = np.sqrt(mean_squared_error(self.y_train, train_preds))
        rmse_test = np.sqrt(mean_squared_error(self.y_test, test_preds))
        return rmse_train, rmse_test


    def save_model(self, model_type="sarima", path="model.pkl"):
        if model_type == "sarima":
            model = self.sarima_model
        elif model_type == "arima":
            model = self.arima_model
        elif model_type == "exp_smooth":
            model = self.exp_smooth_model
        else:
            raise ValueError("The model should be sarima, arima or exp_smooth")

        with open(path, "wb") as f:
            pickle.dump(model, f)

        print(f"{model_type} model saved to {path}")


    def load(self, path, model_type = "sarima"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        if model_type == "sarima":
            self.sarima_model = model
        elif model_type == "arima":
            self.arima_model = model
        elif model_type == "exp_smooth":
            self.exp_smooth_model = model
        print(f"{model_type} model loaded from {path}")

    
    def forecast_future(self, steps=5, model_type="exp_smooth"):
        """
        Forecast future values for given steps using the specified model.
        Fits the model on full data if it hasn't been fitted yet.
        """
        if model_type == "exp_smooth":
            if self.exp_smooth_model is None:
                self.exp_smooth_model = ExponentialSmoothing(
                    self.df[self.target], trend=None, seasonal="add", seasonal_periods=52
                ).fit()
            future = self.exp_smooth_model.forecast(steps=steps)

        elif model_type == "sarima":
            if self.sarima_model is None:
                self.sarima_model = SARIMAX(
                    self.df[self.target], order=(26,0,2), seasonal_order=(2,0,3,52)
                ).fit()
            future = self.sarima_model.forecast(steps=steps)

        elif model_type == "arima":
            if self.arima_model is None:
                self.arima_model = pm.auto_arima(
                    self.df[self.target], seasonal=True, m=52, stepwise=True
                )
            future = self.arima_model.predict(n_periods=steps)

        else:
            raise ValueError("Model must be 'exp_smooth', 'sarima', or 'arima'")
        
        return future



def main():
    data_path = "yerevan_weather.csv"
    target_col = "temp"

    wf = WeatherForecaster(data_path, target_col)
    wf.split_data()

    print("\nSARIMA")
    if os.path.exists("sarima_model.pkl"):
        print("Loading existing SARIMA model...")
        wf.load("sarima_model.pkl", "sarima")
    else:
        print("Training SARIMA...")
        wf.fit_sarima()
        wf.save_model("sarima", "sarima_model.pkl")

    train_preds, test_preds = wf.forecast("sarima")
    rmse_train, rmse_test = wf.evaluate((train_preds, test_preds))
    print(f"SARIMA RMSE -> Train: {rmse_train:.3f}, Test: {rmse_test:.3f}")

    print("\nAUTO ARIMA")
    if os.path.exists("auto_arima_model.pkl"):
        print("Loading existing Auto ARIMA model...")
        wf.load("auto_arima_model.pkl", "arima")
    else:
        print("Training Auto ARIMA...")
        wf.fit_arima()
        wf.save_model("arima", "auto_arima_model.pkl")

    train_preds, test_preds = wf.forecast("arima")
    rmse_train, rmse_test = wf.evaluate((train_preds, test_preds))
    print(f"Auto ARIMA RMSE -> Train: {rmse_train:.3f}, Test: {rmse_test:.3f}")

    print("\nEXPONENTIAL SMOOTHING")
    if os.path.exists("exp_smooth_model.pkl"):
        print("Loading existing Exp Smoothing model...")
        wf.load("exp_smooth_model.pkl", "exp_smooth")
    else:
        print("Training Exp Smoothing...")
        wf.fit_exp_smooth()
        wf.save_model("exp_smooth", "exp_smooth_model.pkl")

    train_preds, test_preds = wf.forecast("exp_smooth")
    rmse_train, rmse_test = wf.evaluate((train_preds, test_preds))
    print(f"Exponential Smoothing RMSE -> Train: {rmse_train:.3f}, Test: {rmse_test:.3f}")
   
    print("\nHOLTE WINTERS FUTURE FORECAST")
    wf.exp_smooth_model = None   # or freshly initialized object
    future_preds = wf.forecast_future(steps=5, model_type="exp_smooth")
    print(f"Future predictions: {future_preds}")

    print("AUTO ARIMA FUTURE FORECAST")
    wf.arima_model = None   # or freshly initialized object

    future_arima = wf.forecast_future(steps=5, model_type="arima")
    print(f"Future Auto ARIMA predictions:\n{future_arima}")

    print("SARIMA FUTURE FORECAST")
    wf.sarima_model = None   # or freshly initialized object

    future_sarima = wf.forecast_future(steps=5, model_type="sarima")
    print(f"Future SARIMA predictions:\n{future_sarima}\n")


if __name__ == "__main__":
    main()


