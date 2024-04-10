import pandas as pd
import matplotlib.pyplot as plt

class Forecaster:
    def __init__(self, model_results):
        self.results = model_results

    def forecast(self, steps=5):
        # For SARIMAXResults objects, the forecast method is used like this
        forecast = self.results.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return mean_forecast, conf_int
    
    def plot_forecast(self, steps=5, series=None):
        """
        Plots the forecasted values along with the original series.
        """
        forecast_mean, conf_int = self.forecast(steps=steps)
        plt.figure(figsize=(12, 6))
        if series is not None:
            plt.plot(series, label='Observed')
        forecast_index = pd.date_range(series.index[-1], periods=steps+1, freq=series.index.freq)[1:]
        plt.plot(forecast_index, forecast_mean, label='Forecast')
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
        plt.legend(loc='best')
        plt.title('Forecast with Confidence Intervals')
        plt.show()

    
        