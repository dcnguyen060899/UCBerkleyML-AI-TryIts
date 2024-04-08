# decomposition_based_forecaster.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from UCBerkeley_MLAI.stl_seasonal_decompose.tools._extrapolate_trend import _extrapolate_trend
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class DecompositionBasedForecaster:
    def __init__(self, train_data, test_data, decomposed, model='additive', train_end_date=None):
        """
        Initializes the forecaster with training data, test data, decomposed components, and the model type.
        
        Parameters:
        - train_data: The training portion of the dataset.
        - test_data: The testing portion of the dataset, for which we want to forecast.
        - decomposed: The result of a seasonal decomposition (from STL or seasonal_decompose).
        - model: 'additive' or 'multiplicative', specifying the type of decomposition.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.decomposed = decomposed
        self.model = model
        self.train_end_date = train_end_date
        self.forecast_components = {}
        
    def forecast_trend(self, npoints):
        """Forecast the trend component as a constant continuation of the last observed trend value."""
        trend_with_nans = self.decomposed.trend
        extrapolated_trend = _extrapolate_trend(trend_with_nans, npoints)

        self.trend_without_nans = extrapolated_trend

        last_trend_value = self.trend_without_nans.iloc[-1]
        self.forecast_components['trend'] = pd.Series(index=self.test_data.index, data=last_trend_value)
    
    def forecast_seasonal(self, period, forecast_periods):
        # Example adjustment
        seasonal_cycle = self.decomposed.seasonal[-period:]  # Last observed cycle
        cycles_needed = forecast_periods // period + (1 if forecast_periods % period else 0)
        repeated_seasonal = pd.concat([seasonal_cycle] * cycles_needed)[:forecast_periods]
        self.forecast_components['seasonal'] = repeated_seasonal
        
    def combine_forecasts(self):
        """Combine the trend and seasonal forecasts according to the model type."""
        if self.model == 'additive':
            self.forecast_components['combined'] = self.forecast_components['trend'] + self.forecast_components['seasonal']
            
        elif self.model == 'multiplicative':
            self.forecast_components['combined'] = self.forecast_components['trend'] * self.forecast_components['seasonal']
        
    def forecast(self, period, forecast_periods, npoints):
        """
        Updates the forecasting process to use both the decomposition period and the number of forecast periods.
        """
        self.forecast_trend(npoints)
        self.forecast_seasonal(period, forecast_periods)  # Updated to accept forecast_periods
        self.combine_forecasts()

    def describe_forecast(self):
        forecast_description = {
            'projected_period_start': self.forecast_components['combined'].index.min().strftime('%Y-%m-%d'),
            'projected_period_end': self.forecast_components['combined'].index.max().strftime('%Y-%m-%d'),
            'model_type': self.model,
        }
        return forecast_description
    
    def print_forecast_description(self):
        description = self.describe_forecast()
        print(f"Forecast Description:")
        for key, value in description.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")

    def plot_forecasts(self):
        """Visualize the historical data, forecasted trend, combined forecast, and true future values."""
        plt.figure(figsize=(16,5))

        # Plot historical data
        plt.plot(self.train_data.index, self.train_data, label='Historical Data', color='blue', linewidth=1)

        # Plot the forecasted trend line
        if 'trend' in self.forecast_components:
            plt.plot(self.forecast_components['trend'], label='Forecasted Trend', color='orange', linestyle='--')

        # Plot the combined forecast (trend + seasonal)
        if 'combined' in self.forecast_components:
            plt.plot(self.forecast_components['combined'].index, self.forecast_components['combined'], label='Combined Forecast', color='red', linewidth=3)

        # If you have actual future data for comparison, plot that
        if self.test_data is not None:
            plt.plot(self.test_data.index, self.test_data, 'k--', alpha=0.5, label='True Future')

        plt.grid()
        plt.legend(loc='upper left', fontsize=18, framealpha=1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.title('Historical Data and Forecast')
        plt.show()

