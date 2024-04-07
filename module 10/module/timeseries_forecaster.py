import pandas as pd
import matplotlib.pyplot as plt
from decomposition_based_forecaster import DecompositionBasedForecaster  # Make sure to define this class
from ts_visualizer import Visualizer
from uncertainty_evaluator import UncertaintyEvaluator

class TimeSeriesForecaster:
    def __init__(self, decomposer, train_end_date, forecast_periods):
        """
        Initializes the forecaster with a TimeSeriesDecomposer instance, training end date,
        and the number of forecast periods.
        
        Parameters:
        - decomposer: An instance of TimeSeriesDecomposer that has already decomposed the series.
        - train_end_date: The date that marks the end of the training data and the start of the testing data.
        - forecast_periods: The number of periods to forecast into the future.
        """
        self.decomposer = decomposer
        self.train_end_date = pd.to_datetime(train_end_date)
        self.forecast_periods = forecast_periods

    def forecast_and_plot(self, title_prefix=''):
        """
        Forecasts future values using the decomposed components and plots the results.
        """

        # Split the dataset using the correct date format
        train_data = self.decomposer.df.loc[:self.train_end_date][self.decomposer.data_col]
        test_data = self.decomposer.df.loc[self.train_end_date:][self.decomposer.data_col]
        
        # Ensure the decomposer has already decomposed the series
        if self.decomposer.decomposed is None:
            self.decomposer.decompose()

        # Initialize and fit the forecaster
        
        forecaster = DecompositionBasedForecaster(train_data, test_data, self.decomposer.decomposed, model=self.decomposer.model, train_end_date=self.train_end_date)
        npoints = self.decomposer.period
        forecaster.forecast(self.decomposer.period, self.forecast_periods, npoints)
        
        self.forecaster = forecaster.forecast(self.decomposer.period, self.forecast_periods, npoints)

        # Plotting the decomposition results
        if self.decomposer.decomposition_method == 'seasonal_decompose':
            Visualizer.plot_decomposition(self.decomposer.decomposed, title_prefix=title_prefix)
        elif self.decomposer.decomposition_method == 'STL':
            Visualizer.plot_stl_decomposition(self.decomposer.decomposed, title_prefix=title_prefix)

        # Plotting the forecast results
        forecaster.plot_forecasts()

        # describe forecast
        forecaster.describe_forecast()
        forecaster.print_forecast_description()

        uncertainty_evaluator = UncertaintyEvaluator(forecaster, test_data)
        uncertainty_evaluator.evaluate()
        uncertainty_evaluator.print_evaluation()