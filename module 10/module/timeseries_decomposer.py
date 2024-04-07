# timeseries_decomposer.py
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from data_preparer import DataPreparer
from ts_visualizer import Visualizer
from decomposition_based_forecaster import DecompositionBasedForecaster

class TimeSeriesDecomposer:
    def __init__(self, df, date_col, data_col, freq, model, period, decomposition_method='seasonal_decompose'):
        self.df = df
        self.date_col = date_col
        self.data_col = data_col
        self.freq = freq
        self.model = model
        self.period = period
        self.decomposition_method = decomposition_method
        self.decomposed = None

    def decompose(self):
        preparer = DataPreparer(self.df, self.date_col, self.freq)
        prepared_df = preparer.prepare()
        ts = prepared_df[self.data_col]

        if self.decomposition_method == 'seasonal_decompose':
            if self.period is None:
                raise ValueError("Period must be specified for seasonal_decompose.")
            self.decomposed = seasonal_decompose(ts, model=self.model, period=self.period)
            
        elif self.decomposition_method == 'STL':
            if self.period is None:
                raise ValueError("Period must be specified for STL decomposition.")
            stl = STL(ts, period=self.period)
            self.decomposed = stl.fit()
        else:
            raise ValueError(f"Unsupported decomposition method: {self.decomposition_method}")

    def plot_decompositions(self, title_prefix=''):
        # Plotting the decomposition results
        if self.decomposition_method == 'seasonal_decompose':
            Visualizer.plot_decomposition(self.decomposed, title_prefix=title_prefix)
        elif self.decomposition_method == 'STL':
            Visualizer.plot_stl_decomposition(self.decomposed, title_prefix=title_prefix)