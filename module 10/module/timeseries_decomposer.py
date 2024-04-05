# timeseries_decomposer.py
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from data_preparer import DataPreparer
from ts_visualizer import Visualizer

class TimeSeriesDecomposer:
    def __init__(self, df, date_col, data_col, freq, model, period):
        self.df = df
        self.date_col = date_col
        self.data_col = data_col
        self.freq = freq
        self.model = model
        self.period = period
        self.decomposed = None

    def decompose(self):
        preparer = DataPreparer(self.df, self.date_col, self.freq)
        prepared_df = preparer.prepare()
        ts = prepared_df[self.data_col]       # Should be an integer
        self.decomposed = seasonal_decompose(ts, model=self.model, period=self.period)
        
    def plot_decompositions(self, title_prefix=''):
        Visualizer.plot_decomposition(self.decomposed, title_prefix=title_prefix)