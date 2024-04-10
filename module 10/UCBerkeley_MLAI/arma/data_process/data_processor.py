import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, train_series, test_series=None, maxlag=10):
        """
        Initializes the DataProcessor with training and optional testing time series.
        :param train_series: The training time series.
        :param test_series: The testing time series (optional).
        """
        self.train_series = train_series
        self.test_series = test_series
        self.maxlag = maxlag

    def test_stationarity(self, window=12, cutoff=0.05):
        """
        Tests the stationarity of the training time series and plots the rolling mean and std.
        :param window: The window size for calculating rolling statistics.
        :param cutoff: The p-value cutoff to determine stationarity.
        """
        rolmean = self.train_series.rolling(window=window).mean()
        rolstd = self.train_series.rolling(window=window).std()
        
        print('Results of Dickey-Fuller Test on Training Series:')
        dftest = adfuller(self.train_series, self.maxlag, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

        plt.figure(figsize=(12, 6))
        plt.plot(self.train_series, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation of Training Series')
        plt.show(block=False)

        return dftest[1] <= cutoff

    def make_stationary(self, diff_order=1):
        """
        Differencing the training series to make it stationary.
        :param diff_order: The differencing order.
        """
        self.train_series = self.train_series.diff(periods=diff_order).dropna()