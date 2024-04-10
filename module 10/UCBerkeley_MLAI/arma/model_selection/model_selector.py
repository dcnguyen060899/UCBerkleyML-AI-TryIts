import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import numpy as np

class ModelSelector:
    def __init__(self, series):
        self.series = series

    def choose_order(self, max_ar=3, max_ma=3):
        lag_acf = acf(self.series, nlags=max_ar)
        lag_pacf = pacf(self.series, nlags=max_ma, method='ols')

        plt.figure(figsize=(16, 7))
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.series)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.series)), linestyle='--', color='gray')
        plt.title('Autocorrelation Function')

        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.series)), linestyle='--', color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.series)), linestyle='--', color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()
