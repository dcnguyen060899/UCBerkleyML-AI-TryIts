from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import itertools

class SARIMA_fit:
    def __init__(self, series):
        self.series = series
        self.model = None
        self.results = None
        self.best_aic = float('inf')
        self.best_order = None
        self.best_seasonal_order = None
        self.best_results = None

    def fit_model(self, order, seasonal_order, trend='n', enforce_stationarity=True, enforce_invertibility=True):
        """
        Fits a SARIMA model to the series.
        
        :param order: tuple, the (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters.
        :param seasonal_order: tuple, the (P,D,Q,s) seasonal order of the model.
        :param trend: str {'n','c','t','ct'} or iterable, parameter controlling the deterministic trend polynomial A(t).
        :param enforce_stationarity: bool, whether or not to transform the AR parameters to enforce stationarity in the autoregressive component of the model.
        :param enforce_invertibility: bool, whether or not to transform the MA parameters to enforce invertibility in the moving average component of the model.
        """
        self.model = SARIMAX(self.series, order=order, seasonal_order=seasonal_order,
                             trend=trend, enforce_stationarity=enforce_stationarity,
                             enforce_invertibility=enforce_invertibility)
        self.results = self.model.fit(disp=False)
        return self.results
    
    def grid_search(self, pdq_values, seasonal_pdq_values, trend='n'):
        """
        Perform grid search to find the best SARIMA model based on AIC.
        
        :param pdq_values: list of tuples, non-seasonal (p,d,q) values to try.
        :param seasonal_pdq_values: list of tuples, seasonal (P,D,Q,s) values to try.
        :param trend: str, the trend parameter of the model.
        :return: tuple, best order and seasonal_order found.
        """

        # Iterate over all combinations of non-seasonal and seasonal parameters
        for param in pdq_values:
            for param_seasonal in seasonal_pdq_values:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.model = SARIMAX(self.series,
                                        order=param,
                                        seasonal_order=param_seasonal,
                                        trend=trend,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

                        results = self.model.fit(disp=False)

                    # Compare this model's AIC with the best AIC found so far
                    if results.aic < self.best_aic:
                        self.best_aic = results.aic
                        self.best_order = param
                        self.best_seasonal_order = param_seasonal
                        self.best_results = results
                except Exception as e:
                    continue

        self.model = SARIMAX(self.series,
                             order=self.best_order,
                             seasonal_order=self.best_seasonal_order,
                             trend=trend,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        self.results = self.best_results
        print(f'Best SARIMA{self.best_order}x{self.best_seasonal_order} - AIC:{self.best_aic}')
        return self.best_order, self.best_seasonal_order
