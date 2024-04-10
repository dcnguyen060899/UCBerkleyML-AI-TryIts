from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings


class ARIMA_fit:
    def __init__(self, series):
        self.series = series
        self.model = None
        self.results = None
        self.best_order = None
        self.best_aic = float('inf')

    def fit_model(self, order):
        """
        Fits an ARMA model to the series with the given order.
        """
        self.model = ARIMA(self.series, order=(order[0], 0, order[1]))
        self.results = self.model.fit()
        return self.results
    
    def grid_search(self, p_values, d_values, q_values):
        """
        Perform grid search to fit an ARIMA model with various combinations of p, d, and q.
        :param p_values: list, range of p values to try.
        :param d_values: list, range of d values to try.
        :param q_values: list, range of q values to try.
        """
        # Generate all different combinations of p, d, and q triplets
        pdq = list(itertools.product(p_values, d_values, q_values))
        
        for order in pdq:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Fit ARIMA model and calculate AIC
                    self.model = ARIMA(self.series, order=order)
                    results = self.model.fit()
                    aic = results.aic

                # Compare with best AIC
                if aic < self.best_aic:
                    self.best_aic = aic
                    self.best_result = results
                    self.best_order = order

            except Exception as e:
                continue

        print(f"Best ARIMA{self.best_order} model with AIC: {self.best_aic}")
        return self.best_result