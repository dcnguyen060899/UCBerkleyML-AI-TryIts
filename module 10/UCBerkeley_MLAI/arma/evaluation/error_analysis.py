from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class ErrorAnalysis:
    def __init__(self, actual, predicted):
        """
        Initialize with actual and predicted values.
        :param actual: array-like, true values.
        :param predicted: array-like, predicted values.
        """
        self.actual = actual
        self.predicted = predicted
        self.errors = actual - predicted

    def mean_absolute_error(self):
        """
        Calculate the mean absolute error.
        :return: float, MAE.
        """
        return mean_absolute_error(self.actual, self.predicted)

    def mean_squared_error(self):
        """
        Calculate the mean squared error.
        :return: float, MSE.
        """
        return mean_squared_error(self.actual, self.predicted)

    def root_mean_squared_error(self):
        """
        Calculate the root mean squared error.
        :return: float, RMSE.
        """
        return np.sqrt(self.mean_squared_error())

    def report(self):
        """
        Generate a report of the errors.
        :return: dict, containing the error metrics.
        """
        return {
            'MAE': self.mean_absolute_error(),
            'MSE': self.mean_squared_error(),
            'RMSE': self.root_mean_squared_error()
        }
