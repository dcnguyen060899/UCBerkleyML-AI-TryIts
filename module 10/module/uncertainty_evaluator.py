import matplotlib.pyplot as plt

class UncertaintyEvaluator:
    def __init__(self, model_results, test_data):
        self.model_results = model_results
        self.test_data = test_data

    def evaluate(self):
        residuals = self.test_data - self.model_results.forecast_components['combined']
        evaluation_metrics = {
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'residual_plot': residuals.plot(title='Residuals over Time'),
        }
        # Additional metrics like MAE, RMSE, etc., can be included here
        return evaluation_metrics

    def print_evaluation(self):
        evaluation = self.evaluate()
        print(f"Uncertainty and Model Evaluation:")
        for key, value in evaluation.items():
            if key == 'residual_plot':
                plt.show()
            else:
                print(f"{key.replace('_', ' ').capitalize()}: {value}")
