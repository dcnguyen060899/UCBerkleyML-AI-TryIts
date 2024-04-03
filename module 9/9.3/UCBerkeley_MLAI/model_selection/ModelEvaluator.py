from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
# Assuming ModelTrainer is updated to return a pipeline
from UCBerkeley_MLAI.model_selection import ModelTrainer
# Optionally, CrossValidator can be used if it provides added value in managing CV strategies
from UCBerkeley_MLAI.split import HoldoutSplit

class ModelEvaluator:
    def __init__(self, model_trainers, cv_strategies):
        """
        Initializes the ModelEvaluator with a list of ModelTrainer instances and a dictionary 
        of cross-validation (CV) strategies and their configurations.

        Parameters:
        - model_trainers: List of ModelTrainer instances.
        - cv_strategies: Dictionary where keys are strategy names and values are configurations.
        """
        self.model_trainers = model_trainers
        self.cv_strategies = cv_strategies

    def evaluate(self, X, y):
        """
        Evaluates each model trained by the model trainers using the specified CV strategies.
        Prints the average Mean Squared Error (MSE) for each combination of CV strategy and model.
        """
        results = {}
        feature_importances = {}

        # Iterate over each CV strategy
        for strategy_name, strategy_config in self.cv_strategies.items():
            cv = self._select_cv(strategy_name, strategy_config)
            
            # Iterate over each split provided by the CV strategy
            for train_index, test_index in cv.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                # Evaluate each model on the current split
                for trainer in self.model_trainers:
                    pipeline = trainer.train(X_train, y_train)  # Trainer now returns a pipeline
                    y_pred = pipeline.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)

                    # Record the MSE scores
                    key = (strategy_name, trainer.description)
                    results.setdefault(key, []).append(mse)

                    if hasattr(trainer, 'get_feature_names'):
                        feature_importances[key] = trainer.get_feature_names()

        # Print the average MSE for each CV strategy and model combination
        self._print_results(results)
        return results, feature_importances

    def _select_cv(self, strategy_name, strategy_config):
        """
        Selects the cross-validation strategy based on the provided name and configuration.
        """
        if strategy_name == "KFold":
            return KFold(**strategy_config)
        elif strategy_name == "LOO":
            return LeaveOneOut()
        elif strategy_name == 'Holdout':
            return HoldoutSplit(**strategy_config)
        else:
            raise ValueError(f"Unsupported CV strategy: {strategy_name}")

    def _print_results(self, results):
        """
        Prints the average MSE for each combination of CV strategy and model.
        """
        for key, scores in results.items():
            avg_mse = np.mean(scores)
            print(f"{key}: Avg MSE = {avg_mse}")
