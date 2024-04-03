from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.linear_model import Lasso, Ridge
from ML_AI_DuyNguyen.feature_selector import RidgeFeatureSelector

class FeatureSelector:
    def __init__(self, n_features_to_select=4, alpha=0.01):
        self.n_features_to_select = n_features_to_select
        self.alpha = alpha

    def get_selector(self, strategy='SFS'):
        if strategy == 'SFS':
            selector = SequentialFeatureSelector(Lasso(alpha=self.alpha), n_features_to_select=self.n_features_to_select)
        elif strategy == 'RFE':
            selector = RFE(Lasso(alpha=self.alpha), n_features_to_select=self.n_features_to_select)
        elif strategy == 'RidgeImportance':
            selector = RidgeFeatureSelector(Ridge(alpha=self.alpha), n_features_to_select=self.n_features_to_select)
        else:
            raise ValueError("Unsupported feature selection strategy")
        
        # return the selected importance feature using fit_transform()
        return selector
