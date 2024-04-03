from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
import numpy as np

class RidgeFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, n_features_to_select):
        self.model = model
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        self.model.fit(X, y)
        self.feature_importances_ = np.abs(self.model.coef_)
        return self

    def transform(self, X):
        # Rank features based on absolute coefficient values
        ranked_features = np.argsort(self.feature_importances_)[::-1]
        # Select the top n_features_to_select
        top_features_indices = ranked_features[:self.n_features_to_select]
        return X[:, top_features_indices]
