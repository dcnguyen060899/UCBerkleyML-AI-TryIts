from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureSelectionAndRegression(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=4, alpha=0.01):
        self.n_features_to_select = n_features_to_select
        self.alpha = alpha


    def create_sfs_pipeline(self):
        """Creates a pipeline for Sequential Feature Selection with Lasso."""
        lasso_for_sfs = Lasso(alpha=self.alpha)
        sfs_pipeline = Pipeline([
            ('sfs', SequentialFeatureSelector(lasso_for_sfs, n_features_to_select=self.n_features_to_select)),
            ('reg', LinearRegression())
        ])
        return sfs_pipeline

    def create_rfe_pipeline(self):
        """Creates a pipeline for Recursive Feature Elimination with Lasso."""
        lasso_for_rfe = Lasso(alpha=self.alpha)
        rfe_pipeline = Pipeline([
            ('rfe', RFE(lasso_for_rfe, n_features_to_select=self.n_features_to_select)),
            ('reg', LinearRegression())
        ])
        return rfe_pipeline

    def create_ridge_pipeline(self):
        """Creates a pipeline for Ridge regression."""
        ridge_pipeline = Pipeline([
            ('ridge', Ridge(alpha=self.alpha))
        ])
        return ridge_pipeline

    def fit(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize pipelines
        self.sfs_pipeline = self.create_sfs_pipeline()
        self.rfe_pipeline = self.create_rfe_pipeline()
        self.ridge_pipeline = self.create_ridge_pipeline()
        
        # Fitting the pipelines
        self.sfs_pipeline.fit(X_train, y_train)
        self.rfe_pipeline.fit(X_train, y_train)
        self.ridge_pipeline.fit(X_train, y_train)

        return self
    

    def calculate_mse(self, pipeline):
        y_pred_train = pipeline.predict(self.X_train)
        y_pred_test = pipeline.predict(self.X_test)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        return mse_train, mse_test
    
    def feature_name_mapping(self, feature_indices, columns):
        """Maps feature indices to names."""
        return [columns[index] for index in feature_indices]
    
    def compare_models(self):
        # Print selected features for SFS and RFE
        sfs_features = np.where(self.sfs_pipeline.named_steps['sfs'].get_support())[0]
        rfe_features = np.where(self.rfe_pipeline.named_steps['rfe'].support_)[0]
        ridge_coefficients = np.argsort(np.abs(self.ridge_pipeline.named_steps['ridge'].coef_))[::-1]
        
        sfs_feature_names = self.feature_name_mapping(sfs_features, self.X_train.columns)
        rfe_feature_names = self.feature_name_mapping(rfe_features, self.X_train.columns)
        ridge_important_feature_names = self.feature_name_mapping(ridge_coefficients[:4], self.X_train.columns)
        
        print("Important features (Ridge):", ridge_important_feature_names)
        print("Selected features (SFS with Lasso):", sfs_feature_names)
        print("Selected features (RFE with Lasso):", rfe_feature_names)
        
        # Calculate MSE for each model
        mse_ridge_train, mse_ridge_test = self.calculate_mse(self.ridge_pipeline)
        mse_sfs_train, mse_sfs_test = self.calculate_mse(self.sfs_pipeline)
        mse_rfe_train, mse_rfe_test = self.calculate_mse(self.rfe_pipeline)

        print(f"Ridge - MSE Train: {mse_ridge_train}, MSE Test: {mse_ridge_test}")
        print(f"SFS with Lasso - MSE Train: {mse_sfs_train}, MSE Test: {mse_sfs_test}")
        print(f"RFE with Lasso - MSE Train: {mse_rfe_train}, MSE Test: {mse_rfe_test}")

        # Determine the best model based on test MSE
        best_test_mse = min(mse_ridge_test, mse_sfs_test, mse_rfe_test)
        best_model = "Ridge" if best_test_mse == mse_ridge_test else ("SFS with Lasso" if best_test_mse == mse_sfs_test else "RFE with Lasso")
        print(f"The best model based on test MSE is: {best_model}")
