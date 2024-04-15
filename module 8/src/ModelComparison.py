from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer

class ModelComparison(BaseEstimator):
    def __init__(self, dataframe, target_column, categorical_columns, numerical_columns, poly_degree=2):
        self.dataframe = dataframe
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.models_mse = {}
        self.poly_degree = poly_degree
        self.fitted_models = {}
        
    def train_baseline_model(self, X_train, y_train, X_test, y_test):
        baseline_pred = np.full_like(y_test, y_train.mean())
        mse = mean_squared_error(y_test, baseline_pred)
        self.models_mse['Baseline Model'] = mse
        
    def train_one_hot_model(self, X_train, y_train, X_test, y_test):
        categorical_transformer = OneHotEncoder()
        preprocessor = ColumnTransformer(transformers=[
            ('cat', categorical_transformer, self.categorical_columns)
        ])
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        self.models_mse['One-Hot Model'] = mse
        
    def train_poly_one_hot_model(self, X_train, y_train, X_test, y_test):
        # Define transformers for numerical and categorical columns
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Impute numerical columns using the median
            ('poly', PolynomialFeatures(degree=self.poly_degree))  # Then apply polynomial features
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical columns using the most frequent value
            ('onehot', OneHotEncoder(drop='if_binary'))  # Then apply one-hot encoding
        ])

        # Use make_column_transformer to apply the transformations
        preprocessor = make_column_transformer(
            (numerical_transformer, self.numerical_columns),
            (categorical_transformer, self.categorical_columns),
            remainder='passthrough'  # Include any columns not specified in numerical or categorical columns without any changes
        )

        # Define the model pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Fit the model
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        self.models_mse['Poly + One-Hot Model'] = mse
        
    def fit(self):
        X = self.dataframe.drop(columns=[self.target_column], axis=1)
        y = self.dataframe[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_baseline_model(X_train, y_train, X_test, y_test)

        self.train_one_hot_model(X_train, y_train, X_test, y_test)
        self.fitted_models['One-Hot Model'] = self.model

        self.train_poly_one_hot_model(X_train, y_train, X_test, y_test)
        self.fitted_models['Poly + One-Hot Model'] = self.model
        
    def get_model_performance(self):
        return self.models_mse

    def get_best_model(self):
        # Identify the model with the lowest MSE
        best_model_name = min(self.models_mse, key=self.models_mse.get)
        best_mse = self.models_mse[best_model_name]
        # Retrieve the best model
        best_model = self.fitted_models.get(best_model_name)
        return best_model_name, best_mse, best_model
    
    def predict(self, X):
        if self.best_model is None:
            raise ValueError("This ModelComparison instance is not fitted yet.")
        return self.best_model.predict(X)
