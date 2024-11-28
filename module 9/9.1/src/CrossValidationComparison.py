from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np


class CrossValidationComparison:
    def __init__(self, X, y, alpha=0.01):
        self.X = X
        self.y = y
        self.alpha = alpha
    
    def auto_detect_columns(self):
        """Automatically detect numeric and categorical columns."""
        categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_columns = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return numeric_columns, categorical_columns

    def setup_model(self):
        """Setup the preprocessing pipeline and model pipeline based on detected column types."""
        numeric_columns, categorical_columns = self.auto_detect_columns()
        preprocessor = self.create_preprocessor(numeric_columns, categorical_columns)
        self.model = self.create_pipeline(preprocessor, alpha=self.alpha)
    
    @staticmethod
    def create_preprocessor(numeric_columns, categorical_columns):
        """Create a preprocessing pipeline."""
        numeric_transformer = Pipeline(steps=[
            ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = OneHotEncoder(drop='if_binary', handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
            ]
        )
        return preprocessor
    
    @staticmethod
    def create_pipeline(preprocessor, alpha=0.01):
        """Create the main model pipeline."""
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('selector', SelectFromModel(Lasso(alpha=alpha))),
            ('linreg', LinearRegression())
        ])
        return model_pipeline


    def loocv_score(self):
        loo = LeaveOneOut()
        scores = cross_val_score(self.model, self.X, self.y, cv=loo, scoring='neg_mean_squared_error')
        return np.mean(scores)

    def holdout_score(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        return -score  # Negated to keep consistency with other scores

    def kfold_score(self, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, self.X, self.y, cv=kf, scoring='neg_mean_squared_error')
        return np.mean(scores)

    def compare_methods(self):
        loocv = self.loocv_score()
        holdout = self.holdout_score()
        kfold = self.kfold_score()

        print(f"LOOCV Score: {loocv}")
        print(f"Holdout Score: {holdout}")
        print(f"K-Fold Score: {kfold}")

        best_score = max(loocv, holdout, kfold)
        best_method = 'LOOCV' if best_score == loocv else ('Holdout' if best_score == holdout else 'K-Fold')
        print(f"{best_method} yielded the best score.")
