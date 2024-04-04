from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from itertools import compress
import numpy as np

class ModelTrainer:
    def __init__(self, model, feature_selector=None, selection_strategy='SFS', numeric_features=None, categorical_features=None, alpha=None, original_columns=None):
        self.model = model
        self.feature_selector = feature_selector
        self.selection_strategy = selection_strategy
        self.numeric_features = numeric_features if numeric_features is not None else []
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.original_columns = original_columns
        self.pipeline = self.create_pipeline()
        self.description = f"{model.__class__.__name__}_{selection_strategy}"

        # Include alpha in the description if it's applicable to the model
        self.alpha = alpha  # Store alpha value
        if hasattr(model, 'alpha'):
            self.description = f"{model.__class__.__name__}_{selection_strategy}_alpha_{model.alpha}"
        else:
            self.description = f"{model.__class__.__name__}_{selection_strategy}"
        
    def create_pipeline(self):
        # Preprocessor for numeric and categorical features
        transformers = []
        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers) if transformers else None
        
        # Initialize steps for the pipeline
        steps = []
        if preprocessor is not None:
            steps.append(('preprocessor', preprocessor))
            
        # Add feature selection step if a feature_selector is provided
        if self.feature_selector:
            selector = self.feature_selector.get_selector(self.selection_strategy)
            steps.append(('feature_selection', selector))
        
        steps.append(('model', self.model))
        
        return Pipeline(steps)
    
    def get_feature_names(self):
        """Returns the feature names selected by the feature selector in the pipeline."""
        # If the original columns were not provided, we cannot retrieve the feature names
        if self.original_columns is None:
            raise ValueError("Original column names were not provided.")

        # If there's a preprocessor in the pipeline, get the transformed feature names
        if 'preprocessor' in self.pipeline.named_steps:
            preprocessor = self.pipeline.named_steps['preprocessor']
            if hasattr(preprocessor, 'get_feature_names_out'):
                transformed_features = preprocessor.get_feature_names_out()
            else:
                raise ValueError("The preprocessor does not support get_feature_names_out method.")
        else:
            transformed_features = self.original_columns

        # If there's a feature selection step, apply the selection to the feature names
        selected_features = transformed_features  # Default to all features
        if 'feature_selection' in self.pipeline.named_steps:
            selector = self.pipeline.named_steps['feature_selection']
            if hasattr(selector, 'get_support'):
                # This works for selectors that have a 'get_support' method
                support_mask = selector.get_support()
                selected_features = list(compress(transformed_features, support_mask))
            elif hasattr(selector, 'suport_'):
                # For selectors with a 'ranking_' attribute
                selected_indices = np.where(selector.support_)[0]
                selected_features = [transformed_features[idx] for idx in selected_indices]
            else:
                # Other types of selectors not covered above
                raise ValueError("Unknown feature selection method in pipeline.")

        return selected_features
    
    def train(self, X, y):
        # Fit the pipeline
        self.pipeline.fit(X, y)
        return self.pipeline