from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class ModelTrainer:
    def __init__(self, model, feature_selector=None, selection_strategy='SFS', numeric_features=None, categorical_features=None):
        self.model = model
        self.feature_selector = feature_selector
        self.selection_strategy = selection_strategy
        self.numeric_features = numeric_features if numeric_features is not None else []
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.pipeline = self.create_pipeline()
        self.description = f"{model.__class__.__name__}_{selection_strategy}"  # For identification
        
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
    
    def train(self, X, y):
        # Fit the pipeline
        self.pipeline.fit(X, y)
        return self.pipeline