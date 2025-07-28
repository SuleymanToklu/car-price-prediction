from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def create_full_pipeline(preprocessor: 'ColumnTransformer') -> Pipeline:
    """
    Creates the full pipeline by combining the preprocessor and the model.
    """
    # Define the model with some robust default parameters
    model = XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=7,
                         random_state=42,
                         n_jobs=-1)  # Use all available CPU cores

    # Create the full pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return full_pipeline
