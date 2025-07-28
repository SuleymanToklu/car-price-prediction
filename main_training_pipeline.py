import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Import our custom modules and config, including the new functions
from src import config
from src.data_preprocessor import (
    initial_clean,
    handle_outliers,
    feature_engineer,
    drop_high_missing_value_columns,
    create_preprocessing_pipeline
)
from src.model_trainer import create_full_pipeline

def run_training():
    """
    Executes the full model training pipeline.
    """
    print("--- Starting Improved Training Pipeline ---")

    # 1. Load Data
    df = pd.read_csv(config.DATA_PATH)
    print(f"Data loaded. Initial shape: {df.shape}")

    # 2. Initial Cleaning
    df_cleaned = initial_clean(df, config.COLUMNS_TO_DROP, config.TARGET_COLUMN)

    # 3. Handle Outliers (NEW STEP)
    # Cap extreme values for price and odometer before they can cause issues.
    df_cleaned = handle_outliers(df_cleaned, 'price', 0.99)
    df_cleaned = handle_outliers(df_cleaned, 'odometer', 0.99)

    # 4. Feature Engineering (NEW STEP)
    # Create the 'vehicle_age' feature.
    df_cleaned = feature_engineer(df_cleaned)

    # 5. Drop columns with too many missing values
    df_processed = drop_high_missing_value_columns(df_cleaned, config.MISSING_VALUE_THRESHOLD)

    # 6. Define features and target
    X = df_processed.drop(config.TARGET_COLUMN, axis=1)
    y = df_processed[config.TARGET_COLUMN]

    # Dynamically identify feature types from the remaining columns
    # 'year' is now 'vehicle_age'
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\nIdentified {len(numerical_features)} numerical features.")
    print(f"Identified {len(categorical_features)} categorical features.")

    # 7. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nData split into training and testing sets.")

    # 8. Create pipelines
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    full_pipeline = create_full_pipeline(preprocessor)
    print("\nFull pipeline created.")

    # 9. Train the model
    print("--- Training the model with improved data... ---")
    full_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # 10. Evaluate the pipeline
    score = full_pipeline.score(X_test, y_test)
    print(f"\nModel evaluation R^2 score on test set: {score:.4f}")

    # 11. Save the trained pipeline
    joblib.dump(full_pipeline, config.PIPELINE_SAVE_PATH)
    print(f"\nPipeline saved to: {config.PIPELINE_SAVE_PATH}")

    print("--- Training Pipeline Finished Successfully ---")

if __name__ == '__main__':
    run_training()