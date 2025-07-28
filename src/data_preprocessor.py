import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def handle_outliers(df: pd.DataFrame, column: str, cap_percentile: float = 0.99) -> pd.DataFrame:
    """
    Caps the outliers in a specific column at a given percentile.
    This prevents extreme values from skewing the model.
    """
    cap_value = df[column].quantile(cap_percentile)
    df[column] = df[column].apply(lambda x: min(x, cap_value))
    print(f"Outliers in '{column}' capped at {cap_value:,.2f} ({cap_percentile * 100}th percentile).")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new, more meaningful features from existing ones.
    """
    current_year = 2024
    df['vehicle_age'] = current_year - df['year']
    df = df.drop('year', axis=1, errors='ignore')
    print("Feature 'vehicle_age' created from 'year'.")
    return df


def initial_clean(df: pd.DataFrame, columns_to_drop: list, target_column: str) -> pd.DataFrame:
    """
    Performs initial data cleaning: drops unnecessary columns and invalid target rows.
    """
    df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
    print(f"Initial columns dropped. Shape: {df.shape}")

    df = df[df[target_column] > 100]
    df = df.dropna(subset=[target_column])
    print(f"Rows with invalid price dropped. Shape: {df.shape}")

    return df


def drop_high_missing_value_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drops columns with a missing value percentage higher than the specified threshold.
    """
    missing_percentage = df.isnull().sum() / len(df)
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    if not cols_to_drop.empty:
        df = df.drop(columns=cols_to_drop, axis=1, errors='ignore')
        print(f"Columns with >{threshold * 100}% missing values dropped: {list(cols_to_drop)}. Shape: {df.shape}")
    else:
        print("No columns found with missing values above the threshold.")
    return df


def create_preprocessing_pipeline(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a scikit-learn pipeline to process numerical and categorical features.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: impute, then one-hot encode.
    # IMPORTANT FIX: We keep the output sparse to avoid memory errors.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # The key change is removing sparse_output=False or setting it to True.
        # This will output a sparse matrix, which is memory-efficient.
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor