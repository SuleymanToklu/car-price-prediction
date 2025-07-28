DATA_PATH = "data/vehicles.csv"
PIPELINE_SAVE_PATH = "saved_pipeline/price_prediction_pipeline.joblib"

# These columns will be dropped at the beginning of the process.
# We add high-cardinality features here to prevent memory errors from OneHotEncoding.
COLUMNS_TO_DROP = [
    'id', 'vin', 'url', 'region_url', 'image_url', 'description',
    'title_status', 'lat', 'long', 'posting_date', 'region', 'model',
    'type',           # Added: Too many unique values
    'paint_color',    # Added: Many unique values
    'manufacturer'    # Added: Many unique values
]

# The target variable we want to predict
TARGET_COLUMN = 'price'

# Features to be used for training, separated by type
# We will identify these more accurately after initial cleaning
NUMERICAL_FEATURES = ['year', 'odometer']
CATEGORICAL_FEATURES = ['condition', 'cylinders', 'fuel', 'transmission', 'drive', 'size', 'state']

# Threshold for dropping columns with high percentage of missing values
MISSING_VALUE_THRESHOLD = 0.4 # 40%