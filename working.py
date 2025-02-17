import pandas as pd
import numpy as np  # Add this import for numpy
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, DMatrix  # Import DMatrix here
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
import xgboost as xgb
import warnings

# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the data
iowa_file_path = './input/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Extract target variable
y = home_data['SalePrice']

# Identify numerical and categorical features dynamically
num_cols = home_data.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice').tolist()
cat_cols = home_data.select_dtypes(include=['object']).columns.tolist()

# Define feature matrix
X = home_data[num_cols + cat_cols]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("num", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())  # Scaling numeric features
        ]), [col for col in X.columns if col not in cat_cols])    ]
)

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models with optimized parameters (CPU use only)

xgb_model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.01, 
    max_depth=6, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    random_state=42,
    tree_method="hist",  # "hist" is efficient on CPU as well
    # Removed 'device="cuda"' for CPU usage
)

lgbm_model = LGBMRegressor(
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    verbose=-1
)

catboost_model = CatBoostRegressor(
    iterations=500, 
    learning_rate=0.01, 
    depth=6, 
    random_state=42, 
    verbose=0,
    task_type="CPU"  # Change to "CPU" for CatBoost to use CPU
)

lr_model = LinearRegression()

# ElasticNet with a higher number of iterations, regularization parameters adjusted
elasticnet_model = ElasticNet(
    max_iter=10000, 
    alpha=10, 
    l1_ratio=0.5, 
    random_state=42
)

# Define the meta-learner (Ridge regression)
meta_learner = Ridge(alpha=1.0)

# Define the stacking model
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('cat', catboost_model),
        ('lr', lr_model),
        ('elastic', elasticnet_model)
    ],
    final_estimator=meta_learner  # Ridge regression as the meta-learner
)

# Create pipeline
stacking_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', stacked_model)])



# Perform cross-validation
cv_r2 = cross_val_score(stacking_pipeline, X, y, scoring='r2', cv=5)
cv_mae = cross_val_score(stacking_pipeline, X, y, scoring='neg_mean_absolute_error', cv=5)

# Get mean and standard deviation
print(f'R² Score: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}')
print(f'MAE: {-cv_mae.mean():.4f} ± {cv_mae.std():.4f}')

# Cross-validation predictions
y_pred_cv = cross_val_predict(stacking_pipeline, X, y, cv=5)

# Additional evaluation
final_r2 = r2_score(y, y_pred_cv)
final_mae = mean_absolute_error(y, y_pred_cv)

print(f'Final Cross-Validation R²: {final_r2:.4f}')
print(f'Final Cross-Validation MAE: {final_mae:.4f}')