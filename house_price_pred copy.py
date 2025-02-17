import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import warnings

import sys
sys.stdout = open('./result.log', 'w')

# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ------------------------------------------------------DATA and FEATURE-------------------------------------
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
        ]), num_cols)
    ],
    verbose_feature_names_out=False  # Ensures clean feature names
)

# Function to apply preprocessing and retain feature names
def transform_data(preprocessor, X_train, X_test):
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert back to DataFrame with correct column names
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    return X_train_transformed, X_test_transformed

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformation and restore feature names
X_train_transformed, X_valid_transformed = transform_data(preprocessor, X_train, X_valid)

# Convert transformed data to NumPy arrays
X_train_np = X_train_transformed.to_numpy()
X_valid_np = X_valid_transformed.to_numpy()
y_train_np = y_train.to_numpy()
y_valid_np = y_valid.to_numpy()

# Create DeviceQuantileDMatrix for GPU training
dtrain = xgb.DeviceQuantileDMatrix(X_train_np, label=y_train_np)
dvalid = xgb.DeviceQuantileDMatrix(X_valid_np, label=y_valid_np)

# Apply preprocessing to the entire dataset
X_transformed, _ = transform_data(preprocessor, X, X)

# Convert transformed data to NumPy arrays
X_np = X_transformed.to_numpy()
y_np = y.to_numpy()


# -------------------------------------------------------MODELS and TRAINING------------------------------------

# Define XGBoost parameters for GPU training
params = {
    'tree_method': 'hist',  # Use histogram-based method
    'device': 'cuda:0',     # Use GPU for training
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',  # Root Mean Squared Error
    'max_depth': 6,
    'eta': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train the model
bst = xgb.train(params, dtrain, num_boost_round=500, evals=[(dvalid, 'validation')], early_stopping_rounds=10)


# Define the model
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Fit the model
elastic_model.fit(X_train, y_train)


#------------------------------------------------------TESTING RESULT------------------------------------------
# Make predictions on validation set
y_pred = bst.predict(dvalid)

# Additional evaluation
final_r2 = r2_score(y_valid_np, y_pred)
final_mae = mean_absolute_error(y_valid_np, y_pred)

print(f'Final Validation R²: {final_r2:.4f}')
print(f'Final Validation MAE: {final_mae:.4f}')

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = []
cv_mae_scores = []

for train_index, val_index in kf.split(X_np):
    X_train, X_val = X_np[train_index], X_np[val_index]
    y_train, y_val = y_np[train_index], y_np[val_index]

    # Create DeviceQuantileDMatrix for GPU training
    dtrain = xgb.DeviceQuantileDMatrix(X_train, label=y_train)
    dval = xgb.DeviceQuantileDMatrix(X_val, label=y_val)

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval, 'validation')], early_stopping_rounds=10)

    # Make predictions on the validation set
    y_pred = bst.predict(dval)

    # Calculate R² and MAE for this fold
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # Append scores to lists
    cv_r2_scores.append(r2)
    cv_mae_scores.append(mae)

    print(f'Validation R² for this fold: {r2:.4f}')
    print(f'Validation MAE for this fold: {mae:.4f}')

# Calculate mean and standard deviation of R² and MAE
mean_r2 = np.mean(cv_r2_scores)
std_r2 = np.std(cv_r2_scores)
mean_mae = np.mean(cv_mae_scores)
std_mae = np.std(cv_mae_scores)

print(f'Cross-Validation R²: {mean_r2:.4f} ± {std_r2:.4f}')
print(f'Cross-Validation MAE: {mean_mae:.4f} ± {std_mae:.4f}')

print(f'Final Validation R²: {final_r2:.4f}')
print(f'Final Validation MAE: {final_mae:.4f}')

sys.stdout.close()