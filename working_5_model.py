import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings

import sys
# sys.stdout = open('./result.log', 'w')

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
            ('scaler', StandardScaler())  
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

# Replace whitespaces with underscores in the feature names before training
X_train_transformed.columns = X_train_transformed.columns.str.replace(' ', '_')
X_valid_transformed.columns = X_valid_transformed.columns.str.replace(' ', '_')


# -------------------------------------------------------MODELS and TRAINING------------------------------------


#----------------------------------------------------xgb----------------------------------------
params = {
    'tree_method': 'hist',  # Use histogram-based method
    'device': 'cuda:0',     # Use GPU for training
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'mae',  # Mean Absolute Error for consistency
    'max_depth': 6,
    'eta': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Set up DMatrix for XGBoost (without using Quantile DMatrix)
dtrain = xgb.DMatrix(X_train_transformed, label=y_train)
dvalid = xgb.DMatrix(X_valid_transformed, label=y_valid)

# Perform cross-validation using xgb.cv
cv_results = xgb.cv(
    params=params, 
    dtrain=dtrain, 
    num_boost_round=922, 
    nfold=5,  # Number of folds in cross-validation
    early_stopping_rounds=10,  # Stop if no improvement in the last 10 rounds
    as_pandas=True,  # Return results as pandas DataFrame
    seed=42
)

# Extract best boosting round based on MAE (not RMSE)
best_num_boost_round = cv_results['test-mae-mean'].idxmin()
xgb_cv_mae = cv_results['test-mae-mean'].min()

print(f"Best number of boosting rounds: {best_num_boost_round}")

# Train the final model using the best number of rounds
bst = xgb.train(
    params=params, 
    dtrain=dtrain, 
    num_boost_round=best_num_boost_round, 
    evals=[(dvalid, 'validation')],
    verbose_eval=False
)

# Make predictions
xgb_predictions_train = bst.predict(dtrain)
xgb_predictions_test = bst.predict(dvalid)

# Compute MAE
mae_train = mean_absolute_error(y_train, xgb_predictions_train)
mae_test = mean_absolute_error(y_valid, xgb_predictions_test)

# Print the MAE for both sets
print(f"XGBoost Model - Training MAE: {mae_train}")
print(f"XGBoost Model - Test MAE: {mae_test}")
print(f"XGBoost Model - CV MAE: {xgb_cv_mae}\n")

#-----------------------------------------------------elastic------------------------------------------

elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_model.fit(X_train_transformed, y_train)

elastic_predictions_train = elastic_model.predict(X_train_transformed)
elastic_predictions_test = elastic_model.predict(X_valid_transformed)

# ElasticNet Model - Training and Test MAE
elastic_mae_train = mean_absolute_error(y_train, elastic_predictions_train)
elastic_mae_test = mean_absolute_error(y_valid, elastic_predictions_test)
elastic_cv_scores = cross_val_score(elastic_model, X_train_transformed, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f"ElasticNet Model - Training MAE: {elastic_mae_train}")
print(f"ElasticNet Model - Test MAE: {elastic_mae_test}")
print(f"ElasticNet Model - CV MAE: {-elastic_cv_scores.mean()}\n")

#--------------------------------------------randomforest--------------------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_transformed, y_train)

rf_predictions_train = rf_model.predict(X_train_transformed)
rf_predictions_valid = rf_model.predict(X_valid_transformed)

# RandomForest Model - Training and Test MAE
rf_mae_train = mean_absolute_error(y_train, rf_predictions_train)
rf_mae_test = mean_absolute_error(y_valid, rf_predictions_valid)
rf_cv_scores = cross_val_score(rf_model, X_train_transformed, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f"RandomForest Model - Training MAE: {rf_mae_train}")
print(f"RandomForest Model - Test MAE: {rf_mae_test}")
print(f"RandomForest Model - CV MAE: {-rf_cv_scores.mean()}\n")

#-----------------------------------------------------LightGBM------------------------------------------
from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgbm_model.fit(X_train_transformed, y_train)

lgbm_predictions_train = lgbm_model.predict(X_train_transformed)
lgbm_predictions_test = lgbm_model.predict(X_valid_transformed)

# LightGBM Model - Training and Test MAE
lgbm_mae_train = mean_absolute_error(y_train, lgbm_predictions_train)
lgbm_mae_test = mean_absolute_error(y_valid, lgbm_predictions_test)
lgbm_cv_scores = cross_val_score(lgbm_model, X_train_transformed, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f"LightGBM Model - Training MAE: {lgbm_mae_train}")
print(f"LightGBM Model - Test MAE: {lgbm_mae_test}")
print(f"LightGBM Model - CV MAE: {-lgbm_cv_scores.mean()}\n")

#-----------------------------------------------------CatBoost------------------------------------------
from catboost import CatBoostRegressor

catboost_model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0, random_state=42)
catboost_model.fit(X_train_transformed, y_train)

catboost_predictions_train = catboost_model.predict(X_train_transformed)
catboost_predictions_test = catboost_model.predict(X_valid_transformed)

# CatBoost Model - Training and Test MAE
catboost_mae_train = mean_absolute_error(y_train, catboost_predictions_train)
catboost_mae_test = mean_absolute_error(y_valid, catboost_predictions_test)
catboost_cv_scores = cross_val_score(catboost_model, X_train_transformed, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f"CatBoost Model - Training MAE: {catboost_mae_train}")
print(f"CatBoost Model - Test MAE: {catboost_mae_test}")
print(f"CatBoost Model - CV MAE: {-catboost_cv_scores.mean()}\n")

#-----------------------------------------------------MLPRegressor------------------------------------------
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp_model.fit(X_train_transformed, y_train)

mlp_predictions_train = mlp_model.predict(X_train_transformed)
mlp_predictions_test = mlp_model.predict(X_valid_transformed)

# MLPRegressor Model - Training and Test MAE
mlp_mae_train = mean_absolute_error(y_train, mlp_predictions_train)
mlp_mae_test = mean_absolute_error(y_valid, mlp_predictions_test)
mlp_cv_scores = cross_val_score(mlp_model, X_train_transformed, y_train, cv=5, scoring='neg_mean_absolute_error')

print(f"MLPRegressor Model - Training MAE: {mlp_mae_train}")
print(f"MLPRegressor Model - Test MAE: {mlp_mae_test}")
print(f"MLPRegressor Model - CV MAE: {-mlp_cv_scores.mean()}\n")

#---------------------------------------------meta-learner ridge--------------------------------------

# Define base models and their predictions
models = {
    "XGBoost": (xgb_predictions_train, xgb_predictions_test),
    "ElasticNet": (elastic_predictions_train, elastic_predictions_test),
    "RandomForest": (rf_predictions_train, rf_predictions_valid),
    "LightGBM": (lgbm_predictions_train, lgbm_predictions_test),
    "CatBoost": (catboost_predictions_train, catboost_predictions_test),
    "MLPRegressor": (mlp_predictions_train, mlp_predictions_test)
}


# List all possible stacking combinations (excluding empty set)
best_mae = float("inf")
best_combo = None

for r in range(1, len(models) + 1):
    for combo in itertools.combinations(models.keys(), r):
        selected_train_preds = [models[m][0] for m in combo]
        selected_test_preds = [models[m][1] for m in combo]

        # Stack selected model predictions
        stacked_train = np.column_stack(selected_train_preds)
        stacked_test = np.column_stack(selected_test_preds)

        # Train the meta-learner
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(stacked_train, y_train)

        # Make final predictions
        final_predictions = meta_learner.predict(stacked_test)

        # Compute MAE
        mae_stack = mean_absolute_error(y_valid, final_predictions)

        # Print the combination and its MAE
        print(f"Stacking {combo} -> MAE: {mae_stack}")

        # Track the best model
        if mae_stack < best_mae:
            best_mae = mae_stack
            best_combo = combo

# Print the best stacking model
print(f"\nBest Stacking Combination: {best_combo} -> MAE: {best_mae}")

sys.exit()

#------------------------------------------------------TESTING RESULT------------------------------------------

# Define KFold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store MAE and R² for each fold
mae_scores = []
r2_scores = []

# Perform cross-validation
for train_index, val_index in kf.split(X_train_transformed):
    # Split data
    X_train_cv, X_valid_cv = X_train_transformed.iloc[train_index], X_train_transformed.iloc[val_index]
    y_train_cv, y_valid_cv = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Train models (same as before)
    bst = xgb.train(params, xgb.DMatrix(X_train_cv, label=y_train_cv), num_boost_round=500, evals=[(xgb.DMatrix(X_valid_cv, label=y_valid_cv), 'validation')], early_stopping_rounds=10, verbose_eval=False)
    elastic_model.fit(X_train_cv, y_train_cv)
    
    # Make predictions
    xgb_predictions_cv = bst.predict(xgb.DMatrix(X_valid_cv))
    elastic_predictions_cv = elastic_model.predict(X_valid_cv)
    
    # Stack predictions and train meta-learner
    stacked_cv = np.column_stack((xgb_predictions_cv, elastic_predictions_cv))
    meta_learner.fit(stacked_cv, y_valid_cv)
    
    # Final predictions
    final_predictions_cv = meta_learner.predict(stacked_cv)
    
    # Calculate MAE and R² for this fold
    mae_fold = mean_absolute_error(y_valid_cv, final_predictions_cv)
    r2_fold = r2_score(y_valid_cv, final_predictions_cv)
    
    # Append scores to lists
    mae_scores.append(mae_fold)
    r2_scores.append(r2_fold)
    
    # Print MAE and R² for this fold
    print(f"Fold MAE: {mae_fold}")
    print(f"Fold R²: {r2_fold}")

# Calculate average MAE and R² across all folds
average_mae = np.mean(mae_scores)
average_r2 = np.mean(r2_scores)

print(f"Average MAE across all folds: {average_mae}")
print(f"Average R² across all folds: {average_r2}")

# sys.stdout.close()
