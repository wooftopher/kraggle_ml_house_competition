from data import load_and_preprocess_data
from models.xgb_m import train_xgb_on_training_data, train_xgb_on_full_data
from models.rf_m import train_rf_on_training_data, train_rf_on_full_data
from models.lgb_m import train_lgbm_on_training_data, train_lgbm_on_full_data
from models.elas_m import train_elastic_on_training_data, train_elastic_on_full_data
from models.catboost_m import train_catboost_on_training_data, train_catboost_on_full_data
from models.gb_m import train_gbm_on_training_data, train_gbm_on_full_data
from models.svr_m import train_svr_on_training_data, train_svr_on_full_data
from models.knn_m import train_knn_on_training_data, train_knn_on_full_data
from models.brr_m import train_bayesian_ridge_on_training_data, train_bayesian_ridge_on_full_data
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import itertools
import numpy as np
import xgboost as xgb
import sys
import heapq 

# Load and preprocess data
X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('../input/train.csv')

# Train XGBoost model on training data (for validation)
bst, mae_train_xgb, mae_valid_xgb, xgb_cv_mae = train_xgb_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Train Random Forest model on training data (for validation)
rf_model, mae_train_rf, mae_valid_rf, cv_mae_rf = train_rf_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Train LightGBM model on training data (for validation)
lgbm_model, mae_train_lgbm, mae_valid_lgbm, cv_mae_lgbm = train_lgbm_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Train ElasticNet model on training data (for validation)
elastic_model, mae_train_elastic, mae_valid_elastic, cv_mae_elastic = train_elastic_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Train CatBoost model on training data (for validation)
catboost_model, mae_train_catboost, mae_valid_catboost, cv_mae_catboost = train_catboost_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Train GBM model on training data (for validation)
gbm_model, mae_train, mae_valid, cv_mae = train_gbm_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Train SVR model on training data (for validation)
svr_model, mae_train, mae_valid, cv_mae, scaler = train_svr_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Train KNN model on training data (for validation)
knn_model, mae_train, mae_valid, cv_mae = train_knn_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)
# Train Bayesian Ridge Regression model on training data (for validation)
bayesian_ridge_model, mae_train_bayes_ridge, mae_valid_bayes_ridge, cv_mae_bayes_ridge = train_bayesian_ridge_on_training_data(
    X_train_transformed, y_train, X_valid_transformed, y_valid
)

# Collect predictions from all models
models = {
    "XGBoost": (bst.predict(xgb.DMatrix(X_train_transformed)), bst.predict(xgb.DMatrix(X_valid_transformed))),
    "RandomForest": (rf_model.predict(X_train_transformed), rf_model.predict(X_valid_transformed)),
    "LightGBM": (lgbm_model.predict(X_train_transformed), lgbm_model.predict(X_valid_transformed)),
    "ElasticNet": (elastic_model.predict(X_train_transformed), elastic_model.predict(X_valid_transformed)),
    "CatBoost": (catboost_model.predict(X_train_transformed), catboost_model.predict(X_valid_transformed)),
    "GBM": (gbm_model.predict(X_train_transformed), gbm_model.predict(X_valid_transformed)),
    "SVR": (
        svr_model.predict(X_train_transformed.to_numpy()),  # Convert to NumPy array
        svr_model.predict(X_valid_transformed.to_numpy())  # Convert to NumPy array
    ),
    "KNN": (
            knn_model.predict(X_train_transformed),
            knn_model.predict(X_valid_transformed)
        ),
    "BayesianRidge": (
            bayesian_ridge_model.predict(X_train_transformed),
            bayesian_ridge_model.predict(X_valid_transformed)
    )
}

# Initialize a list to store the top 3 combinations
top_combos = []

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

        # Track the top 3 combinations with the lowest MAE
        heapq.heappush(top_combos, (-mae_stack, combo))  # Use negative MAE for min-heap behavior
        if len(top_combos) > 3:  # Keep only the top 3
            heapq.heappop(top_combos)

# Print the top 3 stacking models (lowest MAE)
print("\nTop 3 Stacking Combinations (Lowest MAE):")
for i, (neg_mae, combo) in enumerate(sorted(top_combos, reverse=True), start=1):
    print(f"{i}. {combo} -> MAE: {-neg_mae}")  # Convert negative MAE back to positive

sys.exit()


# Train models on full dataset (for final submission)
bst_full = train_xgb_on_full_data(X_transformed, y)
rf_model_full = train_rf_on_full_data(X_transformed, y)
lgbm_model_full = train_lgbm_on_full_data(X_transformed, y)
elastic_model_full = train_elastic_on_full_data(X_transformed, y)
catboost_model_full = train_catboost_on_full_data(X_transformed, y)
gbm_model_full = train_gbm_on_full_data(X_transformed, y)
svr_model_full, scaler_full = train_svr_on_full_data(X_transformed, y)
knn_model_full = train_knn_on_full_data(X_transformed, y)
bayesian_ridge_model_full = train_bayesian_ridge_on_full_data(X_transformed, y)
