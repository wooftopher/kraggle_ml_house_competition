import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import itertools
import sys
import os

PARAMS = {
    'tree_method': 'hist',  # Use histogram-based method
    'device': 'cuda:0',     # Use GPU for training
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'mae',  # Mean Absolute Error for consistency
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.4,
    'seed': 42
}

def train_xgb_on_training_data(X_train, y_train, X_valid, y_valid, params=PARAMS, num_boost_round=2000, early_stopping_rounds=100, nfold=5):
    """Train the final XGBoost model using the best hyperparameters found via grid search."""
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Cross-validation to find the best number of boosting rounds
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        early_stopping_rounds=early_stopping_rounds,
        as_pandas=True,
        seed=42
    )

    # Get the best number of boosting rounds
    best_num_boost_round = cv_results['test-mae-mean'].idxmin()
    best_cv_mae = cv_results['test-mae-mean'].min()

    print(f"Best number of boosting rounds: {best_num_boost_round}")
    print(f"Best CV MAE: {best_cv_mae}")

    # Train final model with the best number of boosting rounds
    final_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=best_num_boost_round,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )

    # Make predictions on train and validation
    train_predictions = final_model.predict(dtrain)
    valid_predictions = final_model.predict(dvalid)

    # Compute MAE
    train_mae = mean_absolute_error(y_train, train_predictions)
    valid_mae = mean_absolute_error(y_valid, valid_predictions)

    print(f"XGBoost Final Model - Training MAE: {train_mae}")
    print(f"XGBoost Final Model - Validation MAE: {valid_mae}")

    return final_model, train_mae, valid_mae, best_cv_mae


# Define a smaller grid for the initial test
param_grid = {
    'max_depth': [5, 6],  # Test both best values
    'eta': [0.04, 0.05],  # Test both best values
    'subsample': [0.7, 0.75],  # Test both best values
    'colsample_bytree': [0.4, 0.45],  # Test both best values
}

def grid_search_with_cv(X_train, y_train, X_valid, y_valid, param_grid, num_boost_round=1000, early_stopping_rounds=300, nfold=5):
    best_mae = float('inf')
    best_params = None
    best_round_overall = None

    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*param_grid.values()))

    for params_comb in param_combinations:
        # Map the parameter combination back to the param names
        params = dict(zip(param_grid.keys(), params_comb))
        params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'device': 'cuda:0',  # Use GPU for training
            'seed': 42
        })

        # Set up DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        # Train model using training and validation sets
        evals = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        # Predict on validation set
        y_pred_valid = model.predict(dvalid)
        mae_valid = mean_absolute_error(y_valid, y_pred_valid)

        # Extract optimized params
        optimized_params = {key: params[key] for key in ['max_depth', 'eta', 'subsample', 'colsample_bytree']}

        print(f"Run with params: {optimized_params}")
        print(f"Best round: {model.best_iteration}, Valid MAE: {mae_valid}\n")

        # Track best model
        if mae_valid < best_mae:
            best_mae = mae_valid
            best_params = params
            best_round_overall = model.best_iteration

    print(f"Best hyperparameters: {best_params}")
    print(f"Best Valid MAE: {best_mae}, Best round: {best_round_overall}")
    return best_params, best_round_overall


# Function to train XGBoost model on full dataset (for final submission)
def train_xgb_on_full_data(X, y, params=None, num_boost_round=500, early_stopping_rounds=25, n_splits=5):
    # Set default PARAMS if none are provided
    if params is None:
        params = {
            'tree_method': 'hist',  # Use histogram-based method
            'device': 'cuda:0',     # Use GPU for training
            'objective': 'reg:squarederror',  # Regression task
            'eval_metric': 'mae',  # Mean Absolute Error for consistency
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.4,
            'seed': 42
        }

    # Set up DMatrix for XGBoost
    dtrain = xgb.DMatrix(X, label=y)

    # Perform cross-validation with early stopping
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=n_splits,  # 5-fold cross-validation
        # early_stopping_rounds=early_stopping_rounds,  # Stop early if no improvement
        metrics='mae',  # Mean Absolute Error for tracking
        as_pandas=True,  # Return results as a pandas DataFrame
        verbose_eval=100  # Print progress every 100 rounds
    )

    # Find the best iteration based on cross-validation MAE
    best_iteration = cv_results['test-mae-mean'].argmin()

    print(f"Best boosting round found during cross-validation: {best_iteration} with MAE: {cv_results['test-mae-mean'][best_iteration]}")

    # Train the model on the full dataset using the best number of boosting rounds
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=best_iteration,  # Use the best boosting round from CV
        verbose_eval=100  # Print progress every 100 rounds
    )

    print("Xgboost model trained on full dataset.")
    return bst, dtrain

# Main block for testing XGBoost model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('../input/train.csv')

    # Run the grid search with cross-validation
    # grid_search_with_cv(X_train_transformed, y_train, X_valid_transformed, y_valid, param_grid)
    # sys.exit()
    # Train XGBoost model on training data (for validation)
    bst, mae_train, mae_valid, xgb_cv_mae = train_xgb_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train XGBoost model on full dataset (for final submission)
    # bst_full = train_xgb_on_full_data(X_transformed, y)