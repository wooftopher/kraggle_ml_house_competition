import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Define XGBoost parameters (keep them in this file)
PARAMS = {
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

# Function to train XGBoost model on training data (for validation)
def train_xgb_on_training_data(X_train, y_train, X_valid, y_valid, params=PARAMS, num_boost_round=922, nfold=5, early_stopping_rounds=10):
    """
    Trains an XGBoost model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        params (dict): XGBoost parameters.
        num_boost_round (int): Maximum number of boosting rounds.
        nfold (int): Number of folds for cross-validation.
        early_stopping_rounds (int): Early stopping rounds.
    
    Returns:
        bst (xgb.Booster): Trained XGBoost model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        xgb_cv_mae (float): Cross-validation MAE.
    """
    # Set up DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Perform cross-validation using xgb.cv
    cv_results = xgb.cv(
        params=params, 
        dtrain=dtrain, 
        num_boost_round=num_boost_round, 
        nfold=nfold,  # Number of folds in cross-validation
        early_stopping_rounds=early_stopping_rounds,  # Stop if no improvement in the last 10 rounds
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
    xgb_predictions_valid = bst.predict(dvalid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, xgb_predictions_train)
    mae_valid = mean_absolute_error(y_valid, xgb_predictions_valid)

    # Print the MAE for both sets
    print(f"XGBoost Model - Training MAE: {mae_train}")
    print(f"XGBoost Model - Valid MAE: {mae_valid}")
    print(f"XGBoost Model - CV MAE: {xgb_cv_mae}\n")

    return bst, mae_train, mae_valid, xgb_cv_mae

# Function to train XGBoost model on full dataset (for final submission)
def train_xgb_on_full_data(X, y, params=PARAMS, num_boost_round=922):
    """
    Trains an XGBoost model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        params (dict): XGBoost parameters.
        num_boost_round (int): Number of boosting rounds.
    
    Returns:
        bst (xgb.Booster): Trained XGBoost model.
    """
    # Set up DMatrix for XGBoost
    dtrain = xgb.DMatrix(X, label=y)

    # Train the final model on the full dataset
    bst = xgb.train(
        params=params, 
        dtrain=dtrain, 
        num_boost_round=num_boost_round, 
        verbose_eval=False
    )

    print("XGBoost model trained on full dataset.")
    return bst

# Main block for testing XGBoost model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Train XGBoost model on training data (for validation)
    bst, mae_train, mae_valid, xgb_cv_mae = train_xgb_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train XGBoost model on full dataset (for final submission)
    bst_full = train_xgb_on_full_data(X_transformed, y)