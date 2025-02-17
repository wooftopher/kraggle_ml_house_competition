from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# Function to train LightGBM model on training data (for validation)
def train_lgbm_on_training_data(X_train, y_train, X_valid, y_valid, n_estimators=100, random_state=42, verbose=-1):
    """
    Trains a LightGBM model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed for reproducibility.
        verbose (int): Controls verbosity (-1 for silent).
    
    Returns:
        lgbm_model (LGBMRegressor): Trained LightGBM model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Initialize and train the model
    lgbm_model = LGBMRegressor(n_estimators=n_estimators, random_state=random_state, verbose=verbose)
    lgbm_model.fit(X_train, y_train)

    # Make predictions
    lgbm_predictions_train = lgbm_model.predict(X_train)
    lgbm_predictions_valid = lgbm_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, lgbm_predictions_train)
    mae_valid = mean_absolute_error(y_valid, lgbm_predictions_valid)

    # Perform cross-validation
    cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print the MAE for both sets
    print(f"LightGBM Model - Training MAE: {mae_train}")
    print(f"LightGBM Model - Valid MAE: {mae_valid}")
    print(f"LightGBM Model - CV MAE: {cv_mae}\n")

    return lgbm_model, mae_train, mae_valid, cv_mae

# Function to train LightGBM model on full dataset (for final submission)
def train_lgbm_on_full_data(X, y, n_estimators=100, random_state=42, verbose=-1):
    """
    Trains a LightGBM model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        n_estimators (int): Number of boosting rounds.
        random_state (int): Random seed for reproducibility.
        verbose (int): Controls verbosity (-1 for silent).
    
    Returns:
        lgbm_model (LGBMRegressor): Trained LightGBM model.
    """
    # Initialize and train the model
    lgbm_model = LGBMRegressor(n_estimators=n_estimators, random_state=random_state, verbose=verbose)
    lgbm_model.fit(X, y)

    print("LightGBM model trained on full dataset.")
    return lgbm_model

# Main block for testing LightGBM model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Train LightGBM model on training data (for validation)
    lgbm_model, mae_train, mae_valid, cv_mae = train_lgbm_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train LightGBM model on full dataset (for final submission)
    lgbm_model_full = train_lgbm_on_full_data(X_transformed, y)