from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# Function to train GBM model on training data (for validation)
def train_gbm_on_training_data(X_train, y_train, X_valid, y_valid, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """
    Trains a Gradient Boosting model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate.
        max_depth (int): Maximum depth of the trees.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        gbm_model (GradientBoostingRegressor): Trained GBM model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Initialize and train the model
    gbm_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    gbm_model.fit(X_train, y_train)

    # Make predictions
    gbm_predictions_train = gbm_model.predict(X_train)
    gbm_predictions_valid = gbm_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, gbm_predictions_train)
    mae_valid = mean_absolute_error(y_valid, gbm_predictions_valid)

    # Perform cross-validation
    cv_scores = cross_val_score(gbm_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print the MAE for both sets
    print(f"GBM Model - Training MAE: {mae_train}")
    print(f"GBM Model - Valid MAE: {mae_valid}")
    print(f"GBM Model - CV MAE: {cv_mae}\n")

    return gbm_model, mae_train, mae_valid, cv_mae

# Function to train GBM model on full dataset (for final submission)
def train_gbm_on_full_data(X, y, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """
    Trains a Gradient Boosting model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate.
        max_depth (int): Maximum depth of the trees.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        gbm_model (GradientBoostingRegressor): Trained GBM model.
    """
    # Initialize and train the model
    gbm_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    gbm_model.fit(X, y)

    print("GBM model trained on full dataset.")
    return gbm_model

# Main block for testing GBM model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Train GBM model on training data (for validation)
    gbm_model, mae_train, mae_valid, cv_mae = train_gbm_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train GBM model on full dataset (for final submission)
    gbm_model_full = train_gbm_on_full_data(X_transformed, y)