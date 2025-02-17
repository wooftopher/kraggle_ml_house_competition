from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# Function to train CatBoost model on training data (for validation)
def train_catboost_on_training_data(X_train, y_train, X_valid, y_valid, iterations=500, depth=6, learning_rate=0.1, verbose=0, random_state=42):
    """
    Trains a CatBoost model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        iterations (int): Number of boosting iterations.
        depth (int): Depth of the trees.
        learning_rate (float): Learning rate.
        verbose (int): Controls verbosity (0 for silent).
        random_state (int): Random seed for reproducibility.
    
    Returns:
        catboost_model (CatBoostRegressor): Trained CatBoost model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Initialize and train the model
    catboost_model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        verbose=verbose,
        random_state=random_state
    )
    catboost_model.fit(X_train, y_train)

    # Make predictions
    catboost_predictions_train = catboost_model.predict(X_train)
    catboost_predictions_valid = catboost_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, catboost_predictions_train)
    mae_valid = mean_absolute_error(y_valid, catboost_predictions_valid)

    # Perform cross-validation
    cv_scores = cross_val_score(catboost_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print the MAE for both sets
    print(f"CatBoost Model - Training MAE: {mae_train}")
    print(f"CatBoost Model - Valid MAE: {mae_valid}")
    print(f"CatBoost Model - CV MAE: {cv_mae}\n")

    return catboost_model, mae_train, mae_valid, cv_mae

# Function to train CatBoost model on full dataset (for final submission)
def train_catboost_on_full_data(X, y, iterations=500, depth=6, learning_rate=0.1, verbose=0, random_state=42):
    """
    Trains a CatBoost model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        iterations (int): Number of boosting iterations.
        depth (int): Depth of the trees.
        learning_rate (float): Learning rate.
        verbose (int): Controls verbosity (0 for silent).
        random_state (int): Random seed for reproducibility.
    
    Returns:
        catboost_model (CatBoostRegressor): Trained CatBoost model.
    """
    # Initialize and train the model
    catboost_model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        verbose=verbose,
        random_state=random_state
    )
    catboost_model.fit(X, y)

    print("CatBoost model trained on full dataset.")
    return catboost_model

# Main block for testing CatBoost model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Train CatBoost model on training data (for validation)
    catboost_model, mae_train, mae_valid, cv_mae = train_catboost_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train CatBoost model on full dataset (for final submission)
    catboost_model_full = train_catboost_on_full_data(X_transformed, y)