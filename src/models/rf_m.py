from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# Function to train Random Forest model on training data (for validation)
def train_rf_on_training_data(X_train, y_train, X_valid, y_valid, n_estimators=100, random_state=42):
    """
    Trains a Random Forest model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        rf_model (RandomForestRegressor): Trained Random Forest model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Initialize and train the model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Make predictions
    rf_predictions_train = rf_model.predict(X_train)
    rf_predictions_valid = rf_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, rf_predictions_train)
    mae_valid = mean_absolute_error(y_valid, rf_predictions_valid)

    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print the MAE for both sets
    print(f"RandomForest Model - Training MAE: {mae_train}")
    print(f"RandomForest Model - Valid MAE: {mae_valid}")
    print(f"RandomForest Model - CV MAE: {cv_mae}\n")

    return rf_model, mae_train, mae_valid, cv_mae

# Function to train Random Forest model on full dataset (for final submission)
def train_rf_on_full_data(X, y, n_estimators=100, random_state=42):
    """
    Trains a Random Forest model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        rf_model (RandomForestRegressor): Trained Random Forest model.
    """
    # Initialize and train the model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X, y)

    print("RandomForest model trained on full dataset.")
    return rf_model

# Main block for testing Random Forest model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Train Random Forest model on training data (for validation)
    rf_model, mae_train, mae_valid, cv_mae = train_rf_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train Random Forest model on full dataset (for final submission)
    rf_model_full = train_rf_on_full_data(X_transformed, y)