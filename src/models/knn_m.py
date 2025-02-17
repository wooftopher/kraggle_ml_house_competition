# K-Nearest Neighbors (KNN):
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

def train_knn_on_training_data(X_train, y_train, X_valid, y_valid, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors (KNN) model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        n_neighbors (int): Number of neighbors to use.
    
    Returns:
        knn_model (KNeighborsRegressor): Trained KNN model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Initialize and train the model
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)

    # Make predictions
    knn_predictions_train = knn_model.predict(X_train)
    knn_predictions_valid = knn_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, knn_predictions_train)
    mae_valid = mean_absolute_error(y_valid, knn_predictions_valid)

    # Perform cross-validation
    cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print the MAE for both sets
    print(f"KNN Model - Training MAE: {mae_train}")
    print(f"KNN Model - Valid MAE: {mae_valid}")
    print(f"KNN Model - CV MAE: {cv_mae}\n")

    return knn_model, mae_train, mae_valid, cv_mae

def train_knn_on_full_data(X, y, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors (KNN) model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        n_neighbors (int): Number of neighbors to use.
    
    Returns:
        knn_model (KNeighborsRegressor): Trained KNN model.
    """
    # Initialize and train the model
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X, y)

    print("KNN model trained on full dataset.")
    return knn_model

if __name__ == "__main__":
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Train KNN model on training data (for validation)
    knn_model, mae_train, mae_valid, cv_mae = train_knn_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train KNN model on full dataset (for final submission)
    knn_model_full = train_knn_on_full_data(X_transformed, y)
