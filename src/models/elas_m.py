from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# Function to train ElasticNet model on training data (for validation)
def train_elastic_on_training_data(X_train, y_train, X_valid, y_valid, alpha=1.0, l1_ratio=0.5):
    """
    Trains an ElasticNet model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        alpha (float): Constant that multiplies the penalty terms.
        l1_ratio (float): Mixing parameter for L1/L2 regularization.
    
    Returns:
        elastic_model (ElasticNet): Trained ElasticNet model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Initialize and train the model
    elastic_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_model.fit(X_train, y_train)

    # Make predictions
    elastic_predictions_train = elastic_model.predict(X_train)
    elastic_predictions_valid = elastic_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, elastic_predictions_train)
    mae_valid = mean_absolute_error(y_valid, elastic_predictions_valid)

    # Perform cross-validation
    cv_scores = cross_val_score(elastic_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print the MAE for both sets
    print(f"ElasticNet Model - Training MAE: {mae_train}")
    print(f"ElasticNet Model - Valid MAE: {mae_valid}")
    print(f"ElasticNet Model - CV MAE: {cv_mae}\n")

    return elastic_model, mae_train, mae_valid, cv_mae

# Function to train ElasticNet model on full dataset (for final submission)
def train_elastic_on_full_data(X, y, alpha=1.0, l1_ratio=0.5):
    """
    Trains an ElasticNet model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        alpha (float): Constant that multiplies the penalty terms.
        l1_ratio (float): Mixing parameter for L1/L2 regularization.
    
    Returns:
        elastic_model (ElasticNet): Trained ElasticNet model.
    """
    # Initialize and train the model
    elastic_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_model.fit(X, y)

    print("ElasticNet model trained on full dataset.")
    return elastic_model

# Main block for testing ElasticNet model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Train ElasticNet model on training data (for validation)
    elastic_model, mae_train, mae_valid, cv_mae = train_elastic_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train ElasticNet model on full dataset (for final submission)
    elastic_model_full = train_elastic_on_full_data(X_transformed, y)