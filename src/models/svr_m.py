# Support Vector Regression (SVR):
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import sys
import os


# Function to train SVR model on training data (for validation)
def train_svr_on_training_data(X_train, y_train, X_valid, y_valid, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Trains an SVR model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
        kernel (str): Kernel type ('rbf', 'linear', 'poly', etc.).
        C (float): Regularization parameter.
        epsilon (float): Epsilon in the epsilon-SVR model.
    
    Returns:
        svr_model (SVR): Trained SVR model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Standardize features (SVR is sensitive to feature scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Initialize and train the model
    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr_model.fit(X_train_scaled, y_train)

    # Make predictions
    svr_predictions_train = svr_model.predict(X_train_scaled)
    svr_predictions_valid = svr_model.predict(X_valid_scaled)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, svr_predictions_train)
    mae_valid = mean_absolute_error(y_valid, svr_predictions_valid)

    # Perform cross-validation
    cv_scores = cross_val_score(svr_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print the MAE for both sets
    print(f"SVR Model - Training MAE: {mae_train}")
    print(f"SVR Model - Valid MAE: {mae_valid}")
    print(f"SVR Model - CV MAE: {cv_mae}\n")

    return svr_model, mae_train, mae_valid, cv_mae, scaler

# Function to train SVR model on full dataset (for final submission)
def train_svr_on_full_data(X, y, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Trains an SVR model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
        kernel (str): Kernel type ('rbf', 'linear', 'poly', etc.).
        C (float): Regularization parameter.
        epsilon (float): Epsilon in the epsilon-SVR model.
    
    Returns:
        svr_model (SVR): Trained SVR model.
        scaler (StandardScaler): Fitted scaler for preprocessing.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and train the model
    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr_model.fit(X_scaled, y)

    print("SVR model trained on full dataset.")
    return svr_model, scaler

# Main block for testing SVR model
if __name__ == "__main__":

    # Add the parent directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('../input/train.csv')

    # Train SVR model on training data (for validation)
    svr_model, mae_train, mae_valid, cv_mae, scaler = train_svr_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train SVR model on full dataset (for final submission)
    svr_model_full, scaler_full = train_svr_on_full_data(X_transformed, y)