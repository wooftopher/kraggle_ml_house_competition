# Support Vector Regression (SVR):
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
import os

# Best hyperparameters structure for SVR
best_params_svr = {
    'C': 100.0,
    'epsilon': 0.2,
    'kernel': 'linear'
}

# Function to train SVR model on training data (for validation)
def train_svr_on_training_data(X_train, y_train, X_valid, y_valid, params=best_params_svr):
    # Standardize features (SVR is sensitive to feature scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Initialize and train the model using the provided or default parameters
    svr_model = SVR(kernel=params['kernel'], C=params['C'], epsilon=params['epsilon'])
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

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and train the model
    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr_model.fit(X_scaled, y)

    print("SVR model trained on full dataset.")
    return svr_model, scaler

# Function to perform Grid Search for SVR model hyperparameter tuning
def grid_search_svr(X_train, y_train, X_valid, y_valid):
    # Standardize features (SVR is sensitive to feature scales)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1.0, 10.0, 100.0],
        'epsilon': [0.01, 0.1, 0.2],
    }

    # Initialize the SVR model
    svr = SVR()

    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=3)

    # Fit the grid search to the training data
    print("Starting grid search...")
    grid_search.fit(X_train_scaled, y_train)

    # Print the results of each parameter combination
    print("\nGrid Search Results:")
    print("Best Parameters: ", grid_search.best_params_)
    print(f"Best Validation MAE: {-grid_search.best_score_}\n")

    # Make predictions using the best model
    svr_predictions_valid = grid_search.best_estimator_.predict(X_valid_scaled)

    # Compute the validation MAE
    best_mae_valid = mean_absolute_error(y_valid, svr_predictions_valid)

    # Print the final validation MAE
    print(f"Validation MAE of the best SVR model: {best_mae_valid}\n")

    return grid_search.best_estimator_, grid_search.best_params_, best_mae_valid

# Main block for testing SVR model
if __name__ == "__main__":

    # Add the parent directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Example usage (replace with your actual data)
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('../input/train.csv')

    # best_svr_model, best_params, best_mae_valid = grid_search_svr(
    #     X_train_transformed, y_train, X_valid_transformed, y_valid
    # )
    # sys.exit()
    # Train SVR model on training data (for validation)
    svr_model, mae_train, mae_valid, cv_mae, scaler = train_svr_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid, best_params_svr
    )

    # Train SVR model on full dataset (for final submission)
    # svr_model_full, scaler_full = train_svr_on_full_data(X_transformed, y)