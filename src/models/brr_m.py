# Bayesian Ridge Regression:
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sys
import os

# Best parameters structure
best_params = {
    'alpha_1': 1e-10,
    'alpha_2': 100.0,
    'fit_intercept': True,
    'lambda_1': 0.01,
    'lambda_2': 0.1
}

def train_bayesian_ridge_on_training_data(X_train, y_train, X_valid, y_valid):
    # Use best_params directly inside the function
    bayesianridge_model = BayesianRidge(
        alpha_1=best_params['alpha_1'],
        alpha_2=best_params['alpha_2'],
        fit_intercept=best_params['fit_intercept'],
        lambda_1=best_params['lambda_1'],
        lambda_2=best_params['lambda_2']
    )
    bayesianridge_model.fit(X_train, y_train)

    # Make predictions
    train_preds = bayesianridge_model.predict(X_train)
    valid_preds = bayesianridge_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, train_preds)
    mae_valid = mean_absolute_error(y_valid, valid_preds)

    # Perform cross-validation
    cv_scores = cross_val_score(bayesianridge_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Print results
    print(f"Bayesian Ridge - Training MAE: {mae_train}")
    print(f"Bayesian Ridge - Validation MAE: {mae_valid}")
    print(f"Bayesian Ridge - CV MAE: {cv_mae}\n")

    return bayesianridge_model, mae_train, mae_valid, cv_mae

def train_bayesian_ridge_on_full_data(X, y, params):
    # Initialize and train the model with passed parameters
    bayesianridge_model = BayesianRidge(
        alpha_1=best_params['alpha_1'],
        alpha_2=best_params['alpha_2'],
        fit_intercept=best_params['fit_intercept'],
        lambda_1=best_params['lambda_1'],
        lambda_2=best_params['lambda_2']
    )
    bayesianridge_model.fit(X, y)

    print("Bayesian Ridge model trained on full dataset.")
    return bayesianridge_model

def optimize_bayesian_ridge(X_train, y_train):
    # Define the model
    model = BayesianRidge()

    # Define the expanded parameter grid
    param_grid = {
        'alpha_1': [1e-10, 1e-08, 1e-06, 1e-04, 1e-02],
        'alpha_2': [0.01, 0.1, 1.0, 10.0, 100.0],
        'lambda_1': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'lambda_2': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'fit_intercept': [True, False]
    }

    # Define the grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    # Print the best parameters and score
    print("Best Parameters:", best_params)
    print("Best MAE (negative):", best_score)

    return best_params, best_score

if __name__ == "__main__":
    # Add the parent directory to the Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('./input/train.csv')

    # Call the training function with best_params
    bayesianridge_model, mae_train, mae_valid, cv_mae = train_bayesian_ridge_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid, best_params
    )

    # Train Bayesian Ridge model on full dataset (for final submission)
    bayesianridge_model_full = train_bayesian_ridge_on_full_data(X_transformed, y, best_params)