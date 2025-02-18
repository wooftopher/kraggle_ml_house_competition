from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
import sys
import os

best_params_lgbm = {
    'learning_rate': 0.09,
    'max_depth': 6,
    'min_child_samples': 25,
    'n_estimators': 140,
    'num_leaves': 29,
    'subsample': 0.7
}

# Function to train LightGBM model on training data (for validation)
def train_lgbm_on_training_data(X_train, y_train, X_valid, y_valid, best_params=best_params_lgbm, random_state=42, verbose=-1):
    # Set default best_params if none are provided
    if best_params is None:
        best_params = {
            'learning_rate': 0.09,
            'max_depth': 6,
            'min_child_samples': 25,
            'n_estimators': 140,
            'num_leaves': 29,
            'subsample': 0.7
        }

    # Initialize and train the model with best_params
    lgbm_model = LGBMRegressor(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_child_samples=best_params['min_child_samples'],
        n_estimators=best_params['n_estimators'],
        num_leaves=best_params['num_leaves'],
        subsample=best_params['subsample'],
        random_state=random_state,
        verbose=verbose
    )
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
    # Initialize and train the model
    lgbm_model = LGBMRegressor(n_estimators=n_estimators, random_state=random_state, verbose=verbose)
    lgbm_model.fit(X, y)

    print("LightGBM model trained on full dataset.")
    return lgbm_model

def grid_search_lgbm(X_train, y_train):
    # Suppress LightGBM warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    # Define a very small and balanced parameter grid
    param_grid = {
        'n_estimators': [140, 150],  # Slight variation around n_estimators
        'learning_rate': [0.09, 0.1],  # Slight variation around learning_rate
        'num_leaves': [29, 30],  # Small variation around num_leaves
        'max_depth': [5, 6],  # Small variation around max_depth
        'min_child_samples': [20, 25],  # Small variation around min_child_samples
        'subsample': [0.7, 0.75]  # Small variation around subsample
    }

    # Initialize LGBM model with fixed parameters
    lgbm = LGBMRegressor(random_state=42, verbose=-1)  # Disable LightGBM logs

    # Custom function to print MAE for each combination
    def print_mae_for_combinations(grid_search):
        results = grid_search.cv_results_
        for i in range(len(results['params'])):
            print(f"Combination {i + 1}:")
            print(f"  Parameters: {results['params'][i]}")
            print(f"  Mean CV MAE: {-results['mean_test_score'][i]:.4f}")  # Convert negative MAE back to positive
            print()

    # Initialize GridSearchCV with 5-fold CV
    grid_search = GridSearchCV(
        lgbm, 
        param_grid, 
        cv=3,  # Use 5-fold cross-validation
        scoring='neg_mean_absolute_error',  # Use MAE for scoring
        verbose=1,  # Print progress (1 for moderate verbosity, 0 for silent)
        n_jobs=4  # Use 4 CPU cores (adjust based on your system)
    )
    grid_search.fit(X_train, y_train)

    # Print MAE for each combination
    print_mae_for_combinations(grid_search)

    # Print best parameters and best MAE
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best MAE: {-grid_search.best_score_:.4f}")  # Convert negative MAE back to positive

    return grid_search.best_params_

# Main block for testing LightGBM model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('../input/train.csv')

    # best_params = grid_search_lgbm(X_train_transformed, y_train)
    # sys.exit()

    # Train LightGBM model on training data (for validation)
    lgbm_model, mae_train, mae_valid, cv_mae = train_lgbm_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # Train LightGBM model on full dataset (for final submission)
    # lgbm_model_full = train_lgbm_on_full_data(X_transformed, y)