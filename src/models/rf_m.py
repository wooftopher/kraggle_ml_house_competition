from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
import numpy as np
import sys
import os

# Best hyperparameters structure
best_params_rf = {
    'n_estimators': 200,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 20,
    'bootstrap': False
}

# Function to train Random Forest model on training data (for validation)
def train_rf_on_training_data(X_train, y_train, X_valid, y_valid, params=best_params_rf):
    # Initialize and train the model with the provided parameters
    rf_model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        max_depth=params['max_depth'],
        bootstrap=params['bootstrap'],
        random_state=42,
        n_jobs=4
    )
    
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

def train_rf_on_full_data(X, y, best_params=best_params_rf, random_state=42):
     # Initialize and train the RandomForest model with best_params_rf
    rf_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        max_depth=best_params['max_depth'],
        bootstrap=best_params['bootstrap'],
        random_state=random_state
    )
    rf_model.fit(X, y)

    print("RandomForest model trained on full dataset.")
    return rf_model

def optimize_rf_parameters(X_train, y_train, X_valid, y_valid, n_iter=50, cv=5, random_state=42):
    # Define the parameter grid
    param_dist = {
        'n_estimators': [150, 200, 250],  # Test fewer/more trees
        'max_depth': [15, 20, 25],  # Control overfitting
        'min_samples_split': [4, 5, 6],  # Slight variation in splits
        'min_samples_leaf': [1, 2],  # Slight variation in leaves
        'max_features': ['sqrt', 'log2'],  # Feature selection strategy
        'bootstrap': [False]  # Keep consistent
    }

    # Initialize the Random Forest model
    rf_model = RandomForestRegressor(random_state=random_state)

    # Define the scorer (MAE)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=cv,
        random_state=random_state,
        verbose=1,
        n_jobs=4  # Use all available CPU cores
    )

    # Perform the search
    random_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Make predictions with the best model
    train_preds = best_model.predict(X_train)
    valid_preds = best_model.predict(X_valid)

    # Compute MAE
    mae_train = mean_absolute_error(y_train, train_preds)
    mae_valid = mean_absolute_error(y_valid, valid_preds)

    # Print results
    print(f"Best Parameters: {best_params}")
    print(f"Training MAE with Best Model: {mae_train}")
    print(f"Validation MAE with Best Model: {mae_valid}")

    return best_model, best_params, mae_train, mae_valid

def evaluate_feature_importance(rf_model, X_train, y_train, X_valid, y_valid, X_transformed, y):
    # Get feature importances from the trained model
    feature_importances = rf_model.feature_importances_

    # Sort features by importance
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X_train.columns[sorted_idx]

    # Print the feature importance
    print("Feature Importance:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_idx]):
        print(f"{feature}: {importance}")

    # Select the top 80% most important features
    threshold = 0.6
    num_top_features = int(len(sorted_features) * threshold)
    selected_features = sorted_features[:num_top_features]

    # Create new datasets using only the top features
    X_train_selected = X_train[selected_features]
    X_valid_selected = X_valid[selected_features]
    X_full_selected = X_transformed[selected_features]

    # Re-train the Random Forest model using only the important features
    rf_model.fit(X_train_selected, y_train)

    # Make predictions with the new model
    rf_predictions_train = rf_model.predict(X_train_selected)
    rf_predictions_valid = rf_model.predict(X_valid_selected)

    # Compute MAE for the new model
    mae_train_new = mean_absolute_error(y_train, rf_predictions_train)
    mae_valid_new = mean_absolute_error(y_valid, rf_predictions_valid)

    # Perform cross-validation for the new model
    cv_scores_new = cross_val_score(rf_model, X_train_selected, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae_new = -cv_scores_new.mean()

    # Print the results for comparison
    print(f"New Model (using top features) - Training MAE: {mae_train_new}")
    print(f"New Model (using top features) - Valid MAE: {mae_valid_new}")
    print(f"New Model (using top features) - CV MAE: {cv_mae_new}\n")

    # Compare the results of the old model with the new one
    print("Comparison with original model:")
    print(f"Original Model - Training MAE: {mae_train}")
    print(f"Original Model - Valid MAE: {mae_valid}")
    print(f"Original Model - CV MAE: {cv_mae}\n")

    # Check if the new model is better
    if mae_valid_new < mae_valid and cv_mae_new < cv_mae:
        print("The new model with selected features is better!")
    else:
        print("The new model did not perform better.")

    # Return the trained model, new MAEs, and the comparison result
    return rf_model, mae_train_new, mae_valid_new, cv_mae_new

# Main block for testing Random Forest model
if __name__ == "__main__":
    # Example usage (replace with your actual data)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data import load_and_preprocess_data

    # Load and preprocess data
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('../input/train.csv')

    # best_rf_model, best_params, mae_train, mae_valid = optimize_rf_parameters(
    #         X_train_transformed, y_train, X_valid_transformed, y_valid
    #     )
    # sys.exit()
    # Train Random Forest model on training data (for validation)
    rf_model, mae_train, mae_valid, cv_mae = train_rf_on_training_data(
        X_train_transformed, y_train, X_valid_transformed, y_valid
    )

    # # Now evaluate feature importance and re-train using selected features
    # rf_model, mae_train_new, mae_valid_new, cv_mae_new = evaluate_feature_importance(
    #     rf_model, X_train_transformed, y_train, X_valid_transformed, y_valid, X_transformed, y
    # )

    # Train Random Forest model on full dataset (for final submission)
    # rf_model_full = train_rf_on_full_data(X_transformed, y)