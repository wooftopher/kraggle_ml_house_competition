# Bayesian Ridge Regression:
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

def train_bayesian_ridge_on_training_data(X_train, y_train, X_valid, y_valid):
    """
    Trains a Bayesian Ridge Regression model on training data and validates it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_valid (pd.DataFrame): Validation features.
        y_valid (pd.Series): Validation target.
    
    Returns:
        bayesianridge_model (BayesianRidge): Trained model.
        mae_train (float): Training MAE.
        mae_valid (float): Validation MAE.
        cv_mae (float): Cross-validation MAE.
    """
    # Initialize and train the model
    bayesianridge_model = BayesianRidge()
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

def train_bayesian_ridge_on_full_data(X, y):
    """
    Trains a Bayesian Ridge Regression model on the full dataset.
    
    Args:
        X (pd.DataFrame): Full dataset features.
        y (pd.Series): Full dataset target.
    
    Returns:
        bayesianridge_model (BayesianRidge): Trained model.
    """
    bayesianridge_model = BayesianRidge()
    bayesianridge_model.fit(X, y)

    print("Bayesian Ridge model trained on full dataset.")
    return bayesianridge_model
