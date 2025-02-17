import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load the data
    home_data = pd.read_csv(file_path)

    # Extract target variable
    y = home_data['SalePrice']

    # Identify numerical and categorical features dynamically
    num_cols = home_data.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice').tolist()
    cat_cols = home_data.select_dtypes(include=['object']).columns.tolist()

    # Define feature matrix
    X = home_data[num_cols + cat_cols]

    # Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Fill missing categorical values
                ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
            ("num", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numeric values with mean
                ('scaler', StandardScaler())  
            ]), num_cols)
        ],
        verbose_feature_names_out=False  # Ensures clean feature names
    )

    # Split data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply transformation and restore feature names
    X_train_transformed, X_valid_transformed = transform_data(preprocessor, X_train, X_valid)

    # Replace whitespaces with underscores in the feature names before training
    X_train_transformed.columns = X_train_transformed.columns.str.replace(' ', '_')
    X_valid_transformed.columns = X_valid_transformed.columns.str.replace(' ', '_')

    # Apply same transformation for full dataset
    X_transformed, _ = transform_data(preprocessor, X, X)
    X_transformed.columns = X_transformed.columns.str.replace(' ', '_')

    return X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor

# Function to apply preprocessing and retain feature names
def transform_data(preprocessor, X_train, X_test):
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert back to DataFrame with correct column names
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    return X_train_transformed, X_test_transformed

# Main block for testing data processing
if __name__ == "__main__":
    # Test the data loading and preprocessing
    X_train_transformed, X_valid_transformed, y_train, y_valid, X_transformed, y, preprocessor = load_and_preprocess_data('../input/train.csv')
    print("X_train_transformed shape:", X_train_transformed.shape)
    print("X_valid_transformed shape:", X_valid_transformed.shape)
    print("X_transformed shape:", X_transformed.shape)