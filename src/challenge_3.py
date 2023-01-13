import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import argparse
import os

def check_file_type(filepath):
    """
    Check if the file extension is .txt
    Args:
        filepath (str): path to the file
    """
    if not filepath.endswith('.txt'):
        raise ValueError("Expected file type as .txt")

def load_wine_data():
    """
    Load the Wine Quality Data Set from a csv file
    Returns:
        X (pandas DataFrame): features of the dataset
        y (pandas Series): labels of the dataset
    """
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=';')
    X = data.drop("quality", axis=1)
    y = data["quality"]
    return X, y

def build_pipeline():
    """
    Build a pipeline that scales the data and trains a RandomForestRegressor
    Returns:
        pipe (Pipeline object): pipeline object
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', RandomForestRegressor(random_state=42))
    ])
    return pipe

def hyper_tune_pipeline(pipe, X, y):
    """
    Hyper-tune the pipeline using GridSearchCV
    Args:
        pipe (Pipeline object): pipeline to be hyper-tuned
        X (pandas DataFrame): features of the dataset
        y (pandas Series): labels of the dataset
    Returns:
        reg (Pipeline object): best estimator from the hyper-tuning process
    """
    # Define the parameter grid
    param_grid = {
        'reg__n_estimators': [10, 20, 30],
        'reg__max_depth': [None, 2, 3],
        'reg__min_samples_split': [2, 5, 10]
    }

    # Create the GridSearchCV object
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid.fit(X, y)

    return grid.best_estimator_

def evaluate_pipeline(reg, X, y):
    """
    Evaluate the pipeline using mean squared error and mean absolute error
    Args:
        reg (Pipeline object): trained pipeline
        X (pandas DataFrame): features of the dataset
        y (pandas Series): labels of the dataset
    Returns:
        mse (float): mean squared error of the pipeline
        mae (float): mean absolute error of the pipeline
    """
    y_pred = reg.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    return mse, mae

def save_metrics(mse, mae, filepath):
    """
    Save the mean squared error and mean absolute error to a local file
    Args:
        mse (float): mean squared error of the pipeline
        mae (float): mean absolute error of the pipeline
        filepath (str): path to save the metrics file
    """
    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, "w") as f:
        f.write("Mean Squared Error: {:.4f}\n".format(mse))
        f.write("Mean Absolute Error: {:.4f}\n".format(mae))


def challenge_3():
    """
        Main function that runs the pipeline and saves the metrics
        Args:
            filepath (str): path to save the metrics file
        """
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("file_path", help="Path where you want to save CSV file")
    # Parse the arguments
    args = parser.parse_args()
    check_file_type(args.file_path)
    print("Loading Data")
    X, y = load_wine_data()
    print("Data Loaded")
    print("Building Pipeline")
    pipe = build_pipeline()
    print("Pipeline built")
    print("Hypertuning...")
    reg = hyper_tune_pipeline(pipe, X, y)
    print('Hypertuning complete')
    print("Evaluating best model")
    mse, mae = evaluate_pipeline(reg, X, y)
    print("Saving {}".format(args.file_path.split('/')[-1]))
    save_metrics(mse, mae, args.file_path)
    print(f"{args.file_path.split('/')[-1]} is saved at {args.file_path}")
if __name__ == "__main__":
    challenge_3()

 