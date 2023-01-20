import argparse
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def create_pipeline():
    """
    This function creates the pipeline for the random forest classifier
    :return: pipeline object
    :rtype: sklearn.pipeline.Pipeline
    """
    try:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])
        return pipe
    except Exception as e:
        print(f"An error occured while creating the pipeline: {e}")
        raise e


def tune_hyperparameters(pipeline, X, y):
    """
    This function performs hyperparameter tuning on the given pipeline and data
    :param pipeline: pipeline object to tune hyperparameters for
    :type pipeline: sklearn.pipeline.Pipeline
    :param X: input feature variables
    :type X: array-like
    :param y: target variable
    :type y: array-like
    :return: best parameters of the model, predicted values, accuracy, confusion matrix
    :rtype: tuple
    """
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    try:
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        y_pred = grid_search.predict(X)
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        return best_params, y_pred, accuracy, cm
    except Exception as e:
       raise ValueError("Error occured while tuning hyperparameters: ", e)


def train_model(X, y):
    """
    This function trains the model and returns the best parameters, predicted values, accuracy and confusion matrix
    :param X: input feature variables
    :type X: array-like
    :param y: target variable
    :type y: array-like
    :return: best parameters of the model, predicted values, accuracy, confusion matrix
    :rtype: tuple
    """
    try:
        assert isinstance(X, (np.ndarray))
        assert isinstance(y, (np.ndarray))
    except AssertionError:
        raise ValueError("X and y must be a numpy array")
    try:
        pipeline = create_pipeline()
        best_params, y_pred, accuracy, cm = tune_hyperparameters(pipeline, X, y)
        return best_params, y_pred, accuracy, cm
    except Exception as e:
       raise ValueError("Error occured while creating pipeline: ", e)


def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """
    This function takes a pipeline, training and test data, and evaluates the model's performance
    :param pipeline: trained pipeline
    :type pipeline: sklearn.pipeline.Pipeline
    :param X_train: training feature variables
    :type X_train: array-like
    :param X_test: test feature variables
    :type X_test: array-like
    :param y_train: training target variable
    :type y_train: array-like
    :param y_test: test target variable
    :type y_test: array-like
    :return: None
    """
    try:
        assert isinstance(pipeline, Pipeline)
    except AssertionError:
        print("pipeline is not a pipeline object")
        return
    try:
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        return
    try:
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
    except Exception as e:
        print(f"Accuracy score calculation failed with error: {e}")
        return
    try:
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
    except Exception as e:
        print(f"Confusion matrix calculation failed with error: {e}")
        return


def save_metrics(filepath, accuracy, confusion_matrix):
    """
    This function saves the accuracy and confusion matrix to a txt file
    :param filepath: filepath to save metrics to
    :type filepath: str
    :param accuracy: accuracy score of model
    :type accuracy: float
    :param confusion_matrix: confusion matrix of model
    :type confusion_matrix: array-like
    :return: None
    """
    try:
        with open(filepath, 'w') as f:
            f.write(f'Accuracy: {accuracy}\n')
            f.write(f'Confusion Matrix: {confusion_matrix}\n')
    except Exception as e:
        print(f'Error saving metrics to {filepath}: {e}')


def challenge_2():
    parser = argparse.ArgumentParser(description='Challenge 2')
    parser.add_argument('filepath', type=str, help='filepath to save metrics to')
    args = parser.parse_args()
    filepath = args.filepath

    # Load the iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    print("Iris dataset loaded.")

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print("Data split.")

    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline()
    print("Pipeline created.")

    # Hyperparameter tuning
    print("Starting hyperparameter tuning...")
    best_params, y_pred, accuracy, cm = tune_hyperparameters(pipeline, X_train, y_train)
    print("Hyperparameter tuning complete.")

    # Evaluate model
    print("Training model...")
    pipeline.set_params(**best_params)
    pipeline.fit(X,y)
    print("Model trained.")

    print("Evaluating model...")
    evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    print("Model evaluated.")

    # Save metrics
    print("Saving Metrics")
    save_metrics(filepath, accuracy, cm)
    print("Metrics Saved")

if __name__ == '__main__':
    challenge_2()

