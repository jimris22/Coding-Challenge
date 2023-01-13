from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def load_iris_data():
    """
    Load the Iris dataset
    Returns:
        X (numpy array): features of the Iris dataset
        y (numpy array): labels of the Iris dataset
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def build_pipeline():
    """
    Build a pipeline with scaling, PCA, and RandomForestClassifier
    Returns:
        pipe (Pipeline object): pipeline with scaling, PCA, and RandomForestClassifier
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3)),('clf', RandomForestClassifier(random_state=42))])
    return pipe

def hyper_tune_pipeline(pipe, X, y):
    """
    Hyper-tune the pipeline using GridSearchCV
    Args:
        pipe (Pipeline object): pipeline to be hyper-tuned
        X (numpy array): features of the dataset
        y (numpy array): labels of the dataset
    Returns:
        clf (Pipeline object): best estimator from the hyper-tuning process
    """
    # Define the parameter grid
    param_grid = {
        'clf__n_estimators': [10,20,30],
        'clf__max_depth': [None, 2],
        'clf__min_samples_split': [2, 5, 10]
    }

    # Create the GridSearchCV object
    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid.fit(X, y)

    return grid.best_estimator_

def evaluate_pipeline(clf, X, y):
    """
    Evaluate the pipeline using cross validation and compute the accuracy and confusion matrix
    Args:
        clf (Pipeline object): trained pipeline
        X (numpy array): features of the dataset
        y (numpy array): labels of the dataset
    Returns:
        accuracy (float): accuracy score of the pipeline
        cm (numpy array): confusion matrix of the pipeline
    """
    X_train, X_test,    y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

def save_metrics(accuracy, cm, filepath):
    """
    Save the accuracy and confusion matrix to a local file
    Args:
        accuracy (float): accuracy score of the pipeline
        cm (numpy array): confusion matrix of the pipeline
        filepath (str): path to save the metrics file
    """
    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, "w") as f:
        (f.write("Accuracy: {:.4f}\n".format(accuracy)))
        (f.write("Confusion Matrix:\n{}".format(cm)))

def challenge_2():
    """
    Main function that runs the pipeline and saves the metrics
    Args:
        filepath (str): path to save the metrics file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Filepath to save the metrics to.")
    args = parser.parse_args()

    check_file_type(args.filepath)
    print("Loading Data")
    X, y = load_iris_data()
    print("Data Loaded")
    print("Building Pipeline")
    pipe = build_pipeline()
    print("Pipeline built")
    print("Hypertuning...")
    clf = hyper_tune_pipeline(pipe, X, y)
    print('Hypertuning complete')
    print("Evaluating best model")
    accuracy, cm = evaluate_pipeline(clf, X, y)
    print("Saving {}".format(args.filepath.split('/')[-1]))
    save_metrics(accuracy, cm, args.filepath)
    print(f"{args.filepath.split('/')[-1]} is saved at {args.filepath}")
if __name__ == "__main__":
    challenge_2()


