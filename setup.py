import setuptools
import os
import sys

# Getting current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

with open("README.md", "w") as f:
    f.write("# Challenge Project\n\n")
    f.write("This project includes 5 challenges that cover a range of topics such as data manipulation, machine learning and web scraping. Each challenge can be run independently using the command line interface.\n\n")
    f.write("## Installation\n\n")
    f.write("To install the dependencies for this project, you will need to have Python 3.7 or later and pip installed on your machine. Once you have those set up, navigate to the directory where you would like to set up the project and run the following command:\n\n")
    f.write("```bash\n")
    f.write("python setup.py install\n")
    f.write("```\n\n")
    f.write("This will install all the necessary packages including pandas, numpy, scikit-learn, requests, and psutil.\n\n")
    f.write("## Running the challenges\n\n")
    f.write("Once the dependencies are installed, you can run any of the challenges by using the command line interface. Navigate to the root directory of the project and run the following command for the corresponding challenge:\n\n")
    f.write("```bash\n")
    f.write("challenge-1 path_to_csv col1 col2 operation path_to_save_csv\n")
    f.write("challenge-2 path_to_save_txt_file\n")
    f.write("challenge-3 path_to_save_txt_file\n")
    f.write("challenge-4 path_to_save_csv\n")
    f.write("challenge-5 url path_to_save_html\n")
    f.write("```\n\n")
    f.write("## Note\n")
    f.write("- Options for operation in challenge 1 are: add, subtract, multiply, divide.\n")
    f.write("- Make sure to download the data from https://catalog.data.gov/dataset/lottery-mega-millions-winning-numbers-beginning-2002/resource/61eea3d0-6b6d-43a4-ae02-6432e6a4e517 and store the file in data_files directory\n")
    f.write("- Make sure you have internet connection to download the data for challenge 1,2,3\n")
    f.write("- The output for challenge 1,2,3,4,5 will be saved in the provided filepath\n")
    f.write("## Requirements\n\n")
    f.write("- Python 3.7 or later\n")
    f.write("- pip\n")
    f.write("- Internet connection\n\n")
    f.write("## Dependencies\n\n")
    f.write("- pandas\n")
    f.write("- numpy\n")
    f.write("- scikit-learn\n")
    f.write("- requests\n")
    f.write("- psutil\n\n")
    f.write("```\n\n")
    f.write("## Author\n\n")
    f.write("-Jimmy Mirchandani\n")
    f.write("```\n\n")
    f.write("## License\n\n")
    f.write("-This project is licensed under the MIT License.\n")



# Creating src directory
src_dir = os.path.join(current_dir, 'src')
if not os.path.exists(src_dir):
    os.makedirs(src_dir)
# Creating __init__.py file if it doesn't exist
init_file = os.path.join(src_dir, '__init__.py')
if not os.path.exists(init_file):
    open(init_file, "w").close()

def create_file(file_name, code):
    """
    Create a file with the given name and write the given code to it.
    If the file already exists, it will be overwritten.
    """
    # Check if the file already exists
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"{file_name} already exists, removed old file")
    # Create the file and write the code to it
    with open(file_name, "w") as file:
        file.write(code)

challenge_1_code = '''import argparse
import pandas as pd


def check_file_type(filepath):
    """Check if the file extension is .csv

    Args:
        filepath (str): path to the file
    """
    if not filepath.endswith('.csv'):
        raise ValueError("Expected file type as .csv")


def load_data(file_path):
    """Load the CSV file into a pandas data frame

    Args:
        file_path (str): path to the csv file

    Returns:
        pd.DataFrame : pandas dataframe
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError("Error occurred while loading data. Error: {}".format(e))


def print_sorted_columns(df):
    """Print the name of each column in sorted order

    Args:
        df (pd.DataFrame): dataframe
    """
    print("Columns in sorted order:")
    print(sorted(df.columns))


def print_total_rows(df):
    """Print the total number of rows in the data frame

    Args:
        df (pd.DataFrame): dataframe
    """
    print("Total number of rows:")
    print(len(df))


def add_new_column(df, column1, column2, operation):
    """
    Add a new column to the data frame that is computed from two other column values.
    The operation to perform on the columns is determined using a list of allowed operations
    Args:
        df (pd.DataFrame): dataframe
        column1 (str): name of first column
        column2 (str): name of second column
        operation (str): operation specified by the user

    Returns:
        pd.DataFrame : dataframe with new column
    """
    operations = ["sum", "subtract", "multiply", "divide"]
    if operation not in operations:
        raise ValueError("Invalid operation. Options are: sum, subtract, multiply, divide")
    if operation == "sum":
        df["new_column"] = df[column1] + df[column2]
    elif operation == "subtract":
        df["new_column"] = df[column1] - df[column2]
    elif operation == "multiply":
        df["new_column"] = df[column1] * df[column2]
    elif operation == "divide":
        df["new_column"] = df[column1] / df[column2]
    return df


def save_to_csv(df, file_path):
    """Save the resulting data frame to a local CSV file
    Args:
        df (pd.DataFrame): dataframe
        file_path (str): path to save the csv file"""
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise ValueError("Error occurred while saving data to csv. Error: {}".format(e))


def challenge_1():
    # Create a parser object
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("data_file_path", help="Path to the CSV file")
    parser.add_argument("column1", help="Name of column 1")
    parser.add_argument("column2", help="Name of column 2")
    parser.add_argument("operation", help="Operation to perform: sum, subtract, multiply, divide")
    parser.add_argument("file_path", help="Path where to save new csv")

    # Parse the arguments
    args = parser.parse_args()
    check_file_type(args.data_file_path)
    # Load the data
    df = load_data(args.data_file_path)

    # Print sorted columns and total rows
    print_sorted_columns(df)
    print_total_rows(df)

    # Add new column
    df = add_new_column(df, args.column1, args.column2, args.operation)

    check_file_type(args.file_path)
    # Save the data to a CSV file
    save_to_csv(df, args.file_path)


if __name__ == "__main__":
    challenge_1()'''
create_file("src/challenge_1.py",challenge_1_code)

challenge_2_code = '''import argparse
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
            f.write(f'Accuracy: {accuracy}\\n')
            f.write(f'Confusion Matrix: {confusion_matrix}\\n')
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

'''
create_file("src/challenge_2.py",challenge_2_code)

challenge_3_code = '''import argparse
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline():
    """
    This function creates the pipeline object that includes the RandomForestRegressor and sets it up for
    hyperparameter tuning :return: pipeline object :rtype: sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', RandomForestRegressor())
    ])
    return pipeline


def tune_hyperparameters(pipeline, X, y):
    """
    This function performs hyperparameter tuning on the given pipeline and data
    :param pipeline: pipeline object to tune hyperparameters for
    :type pipeline: sklearn.pipeline.Pipeline
    :param X: input feature variables
    :type X: array-like
    :param y: target variable
    :type y: array-like
    :return: best parameters of the model
    :rtype: dict
    """
    try:
        param_grid = {
            'reg__n_estimators': [50, 100, 200],
            'reg__max_depth': [None, 5, 10],
            'reg__min_samples_split': [2, 5, 10],
            'reg__min_samples_leaf': [1, 2, 4],
            'reg__max_features': ['sqrt', 'log2', None]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        return best_params
    except:
        raise ValueError("An error occured while tuning the hyperparameters")


def train_model(X_train, y_train, best_params):
    """
    This function trains the model with the best hyperparameters
    :param X_train: input feature variables for training
    :type X_train: array-like
    :param y_train: target variable for training
    :type y_train: array-like
    :param best_params: best hyperparameters of the model
    :type best_params: dict
    :return: trained model
    """
    try:
        assert X_train.shape[0] == y_train.shape[0], "Number of samples in X and y do not match"
        pipeline = create_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        return pipeline
    except AssertionError as e:
        print(e)
    except Exception as e:
        print("Error occured while training model: ", e)


def evaluate_model(pipeline, X, y):
    """
    This function evaluates the performance of the given pipeline on the given data
    :param pipeline: pipeline object to evaluate
    :type pipeline: sklearn.pipeline.Pipeline
    :param X: input feature variables
    :type X: array-like
    :param y: target variable
    :type y: array-like
    :return: mean squared error, mean absolute error
    :rtype: tuple
    """
    try:
        assert pipeline is not None, "Pipeline object should be initialized before evaluation"
        y_pred = pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return mse, mae
    except AssertionError as ae:
        print(ae)
    except Exception as e:
        print(f"An error occurred while evaluating the model: {e}")


def save_metrics(mse, mae, filepath):
    """
    Save the accuracy and confusion matrix to a local file
    """
    try:
        assert filepath is not None, "filepath should be provided"
        with open(filepath, "w") as f:
            f.write("Mean Squared Error: {:.4f}\\n".format(mse))
            f.write("Mean Absolute Error: {:.4f}\\n".format(mae))
    except AssertionError as ae:
        print(ae)
    except Exception as e:
        print(f"An error occurred while saving the metrics to a file: {e}")


def challenge_3():
    """
    This function performs the following tasks:
    1. Loads the wine quality dataset from sklearn
    2. Splits the data into training and testing sets
    3. Creates a pipeline with a RandomForestRegressor and StandardScaler
    4. Hyperparameter tunes the pipeline using GridSearchCV
    5. Trains the pipeline on the training data
    6. Evaluates the pipeline on the testing data
    7. Saves the evaluation metrics to a local file
    """
    parser = argparse.ArgumentParser(description='Challenge 2')
    parser.add_argument('filepath', type=str, help='filepath to save metrics to')
    args = parser.parse_args()
    filepath = args.filepath
    try:
        # Load the wine quality dataset
        print("Loading Wine Quality dataset...")
        X, y = load_wine(return_X_y=True)
        print("Wine Quality dataset loaded.")

        # Split the data into training and testing sets
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("Data split.")

        # Create the pipeline
        print("Creating pipeline...")
        pipeline = create_pipeline()
        print("Pipeline created.")

        # Tune the hyperparameters
        print("Starting hyperparameter tuning...")
        best_params = tune_hyperparameters(pipeline, X_train, y_train)
        print("Hyperparameter tuning complete.")

        # Train the model
        print("Training model...")
        model = train_model(X_train, y_train, best_params)
        print("Model trained.")
        # Evaluate the model
        print("Evaluating model...")
        mse, mae = evaluate_model(model, X_test, y_test)
        print("Model evaluated.")

        # Save the evaluation metrics to a local file
        print("Saving Metrics")
        save_metrics(mse, mae, filepath)
        print("Metrics Saved")
    except Exception as e:
        print("An error occurred: ", e)
        raise e


if __name__ == '__main__':
    challenge_3()
 '''

create_file("src/challenge_3.py",challenge_3_code)


challenge_4_code = '''import psutil
import pandas as pd
import argparse
import os

def check_file_type(filepath):
    """
    Check if the file extension is .csv
    Args:
        filepath (str): path to the file
    """
    try:
        assert filepath.endswith('.csv'), "Expected file type as .csv"
    except:
        print("Expected file type as .csv")
        raise

def get_process_info():
    """
    Get the process ID, process name, and memory utilization
    Returns:
        processes (list): list of dictionaries, each containing the process ID, process name, and memory utilization
    """
    try:
        processes = []
        for process in psutil.process_iter():
            try:
                process_info = process.as_dict(attrs=['pid', 'name', 'memory_info'])
                process_info['memory_utilization'] = process_info['memory_info'].rss / (1024 ** 2)
                processes.append(process_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except:
        print("Error Occured while getting the process information")
        raise
    return processes

def save_process_info(processes, filepath):
    """
    Save the process information to a csv file
    Args:
        processes (list): list of dictionaries, each containing the process ID, process name, and memory utilization
        filepath (str): path to save the csv file
    """
    # Create the directory if it does not exist
    try:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        df = pd.DataFrame(processes)
        df.to_csv(filepath)
    except:
        print("Error Occured while saving the process information")
        raise

def challenge_4():
    """
    Main function that gets the process information and saves it to a csv file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Filepath to save the csv file.")
    args = parser.parse_args()
    check_file_type(args.filepath)
    processes = get_process_info()
    save_process_info(processes, args.filepath)
    print(f"{args.filepath.split('/')[-1]} is saved at {args.filepath}")

if __name__ == "__main__":
    challenge_4()
'''

create_file("src/challenge_4.py",challenge_4_code)

challenge_5_code = '''import requests
import time
import argparse
import os


def check_file_type(filepath):
    """
    Check if the file extension is .html
    Args:
        filepath (str): path to the file
    """
    try:
        if not filepath.endswith('.html'):
            raise ValueError("Expected file type as .html")
    except:
        print("An error occured while checking the file type")


def download_html(url):
    """
    Download the HTML contents of a webpage
    Args:
        url (str): URL of the webpage
    Returns:
        html (str): HTML contents of the webpage
    """
    try:
        response = requests.get(url)
        html = response.text
        return html
    except:
        print("An error occured while downloading the HTML")


def save_html(filepath, html):
    """
    Save the HTML contents to a local file
    Args:
        filepath (str): path to save the HTML file
        html (str): HTML contents of the webpage
    """
    try:
        with open(filepath, 'w') as f:
            f.write(html)
    except:
        print("An error occured while saving the HTML")


def download_webpage(url, filepath):
    """
    Download the HTML contents of a webpage and save it to a local file
    Args:
        url (str): URL of the webpage
        filepath (str): path to save the HTML file
    """
    check_file_type(filepath)
    try:
        start_time = time.time()
        html = download_html(url)
        save_html(filepath, html)
        elapsed_time = time.time() - start_time
        print("Elapsed time: {:.4f} seconds".format(elapsed_time))
    except:
        print("An error occured while downloading and saving the webpage")


def challenge_5():
    """
    Main function that downloads the HTML contents of a webpage
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL of the webpage to download")
    parser.add_argument("filepath", help="Filepath to save the HTML file")
    args = parser.parse_args()

    try:
        check_file_type(args.filepath)
        assert requests.get(args.url).status_code == 200
    except ValueError as e:
        print(f'Error: {e}')
        return
    except AssertionError:
        print(f'Error: Invalid URL')
        return

    start_time = time.time()
    try:
        html = download_html(args.url)
        save_html(args.filepath, html)
    except Exception as e:
        print(f'Error: {e}')
        return

    elapsed_time = time.time() - start_time
    print("Elapsed time: {:.4f} seconds".format(elapsed_time))
    print(f"{args.filepath.split('/')[-1]} is saved at {args.filepath}")

if __name__ == "__main__":
    challenge_5()
'''

create_file("src/challenge_5.py",challenge_5_code)


# Creating data_files directory if it does not exist
data_files_dir = os.path.join(current_dir, 'data_files')
if not os.path.exists(data_files_dir):
    os.makedirs(data_files_dir)

# Creating output directory if it does not exist
data_files_dir = os.path.join(current_dir, 'output')
if not os.path.exists(data_files_dir):
    os.makedirs(data_files_dir)



# Creating 5 entry points
entry_points = {}
entry_points['console_scripts'] = []
for i in range(1,6):
    entry_points['console_scripts'].append(f'challenge-{i} = src.challenge_{i}:challenge_{i}')

setuptools.setup(
    name="LogixCodingChallenge",
    version="0.1",
    author='Jimmy Mirchandani',
    author_email='jimris22@gmail.com',
    description='Attempt at Logix Coding Challenge',
    packages=setuptools.find_packages(),
    install_requires=['pandas', 'numpy','scikit-learn','psutil','requests'],
    entry_points=entry_points,
    python_requires='>=3.7'
)