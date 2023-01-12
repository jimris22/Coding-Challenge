import setuptools
import os
import sys
import ast
 
# Getting current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if os.path.exists("README.md"):
    os.remove("README.md")
    print(f""" "README.md" already exists, removed old file""")

with open("README.md", "w") as f:
    f.write("# Challenge Project\n\n")
    f.write("This project includes 5 challenges that cover a range of topics such as data manipulation, machine learning and web scraping. Each challenge can be run independently using the command line interface.\n\n")
    f.write("## Installation\n\n")
    f.write("To install the dependencies for this project, you will need to have Python 3.7 or later and pip installed on your machine. Once you have those set up, navigate to the root directory of the project and run the following command:\n\n")
    f.write("```bash\n")
    f.write("python setup.py install\n")
    f.write("```\n\n")
    f.write("This will install all the necessary packages including pandas, numpy, scikit-learn, requests, and psutil.\n\n")
    f.write("## Running the challenges\n\n")
    f.write("Once the dependencies are installed, you can run any of the challenges by using the command line interface. Navigate to the root directory of the project and run the following command for the corresponding challenge:\n\n")
    f.write("```bash\n")
    f.write("challenge-1 path_to_data col1 col2 path_where_to_save\n")
    f.write("challenge-2 path_to_store_file_txt \n")
    f.write("challenge-3 path_to_store_file_txt \n")
    f.write("challenge-4 path_to_store_file_csv \n")
    f.write("challenge-5 url path_to_where_to_store_html \n")
    f.write("```\n\n")
    f.write("## Note\n")
    f.write("- Make sure you have internet connection to download the data for challenge 2,3\n")
    f.write("- Make sure you have internet connection to download the data for challenge 1 from https://catalog.data.gov/dataset/lottery-mega-millions-winning-numbers-beginning-2002/resource/61eea3d0-6b6d-43a4-ae02-6432e6a4e517 and store the file in data_files directory\n")
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

challenge_1_code = '''import pandas as pd
import argparse

def check_file_type(filepath):
    """
    Check if the file extension is .txt
    Args:
        filepath (str): path to the file
    """
    if not filepath.endswith('.csv'):
        raise ValueError("Expected file type as .csv")

def load_data(file_path):
    """
    Load the CSV file into a pandas data frame
    """
    return pd.read_csv(file_path)

def print_sorted_columns(df):
    """
    Print the name of each column in sorted order
    """
    print("Columns in sorted order:")
    print(sorted(df.columns))

def print_total_rows(df):
    """
    Print the total number of rows in the data frame
    """
    print("Total number of rows:")
    print(len(df))

def add_new_column(df, column1, column2):
    """
    Add a new column to the data frame that is computed from two other column values
    """
    df["new_column"] = df[column1] + df[column2]
    return df

def save_to_csv(df, file_path):
    """
    Save the resulting data frame to a local CSV file
    """
    df.to_csv(file_path, index=False)

def challenge_1():
    # Create a parser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("data_file_path", help="Path to the CSV file")
    parser.add_argument("column1", help="Name of column 1")
    parser.add_argument("column2", help="Name of column 2")
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
    df = add_new_column(df, args.column1, args.column2)

    check_file_type(args.file_path)
    # Save the data to a CSV file
    save_to_csv(df, args.file_path)

# Call the main function to execute the script
if __name__ == "__main__":
    challenge_1()
'''
create_file("src/challenge_1.py",challenge_1_code)

challenge_2_code = '''from sklearn.datasets import load_iris
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
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 5, 10],
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
        (f.write("Accuracy: {:.4f}\\n".format(accuracy)))
        (f.write("Confusion Matrix:\\n{}".format(cm)))

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


'''
create_file("src/challenge_2.py",challenge_2_code.replace('\n', os.linesep))

challenge_3_code = '''import pandas as pd
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
        'reg__n_estimators': [50, 100, 200],
        'reg__max_depth': [None, 5, 10],
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
        f.write("Mean Squared Error: {:.4f}\\n".format(mse))
        f.write("Mean Absolute Error: {:.4f}\\n".format(mae))


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

 '''

create_file("src/challenge_3.py",challenge_3_code.replace('\n', os.linesep))


challenge_4_code = '''import psutil
import pandas as pd
import argparse
import os

def check_file_type(filepath):
    """
    Check if the file extension is .txt
    Args:
        filepath (str): path to the file
    """
    if not filepath.endswith('.csv'):
        raise ValueError("Expected file type as .csv")

def get_process_info():
    """
    Get the process ID, process name, and memory utilization
    Returns:
        processes (list): list of dictionaries, each containing the process ID, process name, and memory utilization
    """
    processes = []
    for process in psutil.process_iter():
        try:
            process_info = process.as_dict(attrs=['pid', 'name', 'memory_info'])
            process_info['memory_utilization'] = process_info['memory_info'].rss / (1024 ** 2)
            processes.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes

def save_process_info(processes, filepath):
    """
    Save the process information to a csv file
    Args:
        processes (list): list of dictionaries, each containing the process ID, process name, and memory utilization
        filepath (str): path to save the csv file
    """
    # Create the directory if it does not exist
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    df = pd.DataFrame(processes)
    df.to_csv(filepath)

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
    challenge_4()'''

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
    if not filepath.endswith('.html'):
        raise ValueError("Expected file type as .html")


def download_html(url):
    """
    Download the HTML contents of a webpage
    Args:
        url (str): URL of the webpage
    Returns:
        html (str): HTML contents of the webpage
    """
    response = requests.get(url)
    html = response.text
    return html


def save_html(filepath, html):
    """
    Save the HTML contents to a local file
    Args:
        filepath (str): path to save the HTML file
        html (str): HTML contents of the webpage
    """
    with open(filepath, 'w') as f:
        f.write(html)


def download_webpage(url, filepath):
    """
    Download the HTML contents of a webpage and save it to a local file
    Args:
        url (str): URL of the webpage
        filepath (str): path to save the HTML file
    """
    check_file_type(filepath)
    start_time = time.time()
    html = download_html(url)
    save_html(filepath, html)
    elapsed_time = time.time() - start_time
    print("Elapsed time: {:.4f} seconds".format(elapsed_time))


def challenge_5():
    """
    Main function that downloads the HTML contents of a webpage
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL of the webpage to download")
    parser.add_argument("filepath", help="Filepath to save the HTML file")
    args = parser.parse_args()

    download_webpage(args.url, args.filepath)


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