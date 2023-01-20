import argparse
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
    challenge_1()