import pandas as pd
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
