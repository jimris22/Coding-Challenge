import psutil
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
