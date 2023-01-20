import requests
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
