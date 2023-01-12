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
