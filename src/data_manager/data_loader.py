import os
import zipfile
import requests
from tabulate import tabulate

# Mapping of dataset names to URLs
datasets = {
    'DE_KA01': 'https://osf.io/download/gptxv/',
    'DE_OL01': 'https://osf.io/download/qz8xr/',
    'EE01': 'https://osf.io/download/t5ske/',
    'ES_GC01': 'https://osf.io/download/wz42b/',
    'ES_GC02': 'https://osf.io/download/rukat/',
    'ES_PM01': 'https://osf.io/download/2qn9k/',
    'ES_PM02': 'https://osf.io/download/vrc6a/',
    'ES_PM03': 'https://osf.io/download/7xe5r/',
    'FO01': 'https://osf.io/download/a7h5b/',
    'FR01': 'https://osf.io/download/hfsrz/',
    'GB01': 'https://osf.io/download/cfv47/',
    'GB02': 'https://osf.io/download/h5ydu/',
    'HR01': 'https://osf.io/download/r9eh6/',
    'IS01': 'https://osf.io/download/sxph8/',
    'IT01': 'https://osf.io/download/c754b/',
    'PL01': 'https://osf.io/download/wq3te/',
    'PT01': 'https://osf.io/download/5zgwn/',
    'PT_LI01': 'https://osf.io/download/jt84d/',
    'RU01': 'https://osf.io/download/tvsyc/',
    'SE01': 'https://osf.io/download/e2xfb/',
    'sync01': 'https://osf.io/download/p5xyr/',
    'TUR_IS01': 'https://osf.io/download/3kwgv/',
    'US_TX01': 'https://osf.io/download/t5wxz/',
    'US_TX02': 'https://osf.io/download/zngy8/',
    'US_UT': 'https://osf.io/download/8rp4v/',
    'ZA01': 'https://osf.io/download/gzk7d/',
}

def print_available_datasets():
    """
    Prints the dataset names and URLs as a table.

    Args:
    - datasets (dict): A dictionary mapping dataset names to their URLs.
    """
    # Prepare data for tabulate
    data = [(name, url) for name, url in datasets.items()]

    # Print the table
    print(tabulate(data, headers=['Dataset Name', 'URL'], tablefmt='grid'))

def _download_and_extract(url, location, name):
    """
    Downloads a file from a URL, saves it to a specified location, extracts it if it's a zip file,
    deletes the zip file, and renames the CSV file.

    Args:
    - url (str): The URL of the file to be downloaded.
    - location (str): The location where the file should be saved.
    - name (str): The name of the file to be saved.
    """

    # Ensure the directory exists. If not, it creates the directory.
    os.makedirs(location, exist_ok=True)

    # Determine the file type from the URL
    if url.endswith('.zip'):
        is_zip = True
        filename = os.path.join(location, name + '.zip')
    else:
        is_zip = False
        filename = os.path.join(location, name + '.csv')

    # If file already exists, skip download
    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Skipping download.")
        return

    # Send a GET request to the URL
    response = requests.get(url)

    # Save the file in binary format
    with open(filename, 'wb') as f:
        f.write(response.content)

    # If it's a zip file, extract it
    if is_zip:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(location)

            # Find the CSV file and rename it
            for file in zip_ref.namelist():
                if file.endswith('.csv'):
                    csv_file = os.path.join(location, file)
                    new_csv_file = os.path.join(location, name + '.csv')
                    os.rename(csv_file, new_csv_file)

        # Delete the zip file
        os.remove(filename)

    # Print only the final CSV file path
    print(f"Data downloaded and saved as {os.path.join(location, name + '.csv')}")

def load_dataset(link, name, save_location):
    """
    Downloads a dataset from a given link, searches for a CSV file, and saves it to a specified location.

    Args:
    - link (str): The URL of the dataset to be downloaded.
    - name (str): The name of the dataset.
    - save_location (str): The directory where the dataset should be saved.
    """
    _download_and_extract(link, save_location, name)