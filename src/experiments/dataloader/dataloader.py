import requests
import zipfile
import os

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

def _download_and_extract(url, location, name):
    """
    This function downloads a file from a URL, saves it to a specified location,
    extracts it if it's a zip file, deletes the zip file, and renames the CSV file.

    Parameters:
    url (str): The URL of the file to be downloaded.
    location (str): The location where the file should be saved.
    name (str): The name of the file to be saved.
    """

    # Ensure the directory exists. If not, it creates the directory.
    os.makedirs(location, exist_ok=True)

    # Send a GET request to the URL
    response = requests.get(url)

    # Determine the file type from the content
    # Zip files usually start with 'PK' in their binary content
    if b'PK' in response.content[:4]:  
        filename = os.path.join(location, name + '.zip')
    else:
        filename = os.path.join(location, name + '.csv')

    # Save the file in binary format
    with open(filename, 'wb') as f:
        f.write(response.content)

    # If it's a zip file, extract it
    if filename.endswith('.zip'):
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

def fetch_dataset(name, out_dir='frequency_data'):
    """
    This function fetches a dataset given its name and saves it to a specified location.

    Parameters:
    name (str): The name of the dataset to be downloaded. It should be a key in the 'datasets' dictionary.
    out_dir (str): The directory where the dataset should be saved. Defaults to 'frequency_data'.
    """

    # Check if the dataset name is valid
    if name not in datasets:
        print(f"Unknown dataset name: {name}")
        return

    # Get the URL for the dataset
    url = datasets[name]

    # Call the download_and_extract function
    _download_and_extract(url, out_dir, name)

def _get_dataset_options():
    """
    This function returns the available dataset options.

    Returns:
    A list of strings, each string is a dataset name.
    """

    # Get the keys from the 'datasets' dictionary
    # These keys are the names of the available datasets
    dataset_names = list(datasets.keys())

    return dataset_names

# Fetches about 20 GB of data, do not use frequently!
def fetch_all_datasets(out_dir='frequency_data'):
    """
    This function fetches all available datasets and saves them to a specified location.

    Parameters:
    out_dir (str): The directory where the datasets should be saved. Defaults to 'frequency_data'.
    """

    # Get the names of all available datasets
    dataset_names = _get_dataset_options()

    # Fetch each dataset
    for name in dataset_names:
        print(f"Fetching dataset: {name}")
        fetch_dataset(name, out_dir)

    print("All datasets fetched successfully.")

fetch_all_datasets()