import requests
import zipfile
import os
import glob

def download_and_extract(url, location, name):
    """
    This function downloads a file from a URL, saves it to a specified location,
    extracts it if it's a zip file, and deletes the zip file.

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

        # Delete the zip file
        os.remove(filename) 

        # Find the CSV file
        csv_file = glob.glob(os.path.join(location, '*.csv'))[0]

        # Rename the CSV file
        new_csv_file = os.path.join(location, name + '.csv')
        os.rename(csv_file, new_csv_file)

    # Print only the final CSV file path
    print(f"Data downloaded and saved as {filename.replace('.zip', '.csv')}")

DE_KA01 = 'https://osf.io/download/gptxv/'
DE_OL01 = 'https://osf.io/download/qz8xr/'
EE01 = 'https://osf.io/download/t5ske/'

download_and_extract(DE_KA01, "frequency_data", "DE_KA01")

