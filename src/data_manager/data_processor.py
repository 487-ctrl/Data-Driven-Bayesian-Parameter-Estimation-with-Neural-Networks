import datetime
import os
import pandas as pd

def clean_dataset(save_location, name, data_location):
    """
    Cleans a time-frequency dataset by removing any corrupted or invalid data points and saves the cleaned data to a specified location.

    Args:
    - save_location (str): The directory where the cleaned dataset should be saved.
    - name (str): The name of the cleaned dataset.
    - data_location (str): The directory where the original dataset is located.
    """
    # Load the original dataset
    try:
        df = pd.read_csv(os.path.join(data_location, name + '.csv'), encoding='latin-1', delimiter=';')
    except FileNotFoundError:
        print(f"Dataset '{name}' not found in '{data_location}'.")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file: {e}")
        return

    # Check if required columns exist
    expected_columns = ['Time', 'f50_DE_KA']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Dataset is missing required columns: {missing_columns}")
        return

    # Drop rows with missing or invalid values
    df.dropna(subset=expected_columns, inplace=True)

    # Convert 'Time' column to datetime format
    try:
        df['Time'] = pd.to_datetime(df['Time'], format='%d.%m.%Y %H:%M')
    except ValueError as e:
        print(f"Error converting 'Time' column to datetime: {e}")
        return

    # Remove duplicates based on 'Time' column
    df.drop_duplicates(subset='Time', keep='first', inplace=True)

    # Sort the dataset by time
    df.sort_values(by='Time', inplace=True)

    # Save the cleaned dataset
    cleaned_dataset_path = os.path.join(save_location, name + '_cleaned.csv')
    df.to_csv(cleaned_dataset_path, index=False)

    print(f"Cleaned dataset saved as '{cleaned_dataset_path}'.")

def make_observation(dataset_location, dataset_name, T):
    """
    Creates an observation from the dataset by mapping time (in seconds) to the corresponding value.

    Args:
    - dataset_location (str): The directory where the dataset is located.
    - dataset_name (str): The name of the dataset file without the file extension.
    - T (int): The total length of the simulation.

    Returns:
    - observation (dict): A dictionary mapping time (in seconds) to the corresponding value.
    """
    # Construct the full path to the dataset
    dataset_path = os.path.join(dataset_location, dataset_name + '.csv')

    # Load the dataset with 'latin-1' encoding and comma delimiter
    try:
        df = pd.read_csv(dataset_path, encoding='latin-1', delimiter=',')
    except FileNotFoundError:
        print(f"Dataset '{dataset_name}' not found in '{dataset_location}'.")
        return None

    # Assuming the dataset has columns 'Time' and 'Value'
    if 'Time' not in df.columns or 'Value' not in df.columns:
        print("Dataset columns are not formatted correctly.")
        return None

    # Convert time to seconds and create the observation mapping
    observation = {}
    for index, row in df.iterrows():
        time_seconds = datetime.datetime.strptime(row['Time'], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        observation[time_seconds] = row['Value']

    return observation

