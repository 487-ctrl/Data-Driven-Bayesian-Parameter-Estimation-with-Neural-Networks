import pandas as pd
import random
import warnings

dataset_path = "../../../dataset/2019.csv"

def _process_dataset(observation_length):
    """
    Retrieve and filter observations, parse into blocks, and return valid time and frequency lists.

    Parameters:
    - interval (str): Time interval for grouping (default is '15min', options: '15min' or '1H').

    Returns:
    - list of tuple: List of pairs representing time and frequency for each valid block.
    """

    interval = '15min' if observation_length == 900 else '1H' 

    # Load the CSV file into a DataFrame
    df = pd.read_csv(dataset_path, header=None)  # No header specified

    # Convert the first column to datetime with second precision
    df[0] = pd.to_datetime(df[0], format='%Y-%m-%d %H:%M:%S')

    # Extract observations from the same day
    df_same_day = df[df[0].dt.date == df[0].iloc[0].date()]

    # Group into intervals (15 minutes or 1 hour)
    grouped_df = df_same_day.groupby(pd.Grouper(key=0, freq=interval))

    # Extract time and frequency lists for each valid block
    result = []
    for _, group in grouped_df:
        if len(group) >= observation_length:
            # Check if the observation starts at the correct hour/15-minute interval
            if interval == '1H' and group.iloc[0][0].minute == 0 and group.iloc[0][0].second == 0:
                time_list = group.iloc[:observation_length][0].tolist()
                frequency_list = group.iloc[:observation_length][1].tolist()
                result.append((time_list, frequency_list))
            elif interval == '15min' and group.iloc[0][0].minute % 15 == 0 and group.iloc[0][0].second == 0:
                time_list = group.iloc[:observation_length][0].tolist()
                frequency_list = group.iloc[:observation_length][1].tolist()
                result.append((time_list, frequency_list))
            else:
                warnings.warn("Observation does not start at the correct interval. Skipping this observation.")

    return result

def select_observation(interval='15min'):
    """
    Retrieve a random observation pair from the processed dataset within the specified interval.

    Parameters:
    - interval (str): Time interval for grouping (default is '15min', options: '15min' or '1H').

    Returns:
    - tuple: A pair representing time and frequency for the randomly selected observation.
    """
    processed_data = _process_dataset(interval)
    if processed_data:
        random_observation = random.choice(processed_data)
        return random_observation
    else:
        print("No valid observations found.")
        return None
    
def sample_df(dist, param_names, num_samples=25000):
    samples = dist.sample((num_samples,))
    return pd.DataFrame(samples, columns=param_names)
