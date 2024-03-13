import pandas as pd

def sample_df(dist, param_names, num_samples=25000):
    samples = dist.sample((num_samples,))
    return pd.DataFrame(samples, columns=param_names)