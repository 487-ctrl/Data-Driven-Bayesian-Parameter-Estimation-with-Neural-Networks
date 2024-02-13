import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error

def _evaluate_density_metrics(true_parameters, predicted_parameters):
    """
    Evaluate density prediction metrics for model inference.
    
    Parameters:
        true_parameters (array-like): True parameter values.
        predicted_parameters (array-like): Predicted parameter values.
    
    Returns:
        float: Mean squared error (MSE) between the true and predicted parameter densities.
    """
    # Estimate density of true parameters
    true_density = gaussian_kde(true_parameters)
    true_density_values = true_density(true_parameters)
    
    # Estimate density of predicted parameters
    predicted_density = gaussian_kde(predicted_parameters)
    predicted_density_values = predicted_density(predicted_parameters)
    
    # Calculate mean squared error (MSE) between true and predicted parameter densities
    mse = mean_squared_error(true_density_values, predicted_density_values)
    
    return mse

def create_metrics_dataframe(parameters, true_parameters, predicted_parameters):
    """
    Create a DataFrame with columns for parameters of the model and rows for density prediction metrics.
    
    Parameters:
        parameters (list): List of parameter names.
        true_parameters (array-like): True parameter values.
        predicted_parameters (array-like): Predicted parameter values.
    
    Returns:
        pandas.DataFrame: DataFrame containing density prediction metrics for each parameter.
    """
    # Initialize DataFrame
    metrics_df = pd.DataFrame(index=['Density MSE'])
    
    # Evaluate density prediction metrics
    density_mse = _evaluate_density_metrics(true_parameters, predicted_parameters)
    
    # Add density MSE to DataFrame
    metrics_df.loc['Density MSE', :] = density_mse
    
    # Add parameters to DataFrame columns
    metrics_df.columns = parameters
    
    return metrics_df