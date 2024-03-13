import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import torch

class DistributionAnalysis():

    def __init__(self, true_partameter_base, save_location):
        """
        Initialize the PosteriorAnalysis with prior and posterior distribution samples.

        Parameters:
        true_values (list or None, optional): True values for each variable, if known. Defaults to None.
        value_names (list of str or None, optional): Names of the variables. Defaults to None.
        """
        self.true_values = list(true_partameter_base.values())
        self.value_names = list(true_partameter_base.keys())
        self.save_location = save_location
                                
    def plot_distribution_analytics(self, distribution_samples, title, filename, confidence_interval=95, metrics=True):
        """
        Plot the distribution of samples along with common statistics and confidence intervals.

        Parameters:
        confidence_interval (int): Confidence interval percentage. Default is 95.
        metrics (bool): Whether to calculate and display statistics metrics. Default is True.
        """
        # Get the number of parameters
        num_parameters = len(self.true_values)

        # Create subplots for prior and posterior distributions
        fig, axes = plt.subplots(nrows=1, ncols=num_parameters, figsize=(4 * num_parameters, 4))

        for j, ax in enumerate(axes):
            
            samples = distribution_samples[:, j]
            parameter_name = self.value_names[j]

            # Normalize the statistics metrics
            exp = math.floor(math.log10(max(abs(max(samples)), abs(min(samples)))))
            samples /= 10**exp

            # Plot kernel density estimate
            ax.hist(samples, 100, density=True, facecolor='b', alpha=0.75)
            ax.set_ylabel('Density') if j == 0 else ax.set_ylabel('')

            # Plot true value if available
            if self.true_values != None:
                true_value = self.true_values[j] 
                true_value /= 10**exp
                ax.axvline(true_value, color='red', linestyle='dashed', linewidth=1, label='True Value')
            
            # Plot confidence interval if available
            if confidence_interval != None:
                ci_lower, ci_upper = np.percentile(samples, [(100 - confidence_interval) / 2, confidence_interval + (100 - confidence_interval) / 2])
                ax.axvline(ci_lower, color='black', linestyle=':', linewidth=0.5)
                ax.axvline(ci_upper, color='black', linestyle=':', linewidth=0.5)
                ax.fill_betweenx(ax.get_ylim(), ci_lower, ci_upper, color='lightblue', alpha=0.3, label=f'{confidence_interval}% Confidence Interval')

            # Calculate and display statistics metrics
            if metrics:

                # Display statistics metrics
                ax.text(0.05, 0.95, f'Skewness: {self._skewness(samples) :.2f}', fontsize=8, ha='left', va='top', transform=ax.transAxes, color='black')
                ax.text(0.05, 0.90, f'Kurtosis: {self._kurtosis(samples):.2f}', fontsize=8, ha='left', va='top', transform=ax.transAxes, color='black')
                ax.text(0.95, 0.95, f'Mean: {self._mean(samples):.2f}', fontsize=8, ha='right', va='top', transform=ax.transAxes, color='black')
                ax.text(0.95, 0.90, f'Std: {self._std(samples):.2f}', fontsize=8, ha='right', va='top', transform=ax.transAxes, color='black')

            # Add grid to x-axis
            ax.grid(axis='x', linestyle='--', alpha=0.5)

            # Set x-axis label for distribution
            ax.set_xlabel(f'${parameter_name}*10^{{{exp}}}$')

            # Add legend to the last subplot of posterior distribution
            if j == num_parameters - 1:
                ax.legend(loc='lower right', prop={'size': 8})

        # Add title
        plt.suptitle(title)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        plt.savefig(f'{self.save_location}/figures/{filename}.pdf')

        return plt
    
    def distribution_metrics(self, samples, filename):
        df = pd.DataFrame(index= self.value_names, columns=['True Value', 'Skewness', 'Kurtosis', 'Mean', 'Std'])
        df['True Value'] = self.true_values
        df['Skewness'] = self._skewness(samples)
        df['Kurtosis'] = self._kurtosis(samples)
        df['Mean'] = self._mean(samples)
        df['Std'] = self._std(samples)

        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        df.to_csv(f'{self.save_location}/tables/{filename}.csv')

        return df

    def _skewness(self, samples):
        """
        Calculate the skewness of the given samples.

        Parameters:
            self (object): The object itself.
            samples (array-like): Input array.

        Returns:
            float: The skewness of the input samples.
        """
        return stats.skew(samples)
    
    def _kurtosis(self, samples):
        """
        Calculate the kurtosis of the given samples.

        :param samples: List of numerical samples
        :return: Kurtosis value of the samples
        """
        return stats.kurtosis(samples)
    
    def _mean(self, samples):
        """
        Calculate the mean of the given samples.

        Parameters:
            samples (array-like): The samples for which the mean needs to be calculated.

        Returns:
            float: The mean value of the samples.
        """
        return [s.mean().item() for s in samples.T]
    
    def _std(self, samples):
        """
        Calculate the standard deviation of the given samples.

        Parameters:
            samples (array-like): An array-like sequence of samples.

        Returns:
            list: The standard deviation of each sample.
        """
        return [s.std().item() for s in samples.T]


    
