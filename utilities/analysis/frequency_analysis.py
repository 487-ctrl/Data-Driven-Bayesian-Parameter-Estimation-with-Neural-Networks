import os
import numpy as np
import matplotlib.pyplot as plt
import torch

class FrequencyAnalysis():
    def __init__(self, dt, T):
        self.dt = dt
        self.T = T

    def plot_frequency_analytics(self, frequency):

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))

        plt.plot(frequency, linewidth=1)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        return plt

    def save_figure(self, plt, save_location, filename):
        """
        Save the plot as a PDF file.

        Parameters:
        save_location (str): The directory to save the plot.
        filename (str): The name of the file to save the plot.
        """
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        plt.savefig(f'{save_location}/{filename}.pdf')

    def remove_noise(self, frequency_series):
        """
        Remove the noise from the given frequency_series using FFT.
        
        Parameters:
            frequency_series: array-like
                The input frequency series to remove noise from.
        
        Returns:
            array-like
                The frequency series with noise removed.
        """
        # Remove the noise
        fft = np.fft.fft(frequency_series)
        fft[np.abs(fft) < 0.01] = 0
        return torch.from_numpy(np.fft.ifft(fft))