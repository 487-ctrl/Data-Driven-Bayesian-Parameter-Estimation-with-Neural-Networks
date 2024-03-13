import pandas as pd
import torch

class ErrorAnalysis:
    def error_metrics(self, true_values, result_samples):
        """
        Calculates error metrics for true values and predicted samples.

        Args:
            true_values (list): List of true values.
            result_samples (list): List of predicted samples.

        Returns:
            pd.DataFrame: Dataframe with the calculated error metrics.
        """
        metrics_list = [self._mse(true_values, result_samples)]
        metrics_names = ['MSE']
        for metric_func in [self._mae, self._rmse, self._mape, self._mase]:
            metrics_list.append(metric_func(true_values, result_samples))
            metrics_names.append(metric_func.__name__[1:])

        df = pd.DataFrame(metrics_list).T
        df.index = range(len(result_samples[0]))
        df.columns = metrics_names
        return df

    def _mse(self, true_values, result_samples):
        """
        Calculates the Mean Squared Error (MSE) between true values and predicted samples.

        Args:
            true_values (list): List of true values.
            result_samples (list): List of predicted samples.

        Returns:
            float: Mean Squared Error (MSE).
        """
        mse = torch.zeros(len(result_samples[0]))
        for samples, ground_truth in zip(result_samples, true_values):
            mse += ((samples - ground_truth) ** 2).mean(dim=0)
        return mse.mean().item()

    def _mae(self, true_values, result_samples):
        """
        Calculates the Mean Absolute Error (MAE) between true values and predicted samples.

        Args:
            true_values (list): List of true values.
            result_samples (list): List of predicted samples.

        Returns:
            float: Mean Absolute Error (MAE).
        """
        mae = torch.zeros(len(result_samples[0]))
        for samples, ground_truth in zip(result_samples, true_values):
            mae += (torch.abs(samples - ground_truth)).mean(dim=0)
        return mae.mean().item()

    def _rmse(self, true_values, result_samples):
        """
        Calculates the Root Mean Squared Error (RMSE) between true values and predicted samples.

        Args:
            true_values (list): List of true values.
            result_samples (list): List of predicted samples.

        Returns:
            float: Root Mean Squared Error (RMSE).
        """
        return self._mse(true_values, result_samples).sqrt()

    def _mape(self, true_values, result_samples):
        """
        Calculates the Mean Absolute Percentage Error (MAPE) between true values and predicted samples.

        Args:
            true_values (list): List of true values.
            result_samples (list): List of predicted samples.

        Returns:
            float: Mean Absolute Percentage Error (MAPE).
        """
        mape = torch.zeros(len(result_samples[0]))
        for samples, ground_truth in zip(result_samples, true_values):
            mask = ground_truth != 0
            mape += torch.where(mask, (torch.abs(samples - ground_truth) / ground_truth).mean(dim=0), torch.tensor(0.0))
        return mape.mean().item()

    def _mase(self, true_values, result_samples):
        """
        Calculates the Mean Absolute Scaled Error (MASE) between true values and predicted samples.

        Args:
            true_values (list): List of true values.
            result_samples (list): List of predicted samples.

        Returns:
            float: Mean Absolute Scaled Error (MASE).
        """
        mase = torch.zeros(len(result_samples[0]))
        for samples, ground_truth in zip(result_samples, true_values):
            mase += torch.abs(samples - ground_truth).mean(dim=0) / (torch.abs(ground_truth).mean(dim=0) - ground_truth.mean(dim=0))
        return mase.mean().item()
    