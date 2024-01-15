import matplotlib.pyplot as plt
import numpy as np

class PlotFunctions:
    def __init__(self):
        pass

    # plot prior and posterior of given parameters
    def plot_prior_posterior(self, prior, posterior, theta_true):

        # label plots
        labels = ['M', 'Pm', 'Pe']

        # draw 6 subplots 
        fig, axes = plt.subplots(2,3, figsize=(12,8))  

        for i in range(3):

            # sample 1000 from prior and posterior
            prior_samples = prior.sample((10000,))[:,i]
            posterior_samples = posterior.sample((1000,))[:,i]

            # draw histogram for each parameter using prior
            axes[0,i].hist(prior_samples, bins=20, density=True)
            axes[0,i].set_xlabel(labels[i])
            axes[0,i].set_ylabel('Density')
            axes[0,i].set_title('Prior distribution')

            # draw histogram for each parameter using posterior
            axes[1,i].hist(posterior_samples, bins=20, density=True)
            axes[1,i].axvline(theta_true[0][i].item(), color='red', label='True value') # true value
            axes[1,i].set_xlabel(labels[i])
            axes[1,i].set_ylabel('Density')
            axes[1,i].legend()
            axes[1,i].set_title('Posterior distribution')

        # save plot
        plt.tight_layout()
        plt.savefig("prior_posterior_comparison.pdf", format="pdf", bbox_inches="tight")

    def plot_simulated_time_frequency(self, simulator_outputs):

        # extract delta and omega from simulator output
        delta_samples = simulator_outputs[:, 0].cpu().numpy()
        omega_samples = simulator_outputs[:, 1].cpu().numpy()

        # create timeline
        t_max = 10.0 
        dt = 0.01 
        t = np.arange(0, t_max, dt)

        # plot delta and omega by time
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, delta_samples.T, color='blue', alpha=0.1)
        plt.xlabel('Time')
        plt.ylabel('delta')

        plt.subplot(2, 1, 2)
        plt.plot(t, omega_samples.T, color='blue', alpha=0.1)
        plt.xlabel('Time')
        plt.ylabel('omega')

        plt.tight_layout()
        plt.savefig("pdf_out/simulated_time_frequency.pdf", format="pdf", bbox_inches="tight")
