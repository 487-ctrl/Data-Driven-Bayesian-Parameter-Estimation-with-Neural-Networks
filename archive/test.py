# Import sbi and other libraries
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import matplotlib.pyplot as plt

# Define the simulator function
def linear_model(theta):
    # theta is a tensor with shape (num_simulations, 2)
    # theta[:, 0] is the slope and theta[:, 1] is the intercept
    # Generate 30 data points for each simulation
    x = torch.rand(theta.shape[0], 30) * 10
    # Add some noise to the data
    noise = torch.randn(theta.shape[0], 30) * 0.5
    # Compute the linear function
    y = theta[:, 0].unsqueeze(1) * x + theta[:, 1].unsqueeze(1) + noise
    # Return the simulated data
    return y

# Define the prior distribution over the parameters
prior = utils.BoxUniform(low=torch.tensor([0.0, 0.0]), high=torch.tensor([10.0, 10.0]))

# Generate some observed data from the simulator
theta_true = torch.tensor([3.0, 2.0]) # true slope and intercept
x_o = torch.rand(30) * 10 # observed x values
y_o = linear_model(theta_true).squeeze() # observed y values

# Plot the observed data
plt.scatter(x_o, y_o)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Observed data")
plt.show()

# Run inference with SNPE-C as the density estimator
posterior = infer(linear_model, prior, method='SNPE_C', num_simulations=1000, num_workers=4)

# Sample from the posterior distribution
samples = posterior.sample((1000,), x=x_o)

# Plot the posterior samples
plt.hist(samples[:, 0], bins=20, alpha=0.5, label="slope")
plt.hist(samples[:, 1], bins=20, alpha=0.5, label="intercept")
plt.legend()
plt.title("Posterior samples")
plt.show()

# Compute the posterior mean and standard deviation
posterior_mean = samples.mean(0)
posterior_std = samples.std(0)
print(f"Posterior mean: {posterior_mean}")
print(f"Posterior std: {posterior_std}")

# Plot the posterior predictive distribution
x_range = torch.linspace(0, 10, 100)
y_range = linear_model(posterior_mean.unsqueeze(0)).squeeze()
y_range_lower = linear_model((posterior_mean - posterior_std).unsqueeze(0)).squeeze()
y_range_upper = linear_model((posterior_mean + posterior_std).unsqueeze(0)).squeeze()
plt.fill_between(x_range, y_range_lower, y_range_upper, alpha=0.3, label="Uncertainty")
plt.plot(x_range, y_range, label="Prediction")
plt.scatter(x_o, y_o, label="Observation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Posterior predictive distribution")
plt.show()
