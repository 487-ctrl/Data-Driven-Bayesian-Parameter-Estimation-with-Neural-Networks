# Data-Driven Bayesian Parameter Estimation with Neural Networks
This Repository consists of the main document of my Bachelor's Thesis called "Data-driven Bayesian Parameter Estimation with Neural Networks", along with the Code needed to reproduce my findings.

```python
# Import the sbi library and other dependencies
import sbi
import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the simulator function that implements the aggregated Swing equation
# The inputs are the parameters H and D, and the outputs are the frequency changes
def simulator(theta):
    # Unpack the parameters
    H, D = theta
    # Set the constants
    f_0 = 50 # nominal frequency in Hz
    S_B = 1000 # total apparent power in MVA
    P_m = 500 # mechanical power of generators in MW
    P_L = 400 # electrical power of loads in MW
    P_loss = 50 # loss power in MW
    # Set the initial conditions
    f = f_0 # initial frequency in Hz
    t = 0 # initial time in s
    dt = 0.01 # time step in s
    T = 10 # simulation time in s
    # Initialize the output array
    output = np.zeros(int(T/dt))
    # Run the simulation loop
    for i in range(len(output)):
        # Update the frequency change using the aggregated Swing equation
        f_dot = (f_0 / (2 * H * S_B)) * (P_m - P_L - P_loss - D * f)
        # Update the frequency using Euler's method
        f = f + f_dot * dt
        # Update the time
        t = t + dt
        # Store the frequency change in the output array
        output[i] = f_dot
    # Return the output array as a tensor
    return torch.from_numpy(output)

# Define the prior distribution over the parameters H and D
# Use a uniform distribution with lower and upper bounds
prior = sbi.utils.BoxUniform(low=torch.tensor([0.1, 0.1]), high=torch.tensor([10.0, 10.0]))

# Load the frequency data of the power grid of Mallorca as observation data
# Use a numpy array and convert it to a tensor
observation = np.loadtxt("frequency_data.txt")
observation = torch.from_numpy(observation)

# Choose an inference method from the sbi library, e.g. SNPE
inference = sbi.inference.SNPE(prior)

# Train a neural network that learns the posterior distribution over the parameters H and D
# Use 1000 simulations for training
posterior = inference.append_simulations(simulator, 1000).train()

# Evaluate the quality of the learned posterior distribution
# Plot the posterior distribution
samples = posterior.sample((1000,))
_ = sbi.analysis.pairplot(samples)
plt.show()
# Plot the posterior predictions
posterior_samples = posterior.sample((100, observation))
_ = sbi.analysis.pairplot(posterior_samples, limits=[[0, 10], [0, 10]], figsize=(5, 5))
plt.show()
# Calculate the likelihood of the observation data under the posterior distribution
log_prob = posterior.log_prob(observation)
print(f"The log probability of the observation data is {log_prob}")
```