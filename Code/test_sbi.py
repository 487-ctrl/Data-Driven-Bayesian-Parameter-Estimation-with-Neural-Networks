import torch
from plot_functions import PlotFunctions
from simulator_functions import SimulatorFunctions
from torch.distributions import Uniform, Independent
from sbi.inference import infer

# definition prior distribution
prior_min = torch.tensor([0.1, 0.0, 0.0]) 
prior_max = torch.tensor([1.0, 1.0, 1.0])
prior = Independent(Uniform(prior_min, prior_max), 1)

# generate synthetic features
theta_true = torch.tensor([[0.5, 0.8, 0.6]])
simulator = SimulatorFunctions()
x_o = simulator.swing_equation(theta_true)

# run sbi using 'SNPE' and 1000 simulations
posterior = infer(simulator.swing_equation, prior, method='SNPE', num_simulations=1000)

# set default value of x to x_0
posterior.set_default_x(x_o)

plot_prior_posterior(prior, posterior, theta_true)
plot_simulated_time_frequency(simulator.swing_equation(posterior.sample((1000,))))

