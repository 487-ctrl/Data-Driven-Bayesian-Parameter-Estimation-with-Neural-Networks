import torch
from sbi.inference import SNPE, prepare_for_sbi
from plot_functions import PlotFunctions
from simulator_functions import SimulatorFunctions
from torch.distributions import Uniform, Independent

# check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# definition prior distribution
prior_min = torch.tensor([0.0, 0.0, 0.0]).to(device)
prior_max = torch.tensor([1.0, 1.0, 1.0]).to(device)
prior = Independent(Uniform(prior_min, prior_max), 1)

# generate synthetic features
theta_true = torch.tensor([[0.5, 0.8, 0.6]]).to(device)
simulator = SimulatorFunctions(device)
x_o = simulator.swing_equation(theta_true)

# prepare for sbi
simulator1, prior1 = prepare_for_sbi(simulator.swing_equation, prior)

# run sbi using 'SNPE' and 1000 simulations
inference = SNPE(simulator1, prior1, device="cuda")
posterior = inference(num_simulations=1000)

# set default value of x to x_0
posterior.set_default_x(x_o)

plotter = PlotFunctions()
plotter.plot_prior_posterior(prior, posterior, theta_true)
plotter.plot_simulated_time_frequency(simulator.swing_equation(posterior.sample((1000,))))
