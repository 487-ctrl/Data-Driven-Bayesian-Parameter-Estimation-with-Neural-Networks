import torch
import numpy as np

# Define the simulator function that implements the aggregated Swing equation
# The inputs are the parameters H and D, and the outputs are the frequency changes
def swing_equation(theta):

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