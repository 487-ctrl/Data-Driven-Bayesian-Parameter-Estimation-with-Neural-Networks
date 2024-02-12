import numpy as np
import torch

# Formulation with 5 parameters c_1, c_2, P_0, P_1 and epsilon
def swing_equation(theta, dt=0.01, T=10):
    """
    This function implements the swing equation using the Euler-Maruyama method 
    to solve a second-order stochastic differential equation.

    Parameters:
        theta (tuple): A tuple containing the parameters c_1, c_2, P_const, and epsilon.
        dt (float): The time step for the Euler-Maruyama method. Default is 0.01.
        T (float): The total simulation time. Default is 10.

    Returns:
        torch.Tensor: A tensor containing the frequency changes.
    """
    # Unpack the parameters
    c1, c2, P0, P1, epsilon = theta

    # Define the time span
    t_span = np.arange(0, T, dt)

    # Initialize the state variables
    deltaomega = np.zeros([len(t_span), 2])

    # Generate random numbers for the Wiener process
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=[t_span.size, 1])

    # Define the system matrix
    A = np.array([[1, dt], [-dt * c2, 1 - dt * c1]])

    # Run the simulation loop
    for i in range(1, len(t_span)):
        # Update the state variables using the Euler-Maruyama method
        deltaomega[i] = np.matmul(A, deltaomega[i-1]) + np.array([0, dt * P0 + P1 + epsilon * dW[i, 0]])

    # Return the second state variable as a tensor
    return torch.from_numpy(deltaomega[:, 1])

# Alternative formulation with 4 parameters c_1, c_2, P_const, and epsilon
def swing_equation_alt(theta, dt=0.01, T=10):
    """
    This function implements the swing equation using the Euler-Maruyama method 
    to solve a second-order stochastic differential equation.

    Parameters:
    theta (tuple): A tuple containing the parameters c_1, c_2, P_const, and epsilon.
    dt (float): The time step for the Euler-Maruyama method. Default is 0.01.
    T (float): The total simulation time. Default is 10.

    Returns:
    torch.Tensor: A tensor containing the frequency changes.
    """
    # Unpack the parameters
    c_1, c_2, P_const, epsilon = theta

    # Define the time span
    t_span = np.arange(0, T, dt)

    # Initialize the state variables
    deltaomega = np.zeros([len(t_span), 2])

    # Generate random numbers for the Wiener process
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=[t_span.size, 1])

    # Define the system matrix
    A = np.array([[1, dt], [-dt * c_2, 1 - dt * c_1]])

    # Run the simulation loop
    for i in range(1, len(t_span)):

        # Update the state variables using the Euler-Maruyama method
        deltaomega[i] = np.matmul(A, deltaomega[i-1]) + np.array([0, dt * P_const + epsilon * dW[i, 0]])

    # Return the second state variable as a tensor
    return torch.from_numpy(deltaomega[:, 1])
