import torch

def _swing_equation(t, omega, theta, parameters):
    """
    Represents the swing equation in power systems.
    
    Args:
    - t (torch.Tensor): Time.
    - omega (torch.Tensor): Angular velocity.
    - theta (torch.Tensor): Rotor angle.
    - parameters (tuple): Tuple of parameters (c1, c2, P0, P1, epsilon).
    
    Returns:
    - torch.Tensor: Derivative of the angular velocity.
    """
    # Unpack the parameters
    c1, c2, P0, P1, epsilon = parameters

    # Define the swing equation incorporating time and theta
    domega_dt = c1 * omega + c2 * theta + P0 + P1 * t + epsilon

    return domega_dt

def _euler_maruyama(t_span, initial_conditions, dt, parameters, equation):
    """
    Approximates the solution of a stochastic differential equation (SDE) using Euler-Maruyama method.
    
    Args:
    - t_span (tuple): A tuple (t_start, t_end) specifying the time span.
    - initial_conditions (tuple): Tuple of initial conditions (initial_omega, initial_theta).
    - dt (float): The time step size.
    - parameters (tuple): Parameters required by the equation.
    - equation (callable): The function representing the SDE.
    
    Returns:
    - tuple: A tuple containing arrays of time points and corresponding approximated solutions.
    """
    t = torch.arange(t_span[0], t_span[1], dt)
    num_steps = len(t)
    
    # Initialize solution arrays
    omega = torch.zeros(num_steps)
    theta = torch.zeros(num_steps)
    
    # Set initial conditions
    omega[0], theta[0] = initial_conditions
    
    for i in range(num_steps - 1):
        
        # Generate Gaussian white noise for epsilon
        epsilon = torch.normal(0, torch.sqrt(torch.tensor(dt)))
        
        # Compute derivative of the solution at current time step
        domega_dt = equation(t[i], omega[i], theta[i], parameters)
        
        # Update solution using Euler-Maruyama method
        omega[i+1] = omega[i] + domega_dt * dt
        
    return t, omega

def simulator(parameters, initial_conditions=(0.0, 0.0), T=10, dt=0.1):
    """
    Simulator function for the swing equation.
    
    Args:
    - parameters (tuple): Parameters required for simulation (c1, c2, P0, P1, epsilon).
    - initial_conditions (tuple): Initial conditions for angular velocity and rotor angle (initial_omega, initial_theta).
    - T (float): Total time of simulation.
    - dt (float): Time step size.
    
    Returns:
    - torch.Tensor: Simulated angular velocities over time.
    """

    # Define simulation time span
    t_span = (0, T)  

    t, omega = _euler_maruyama(t_span, initial_conditions, dt, parameters, _swing_equation)
    
    return omega 
