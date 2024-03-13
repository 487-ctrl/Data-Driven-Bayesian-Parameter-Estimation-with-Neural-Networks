import numpy as np
import torch
from utilities.simulator.sde_base import SDE

class Swing(SDE):
    """
    This class implements the Euler-Maruyama method for solving the given system of nonlinear or linear stochastic differential equations (SDEs).
    """

    def __init__(self, dt=1, T=900, base_frequency=50, X0=[0.0, 0.0], f=lambda x, y: -x * y, g=lambda x, y: -x * y):
        """
        Initializes the Swing class.

        Parameters:
        dt: Time step
        T: End time
        X0: Initial value
        f: Nonlinear function for the deterministic part of the SDE
        g: Nonlinear function for the deterministic part of the SDE
        """
        super().__init__(dt, T, X0)
        self.f = f
        self.g = g
        self.base_frequency = base_frequency

    def _a(self, X, t, parameters):
        """
        Function for the deterministic part of the SDE.
        The deterministic part of the SDE is given by the formula:
        dω/dt = f(c1, ω) + g(c2, θ) + P0 + t * P1
        dθ/dt = ω
        where f(c1, ω) is a nonlinear function of ω and c1, and g(c2, θ) respectively.
        """
        c1, c2, P0, P1, epsilon = parameters
        omega, theta = X
        return np.array([self.f(c1, omega) + self.g(c2, theta) + P0 + t * P1, omega])

    def _b(self, X, t, parameters):
        """
        Function for the stochastic part of the SDE.
        The stochastic part of the SDE is given by the formula:
        dω/dt = ε
        dθ/dt = 0
        """
        c1, c2, P0, P1, epsilon = parameters
        return np.array([epsilon, 0])

    def simulator(self, parameters):
        """
        Simulates the SDE using the Euler-Maruyama method.

        Parameters:
        parameters: Parameters of the SDE

        Returns:
        X: Approximate solution of the SDE
        t: Time grid
        """
        # The _solve method is called with the deterministic and stochastic parts of the SDE (_a and _b methods)
        # and the parameters of the SDE. It returns the approximate solution of the SDE and the time grid.
        (omega, theta), t = self._solve(lambda X, t: self._a(X, t, parameters), lambda X, t: self._b(X, t, parameters), parameters)
        
        # The simulator method returns the frequency deviations omega shifted by the base frequency.
        return torch.from_numpy(omega + self.base_frequency)
