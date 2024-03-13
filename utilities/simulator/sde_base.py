import numpy as np

class SDE(object):
    """
    This class implements the Euler-Maruyama method for solving stochastic differential equations (SDEs).
    """

    def __init__(self, dt=1, T=900, X0=[0.0, 0.0]):
        """
        Initializes the SDE class.

        Parameters:
        dt: Time step
        T: End time
        X0: Initial value
        """
        self.dt = dt
        self.T = T
        self.X0 = X0

    def _a(self, X, t, parameters):
        raise NotImplementedError("Subclass must implement simulate method.")
    
    def _b(self, X, t, parameters):
        raise NotImplementedError("Subclass must implement simulate method.")

    def _solve(self, a, b, parameters):
        """
        Solves the SDE using the Euler-Maruyama method.

        Parameters:
        a: Function for the deterministic part of the SDE
        b: Function for the stochastic part of the SDE

        Returns:
        X: Approximate solution of the SDE
        t: Time grid
        """
        N = int(self.T / self.dt) # Number of time steps
        d = len(self.X0) # Dimension of the system
        X = np.zeros((d, N)) # Initialization of the solution
        X[:, 0] = self.X0 # Setting the initial value
        t = np.linspace(0, self.T, N) # Time grid
        W = np.zeros((d, N)) # Initialization of the Wiener process

        for i in range(N-1):

            # Random fluctuation, corresponds to \sqrt{dt} * Z, where Z ~ N(0,1), so Z ~ N(0,dt)
            dW = np.sqrt(self.dt) * np.random.normal(size=d) # Wiener process increment

            # Euler-Maruyama step, corresponds to X_{i+1} = X_i + a(X_i, t_i) * dt + b(X_i, t_i) * dW
            X[:, i+1] = X[:, i] + self._a(X[:, i], t[i], parameters) * self.dt + self._b(X[:, i], t[i], parameters) * dW 

        return X, t
    
    def simulator(self, parameters):
        raise NotImplementedError("Subclass must implement simulate method.")
