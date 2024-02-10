import numpy as np
# from simulator.utils.simulator_utils import check_input_size

class Swing(object):
    """
    A base class for the swing equation of a synchronous generator.
    Developed by [Developer's Name].

    Attributes:
        M (float): Moment of inertia of the rotor.
        P_m0 (float): Constant mechanical power.
        K_p (float): Proportionality factor for the mechanical power.
        f_0 (float): Nominal frequency.
    """

    # @check_input_size(4)
    def __init__(self, ξ):
        """
        Initializes the parameters of the swing equation.

        Args:
            ξ (tuple): A tuple containing the parameters M, P_m0, K_p, and f_0.
        """
        
        # Unpack the parameter vector
        self.M, self.P_m0, self.K_p, self.f_0 = ξ

    def P_e(self, delta):
        """
        Returns the electrical power as a function of the angle (abstract method).

        Args:
            delta (float): The angle.

        Raises:
            NotImplementedError: This method must be overridden in a subclass.
        """
        raise NotImplementedError

    def D(self, omega):
        """
        Returns the damping force as a function of the angular velocity (abstract method).

        Args:
            omega (float): The angular velocity.

        Raises:
            NotImplementedError: This method must be overridden in a subclass.
        """
        raise NotImplementedError

    def T(self, delta, theta1, theta2):
        """
        Returns the torque on the rotor as a function of the angles (abstract method).

        Args:
            delta (float): The angle.
            theta1 (float): The first angle.
            theta2 (float): The second angle.

        Raises:
            NotImplementedError: This method must be overridden in a subclass.
        """
        raise NotImplementedError

    def dydt(self, t, y):
        """
        Returns the derivatives of the state variables as a function of time and state.

        Args:
            t (float): The time.
            y (list): The state variables.

        Returns:
            list: The derivatives of the state variables.
        """
        delta, omega, theta1, theta2 = y

        # Ensure omega is within reasonable bounds to prevent overflow
        omega = np.clip(omega, -1e6, 1e6)
        f = 2 * np.pi * omega
        P_e = self.P_e(delta)
        D = self.D(omega)
        T = self.T(delta, theta1, theta2)

        # Check for division by zero and handle appropriately
        dydt = [omega, 0, 0, 0] if self.M == 0 else [omega, (self.P_m0 + self.K_p * (f - self.f_0) - P_e - D - T) / self.M, 0, 0]
        return dydt

    def solve(self, y0, t_span, t_eval):
        """
        Solves the swing equation for the given initial conditions and time points using the Euler-Maruyama method.

        The Euler-Maruyama method is a numerical method for the approximation of solutions of stochastic differential equations.
        It is a simple method that is easy to implement but may not be accurate for large time steps or for problems that are not well-behaved.

        Args:
            y0 (list): The initial conditions.
            t_span (tuple): A tuple containing the start and end time.
            t_eval (list): The time points at which the solution should be evaluated.

        Returns:
            np.ndarray: The solution of the swing equation.
        """
        num_steps = len(t_eval)
        dt = t_eval[1] - t_eval[0]
        sol = np.zeros((4, num_steps))
        sol[:, 0] = y0
        for i in range(1, num_steps):
            t_prev, y_prev = t_eval[i - 1], sol[:, i - 1]
            dydt = self.dydt(t_prev, y_prev)

            # Generate a normally distributed random number scaled by the square root of dt
            dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(4,))

            # Update the solution using the Euler-Maruyama method
            sol[:, i] = y_prev + np.array(dydt) * dt + dW
        return sol


    def __str__(self):
        """
        Returns a string representation of the class instance.

        Returns:
            str: A string representation of the class instance.
        """
        return f"{self.__class__.__name__} with parameters M={self.M}, P_m0={self.P_m0}, K_p={self.K_p}, f_0={self.f_0}"