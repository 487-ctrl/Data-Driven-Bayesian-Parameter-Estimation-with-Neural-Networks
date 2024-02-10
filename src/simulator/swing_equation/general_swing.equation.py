import numpy as np
from swing_equation_base import Swing
# from simulator.utils.simulator_utils import check_input_size

class GeneralSwing(Swing):
    """
    A subclass of the Swing class that models a synchronous generator with a sinusoidal dependence of the electrical power on the angle, a proportional damping force, and a torque from two pendulums.

    Attributes:
        P_max (float): Maximum electrical power.
        d (float): Damping constant.
        l1 (float): Length of the first pendulum.
        l2 (float): Length of the second pendulum.
    """

    # @check_input_size(8)
    def __init__(self, ξ):
        """
        Initializes the parameters of the general swing equation.

        Args:
            ξ (tuple): A tuple containing the parameters M, P_m0, K_p, f_0, P_max, d, l1, and l2.
        """
        
        # Unpack the parameter vector
        M, P_m0, K_p, f_0, P_max, d, l1, l2 = ξ

        # Call the constructor of the base class with the first four parameters
        super().__init__(ξ[:4])

        self.P_max = P_max  # Maximum electrical power
        self.d = d  # Damping constant
        self.l1 = l1  # Length of the first pendulum
        self.l2 = l2  # Length of the second pendulum

    def P_e(self, delta):
        """
        Returns the electrical power as a sinusoidal function of the angle.

        Args:
            delta (float): The angle.

        Returns:
            float: The electrical power.
        """
        return self.P_max * np.sin(delta)

    def D(self, omega):
        """
        Returns the damping force as proportional to the angular velocity.

        Args:
            omega (float): The angular velocity.

        Returns:
            float: The damping force.
        """
        return self.d * omega

    def T(self, delta, theta1, theta2):
        """
        Returns the torque on the rotor as a function of the angles of the pendulums.

        Args:
            delta (float): The angle.
            theta1 (float): The first angle.
            theta2 (float): The second angle.

        Returns:
            float: The torque on the rotor.
        """
        x1 = self.l1 * np.sin(theta1)  # x-coordinate of the first pendulum
        y1 = -self.l1 * np.cos(theta1)  # y-coordinate of the first pendulum
        x2 = x1 + self.l2 * np.sin(theta2)  # x-coordinate of the second pendulum
        y2 = y1 - self.l2 * np.cos(theta2)  # y-coordinate of the second pendulum
        F_x = -self.P_e(delta) * x2  # Horizontal force on the rotor
        F_y = -self.P_e(delta) * y2  # Vertical force on the rotor
        T = F_x * y2 - F_y * x2  # Torque on the rotor
        return T

    def __str__(self):
        """
        Returns a string representation of the class instance.

        Returns:
            str: A string representation of the class instance.
        """
        return f"{self.__class__.__name__} with parameters M={self.M}, P_m0={self.P_m0}, K_p={self.K_p}, f_0={self.f_0}, P_max={self.P_max}, d={self.d}, l1={self.l1}, l2={self.l2}"
