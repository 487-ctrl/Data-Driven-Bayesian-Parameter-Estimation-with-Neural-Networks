import numpy as np
from swing_equation_base import Swing
# from simulator.utils.simulator_utils import check_input_size

class ClassicalSwing(Swing):
    """
    A subclass of the Swing class that models a synchronous generator with a sinusoidal dependence of the electrical power on the angle.

    Attributes:
        P_max (float): Maximum electrical power.
    """

    # @check_input_size(5)
    def __init__(self, ξ):
        """
        Initializes the parameters of the classical swing equation.

        Args:
            ξ (tuple): A tuple containing the parameters M, P_m0, K_p, f_0, and P_max.
        """
        
        # Unpack the parameter vector
        M, P_m0, K_p, f_0, P_max = ξ

        # Call the constructor of the base class with the first four parameters
        super().__init__(ξ[:4])

        self.P_max = P_max  # Maximum electrical power

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
        Returns the damping force as zero.

        Args:
            omega (float): The angular velocity.

        Returns:
            int: The damping force, which is always zero in this model.
        """
        return 0

    def T(self, delta, theta1, theta2):
        """
        Returns the torque on the rotor as zero.

        Args:
            delta (float): The angle.
            theta1 (float): The first angle.
            theta2 (float): The second angle.

        Returns:
            int: The torque, which is always zero in this model.
        """
        return 0
