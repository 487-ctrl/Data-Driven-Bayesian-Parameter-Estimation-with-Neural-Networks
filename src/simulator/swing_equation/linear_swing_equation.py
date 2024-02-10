from swing_equation_base import Swing
# from simulator.utils.simulator_utils import check_input_size

class LinearSwing(Swing):
    """
    A subclass of the Swing class that models a synchronous generator with a linear dependence of the electrical power on the angle.

    Attributes:
        k (float): Constant for the linear swing equation.
    """

    # @check_input_size(5)
    def __init__(self, ξ):
        """
        Initializes the parameters of the linear swing equation.

        Args:
            ξ (tuple): A tuple containing the parameters M, P_m0, K_p, f_0, and k.
        """
        
        # Unpack the parameter vector
        M, P_m0, K_p, f_0, k = ξ

        # Call the constructor of the base class with the first four parameters
        super().__init__(ξ[:4])

        self.k = k  # Constant for the linear swing equation

    def P_e(self, delta):
        """
        Returns the electrical power as a linear function of the angle.

        Args:
            delta (float): The angle.

        Returns:
            float: The electrical power.
        """
        return self.k * delta

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
