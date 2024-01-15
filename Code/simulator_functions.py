import torch

class SimulatorFunctions:
    def __init__(self):
        pass

    # define the swing equation as simulator
    def swing_equation(self, theta):

        # select parameters from input vector
        M = theta[:,0] # Trägheitskonstante
        Pm = theta[:,1] # Mechanische Leistung
        Pe = theta[:,2] # Elektrische Leistung

        # definition of starting criteria
        delta0 = 0.0 # Anfangswinkel
        omega0 = 0.0 # Anfangsgeschwindigkeit

        # definition of timeline
        t_max = 10.0 # Maximale Zeit
        dt = 0.01 # Zeitschritt
        t = torch.arange(0, t_max, dt) # Zeitvektor

        # initialize output vectors
        delta = torch.zeros((len(theta), len(t))) # Winkelverlauf
        omega = torch.zeros((len(theta), len(t))) # Geschwindigkeitsverlauf

        # solve swing-equation using the euler method (https://en.wikipedia.org/wiki/Euler_method)
        for i in range(len(t)-1):

            # calculate acceleration
            alpha = (Pm - Pe - M * omega[:,i]) / M

            # update speed and angle 
            omega[:,i+1] = omega[:,i] + alpha * dt
            delta[:,i+1] = delta[:,i] + omega[:,i] * dt

        # return speed and angle
        return torch.stack((delta[:,-1], omega[:,-1]), dim=1)
    
    # define some linear simulator moddeling a simple linear relationship: y = mx + c
    def linear(self, theta):

        # theta is a tensor of parameters with shape (num_samples, num_parameters)
        m = theta[:, 0]  # slope
        c = theta[:, 1]  # intercept
        x = torch.linspace(-10, 10, 100)  # input values
        y = m * x + c  # output values
        return y

    # define some quadratic simulator moddeling a simple quadratic relationship: y = ax^2 + bx + c
    def quadratic(self, theta):

        # theta is a tensor of parameters with shape (num_samples, num_parameters)
        a = theta[:, 0]  # quadratic term coefficient
        b = theta[:, 1]  # linear term coefficient
        c = theta[:, 2]  # constant term
        x = torch.linspace(-10, 10, 100)  # input values
        y = a * x**2 + b * x + c  # output values
        return y

    # define some exponential simulator moddeling a simple exponential relationship: y = a * e^(bx)
    def exponential(self, theta):

        # theta is a tensor of parameters with shape (num_samples, num_parameters)
        a = theta[:, 0]  # coefficient
        b = theta[:, 1]  # exponent
        x = torch.linspace(-10, 10, 100)  # input values
        y = a * torch.exp(b * x)  # output values
        return y


