import torch

# define the swing equation as simulator
def swing_equation(theta):

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

