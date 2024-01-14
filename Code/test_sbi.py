# Importieren Sie die sbi-Bibliothek und andere nützliche Pakete
import torch
from torch.distributions import Uniform, Independent
import numpy as np
import matplotlib.pyplot as plt
from sbi.inference import infer

# Definieren Sie die Swing-Gleichung als Simulator
def swing_equation(theta):
    # Extrahieren Sie die Parameter aus dem Eingabevektor
    M = theta[:,0] # Trägheitskonstante
    Pm = theta[:,1] # Mechanische Leistung
    Pe = theta[:,2] # Elektrische Leistung
    # Definieren Sie die Anfangsbedingungen
    delta0 = 0.0 # Anfangswinkel
    omega0 = 0.0 # Anfangsgeschwindigkeit
    # Definieren Sie die Zeitachse
    t_max = 10.0 # Maximale Zeit
    dt = 0.01 # Zeitschritt
    t = torch.arange(0, t_max, dt) # Zeitvektor
    # Initialisieren Sie die Ausgabevektoren
    delta = torch.zeros((len(theta), len(t))) # Winkelverlauf
    omega = torch.zeros((len(theta), len(t))) # Geschwindigkeitsverlauf
    # Lösen Sie die Swing-Gleichung mit dem Euler-Verfahren
    for i in range(len(t)-1):
        # Berechnen Sie die Beschleunigung
        alpha = (Pm - Pe - M * omega[:,i]) / M
        # Aktualisieren Sie die Geschwindigkeit und den Winkel
        omega[:,i+1] = omega[:,i] + alpha * dt
        delta[:,i+1] = delta[:,i] + omega[:,i] * dt
    # Geben Sie den Winkel und die Geschwindigkeit zurück
    return torch.stack((delta[:,-1], omega[:,-1]), dim=1)

# Definieren Sie die Prior-Verteilung der Parameter
prior_min = torch.tensor([0.1, 0.0, 0.0]) # Minimale Werte der Parameter
prior_max = torch.tensor([1.0, 1.0, 1.0]) # Maximale Werte der Parameter
prior = Independent(Uniform(prior_min, prior_max), 1) # Gleichverteilung

# Generieren Sie einige synthetische Beobachtungen
theta_true = torch.tensor([[0.5, 0.8, 0.6]]) # Wahre Parameterwerte
x_o = swing_equation(theta_true) # Beobachtete Ausgabe

# Führen Sie die simulationsbasierte Inferenz durch
# Verwenden Sie SNPE als Methode und 1000 Simulationen
posterior = infer(swing_equation, prior, method='SNPE', num_simulations=1000)

# Setzen Sie den Standardwert für x
posterior.set_default_x(x_o)

# Zeichnen Sie die posteriori Verteilung
samples = posterior.sample((10000,)) # Ziehen Sie 10000 Stichproben aus der posteriori Verteilung

labels = ['M', 'Pm', 'Pe'] # Beschriftungen der Parameter
# Zeichnen Sie die Prior- und Posteriori-Verteilungen
fig, axes = plt.subplots(2,3, figsize=(12,8)) # Erstellen Sie eine Abbildung mit sechs Unterabbildungen
for i in range(3):
    # Ziehen Sie 10000 Stichproben aus der Prior-Verteilung
    prior_samples = prior.sample((10000,))[:,i]
    # Zeichnen Sie ein Histogramm der Prior-Verteilung für jeden Parameter
    axes[0,i].hist(prior_samples, bins=20, density=True)
    # Beschriften Sie die Achsen
    axes[0,i].set_xlabel(labels[i])
    axes[0,i].set_ylabel('Density')
    # Fügen Sie eine Überschrift hinzu
    axes[0,i].set_title('Prior distribution')

    # Zeichnen Sie ein Histogramm der Posteriori-Verteilung für jeden Parameter
    axes[1,i].hist(samples[:,i], bins=20, density=True)
    # Zeichnen Sie eine vertikale Linie für den wahren Parameterwert
    axes[1,i].axvline(theta_true[0][i].item(), color='red', label='True value')
    # Beschriften Sie die Achsen
    axes[1,i].set_xlabel(labels[i])
    axes[1,i].set_ylabel('Density')
    # Fügen Sie eine Legende hinzu
    axes[1,i].legend()
    # Fügen Sie eine Überschrift hinzu
    axes[1,i].set_title('Posterior distribution')

# Zeigen Sie die Abbildung an
plt.tight_layout()
plt.show()

# Ziehen Sie einige Stichproben aus der Posteriori-Verteilung
theta_samples = posterior.sample((1000,))

# Berechnen Sie die Ausgabe des Simulators für jede Stichprobe
simulator_outputs = swing_equation(theta_samples)

# Extrahieren Sie delta und omega aus den Simulatorausgaben
delta_samples = simulator_outputs[:, 0].cpu().numpy()
omega_samples = simulator_outputs[:, 1].cpu().numpy()

# Erstellen Sie eine Zeitachse
t_max = 10.0 # Maximale Zeit
dt = 0.01 # Zeitschritt
t = np.arange(0, t_max, dt) # Zeitvektor

# Plotten Sie delta und omega über die Zeit
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, delta_samples.T, color='blue', alpha=0.1)
plt.xlabel('Time')
plt.ylabel('delta')

plt.subplot(2, 1, 2)
plt.plot(t, omega_samples.T, color='blue', alpha=0.1)
plt.xlabel('Time')
plt.ylabel('omega')

plt.tight_layout()
plt.show()
