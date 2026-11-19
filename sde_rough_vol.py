import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM

def simulate_fractional_sde(H, T, N, X0, mu, sigma):
    """
    Simulates a Fractional SDE using the Euler-Maruyama Scheme
    dX_t = mu * X_t * dt + sigma * X_t * dB^H_t
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    # Generate Fractional Brownian Motion paths using Hosking's or Davies-Harte method via fbm library
    f = FBM(n=N, hurst=H, length=T, method='daviesharte')
    fbm_sample = f.fbm()
    
    # Increments of fBm: dB^H = B^H_{t+1} - B^H_t
    dBH = np.diff(fbm_sample)
    
    # Array to store simulated path
    X = np.zeros(N + 1)
    X[0] = X0
    
    # Euler-Maruyama Integration loop
    for n in range(N):
        X[n+1] = X[n] + mu * X[n] * dt + sigma * X[n] * dBH[n]
        
    return t, X, fbm_sample

# --- Parameters Configuration ---
T = 1.0          # Total Time Horizon
N = 1000         # Number of Time Steps
X0 = 1.0         # Initial State
mu = 0.5         # Drift Coefficient
sigma = 0.3      # Volatility (Noise Intensity)

# 1. Simulate for Standard/Smooth Case (H = 0.5)
t_standard, X_standard, fbm_standard = simulate_fractional_sde(H=0.5, T=T, N=N, X0=X0, mu=mu, sigma=sigma)

# 2. Simulate for Extreme Rough Volatility Case (H = 0.1)
t_rough, X_rough, fbm_rough = simulate_fractional_sde(H=0.1, T=T, N=N, X0=X0, mu=mu, sigma=sigma)

# --- Plotting the Results ---
plt.figure(figsize=(14, 6))

# Plotting the Underlying Noise (fBm Paths)
plt.subplot(1, 2, 1)
plt.plot(t_standard, fbm_standard, label='Standard Noise (H = 0.5)', color='blue', alpha=0.7)
plt.plot(t_rough, fbm_rough, label='Rough Noise (H = 0.1)', color='red', alpha=0.7)
plt.title('Fractional Brownian Motion Paths ($B^H_t$)')
plt.xlabel('Time (t)')
plt.ylabel('Noise Trajectory')
plt.legend()
plt.grid(True)

# Plotting the SDE Trajectories
plt.subplot(1, 2, 2)
plt.plot(t_standard, X_standard, label='SDE Path with H = 0.5', color='darkblue')
plt.plot(t_rough, X_rough, label='SDE Path with H = 0.1 (Rough Vol)', color='darkred')
plt.title('Euler-Maruyama Simulation of Fractional SDE')
plt.xlabel('Time (t)')
plt.ylabel('Asset Price / State X(t)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
