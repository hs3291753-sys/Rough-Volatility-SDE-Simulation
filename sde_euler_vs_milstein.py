import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM

def compare_schemes(H, T, N, X0, mu, sigma):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    # Generate a common fractional Brownian motion (fBm) path for a fair comparison
    f = FBM(n=N, hurst=H, length=T, method='daviesharte')
    fbm_sample = f.fbm()
    dBH = np.diff(fbm_sample)
    
    # Arrays to store simulated solution trajectories
    X_euler = np.zeros(N + 1)
    X_milstein = np.zeros(N + 1)
    
    X_euler[0] = X0
    X_milstein[0] = X0
    
    # Numerical Integration Loop
    for n in range(N):
        # 1. Euler-Maruyama Scheme
        X_euler[n+1] = X_euler[n] + mu * X_euler[n] * dt + sigma * X_euler[n] * dBH[n]
        
        # 2. Milstein Scheme (incorporating the Itô correction term structure)
        # For dX = mu*X*dt + sigma*X*dBH, the diffusion derivative term b*(db/dx) equals sigma^2 * X
        correction = 0.5 * (sigma**2) * X_milstein[n] * (dBH[n]**2 - dt)
        X_milstein[n+1] = X_milstein[n] + mu * X_milstein[n] * dt + sigma * X_milstein[n] * dBH[n] + correction
        
    return t, X_euler, X_milstein

# --- Model Parameters ---
T = 1.0       # Total time horizon
N = 1000      # Number of discretization time steps
X0 = 1.0      # Initial state value
mu = 0.5      # Drift coefficient
sigma = 0.5   # Volatility / Noise intensity

# Simulate under extreme rough volatility conditions (H = 0.1)
t, euler_path, milstein_path = compare_schemes(H=0.1, T=T, N=N, X0=X0, mu=mu, sigma=sigma)

# --- Graphical Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(t, euler_path, label='Euler-Maruyama Scheme (H = 0.1)', color='red', alpha=0.8)
plt.plot(t, milstein_path, label='Milstein Scheme (H = 0.1)', color='green', alpha=0.8)
plt.title('Trajectory Comparison under Extreme Rough Volatility ($H = 0.1$)', fontsize=12, fontweight='bold')
plt.xlabel('Time ($t$)', fontsize=11)
plt.ylabel('Asset Price / State $X(t)$', fontsize=11)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()