import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM

def calculate_convergence():
    # --- System Parameters ---
    T = 1.0          # Total time horizon
    X0 = 1.0         # Initial value
    mu = 0.5         # Drift coefficient
    sigma = 0.5      # Volatility coefficient
    H = 0.1          # Extreme Rough Volatility parameter
    
    # List of different grid sizes to test the error reduction rate
    N_list = [16, 32, 64, 128, 256]
    
    euler_errors = []
    milstein_errors = []
    
    # 1. Generate a high-resolution reference noise path (Fine Grid)
    # This acts as our proxy for the "True Continuous Solution"
    N_fine = 1024
    f = FBM(n=N_fine, hurst=H, length=T, method='daviesharte')
    fbm_fine = np.array(f.fbm())
    
    # Simulate the "True" reference solution using fine grid Milstein scheme
    X_true = X0
    dt_fine = T / N_fine
    for n in range(N_fine):
        dBH = fbm_fine[n+1] - fbm_fine[n]
        X_true = X_true + mu * X_true * dt_fine + sigma * X_true * dBH + 0.5 * (sigma**2) * X_true * (dBH**2 - dt_fine)
    
    # 2. Compute approximations over coarser grids to evaluate convergence
    for N in N_list:
        dt = T / N
        factor = N_fine // N  # Mapping coarse steps to the underlying fine noise path
        
        X_euler = X0
        X_milstein = X0
        
        for n in range(N):
            # Extracting coarse Brownian increments from the identical fine noise path
            dBH = fbm_fine[(n+1)*factor] - fbm_fine[n*factor]
            
            # --- Euler-Maruyama Integration Scheme ---
            X_euler = X_euler + mu * X_euler * dt + sigma * X_euler * dBH
            
            # --- Milstein Integration Scheme (with Itô correction term) ---
            correction = 0.5 * (sigma**2) * X_milstein * (dBH**2 - dt)
            X_milstein = X_milstein + mu * X_milstein * dt + sigma * X_milstein * dBH + correction
            
        # Store absolute strong errors at final time T
        euler_errors.append(np.abs(X_euler - X_true))
        milstein_errors.append(np.abs(X_milstein - X_true))
        
    return N_list, euler_errors, milstein_errors

# --- Execution and Visualization Engine ---
N_list, euler_err, milstein_err = calculate_convergence()
dts = [1.0 / N for N in N_list]

plt.figure(figsize=(9, 7))

# Plot empirical absolute errors computed from our simulations
plt.loglog(dts, euler_err, '-or', linewidth=2, label='Euler-Maruyama Empirical Error')
plt.loglog(dts, milstein_err, '-og', linewidth=2, label='Milstein Empirical Error')

# Plot mathematical reference lines representing theoretical slopes 0.5 and 1.0
plt.loglog(dts, [0.3 * np.sqrt(d) for d in dts], '--k', alpha=0.5, label='Theoretical Reference: Slope 0.5 (Standard Euler)')
plt.loglog(dts, [0.3 * d for d in dts], '-.k', alpha=0.5, label='Theoretical Reference: Slope 1.0 (Standard Milstein)')

# Graphical Formatting for Academic Publications
plt.xlabel('Time Step Size ($\Delta t$)', fontsize=12)
plt.ylabel('Strong Absolute Error $|X_{Coarse}(T) - X_{True}(T)|$', fontsize=12)
plt.title('Log-Log Strong Convergence Profile under Rough Volatility ($H=0.1$)', fontsize=13, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
