# Rough-Volatility-SDE-Simulation
Python simulation and convergence analysis of Euler-Maruyama and Milstein schemes under Rough Volatility (H = 0.1).
# A Comparative Quantitative Analysis of Euler-Maruyama and Milstein Numerical Schemes under Rough Volatility Frameworks using Fractional Brownian Motion

This repository contains the complete Python simulation codes, numerical pipelines, and empirical verification scripts accompanying our research paper:

> **Title:** A Comparative Quantitative Analysis of Euler-Maruyama and Milstein Numerical Schemes under Rough Volatility Frameworks using Fractional Brownian Motion  
> **Authors:** Hussain Sheikh, Dr. Renuka S. Namwad  
> **Affiliation:** Department of Mathematics, Guru Nanak College of Science & Commerce, Ballarpur, Maharashtra, India  

---

## 1. Research Background & Objective
Recent empirical studies in quantitative finance demonstrate that high-frequency asset price volatility does not follow smooth standard Brownian motion ($H = 0.5$). Instead, it exhibits jagged, anti-persistent, non-Markovian behavior known as **Rough Volatility**, characterized by a Hurst parameter $H \approx 0.1$.

Because analytical closed-form solutions for non-linear Fractional Stochastic Differential Equations (fSDEs) do not exist, discrete numerical approximations are required. **The primary objective of this work is to evaluate how classical numerical integration schemes behave when shifted from standard diffusive noise ($H = 0.5$) to extreme rough noise ($H = 0.1$).**

---

## 2. Mathematical Methodology
We model the multiplicative rough asset price dynamics using the Geometric Fractional SDE:

$$dX_t = \mu X_t dt + \sigma X_t dB_t^H, \quad X_0 = x_0$$

where $B_t^H$ is a Fractional Brownian Motion with covariance structure $\mathbb{E}[B_t^H B_s^H] = \frac{1}{2}(t^{2H} + s^{2H} - |t - s|^{2H})$. We compare two discrete numerical schemes over partitioned time steps $\Delta t = T/N$:

1. **Euler-Maruyama (EM) Scheme** (Standard Strong Order $\mathcal{O}(\Delta t^{0.5})$):
   $$X_{n+1} = X_n + \mu X_n \Delta t + \sigma X_n \Delta B_n^H$$

2. **Milstein Scheme** (Higher-Order Itô-Corrected Scheme, Standard Strong Order $\mathcal{O}(\Delta t^{1.0})$):
   $$X_{n+1} = X_n + \mu X_n \Delta t + \sigma X_n \Delta B_n^H + \frac{1}{2}\sigma^2 X_n \left[ (\Delta B_n^H)^2 - \Delta t \right]$$

---

## 3. Simulation Pipeline & Repository Structure
All fractional noise increments ($\Delta B_n^H$) are generated using the exact **Davies-Harte Fast Fourier Transform (FFT) algorithm** with baseline parameters $T = 1.0$, $X_0 = 1.0$, $\mu = 0.5$, and $\sigma = 0.5$:

* **`sde_rough_vol.py`**: Simulates Fractional Brownian Motion paths ($H = 0.5$ vs. $H = 0.1$) and demonstrates how extreme rough noise causes the Euler-Maruyama trajectory to collapse into the absorbing boundary $X(t) \approx 0$ *(Reproduces **Figure 1** & **Table I/Matrix**)*.
* **`sde_euler_vs_milstein.py`**: Runs a direct pathwise comparison between the Euler-Maruyama scheme and the Milstein scheme at $H = 0.1$, proving how the second-order quadratic correction term stabilizes the asset trajectory *(Reproduces **Figure 2**)*.
* **`sde_convergence.py`**: Conducts a multi-resolution log-log strong convergence analysis across step sizes $\Delta t \in \{2^{-4}, 2^{-5}, 2^{-6}, 2^{-7}, 2^{-8}\}$ against a fine reference mesh ($N = 1024$) to compute absolute strong error $\epsilon = \mathbb{E}[|X_{\text{Coarse}}(T) - X_{\text{True}}(T)|]$ *(Reproduces **Figure 3** & **Table II**)*.

---

## 4. Key Quantitative Findings ($H = 0.1$)

| Numerical Scheme | Coarse ($\Delta t = 2^{-4}$) | Medium ($\Delta t = 2^{-6}$) | Fine ($\Delta t = 2^{-8}$) | Boundary Crash Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Euler-Maruyama** | $1.1800$ | $1.5500$ | $1.5800$ | **100.0% Failure** |
| **Milstein Scheme** | $0.9800$ | $0.7800$ | $0.6500$ | **0.0% (Stable)** |

* **Euler-Maruyama Breakdown:** Under $H = 0.1$, shrinking the step size fails to reduce error; strong convergence degrades to $\approx 0$, and $100\%$ of simulated paths crash at $\Delta t = 2^{-8}$.
* **Milstein Resilience:** The second-order quadratic correction acts as a stabilizing barrier against rough shocks, maintaining $0\%$ boundary crashes and monotonic error decay from $0.9800$ down to $0.6500$.

---

## 5. Installation & Execution
Install the required Python dependencies (`numpy` and `matplotlib`):

```bash
pip install numpy matplotlib
```

Execute the simulation scripts from your terminal:

```bash
python sde_rough_vol.py
python sde_euler_vs_milstein.py
python sde_convergence.py
```

---

## 6. Citation
If you use this code or refer to our findings in your research, please cite:

```bibtex
@article{sheikh2026rough,
  title={A Comparative Quantitative Analysis of Euler-Maruyama and Milstein Numerical Schemes under Rough Volatility Frameworks using Fractional Brownian Motion},
  author={Sheikh, Hussain and Namwad, Renuka S.},
  year={2026}
}
```

