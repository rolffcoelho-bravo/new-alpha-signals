"""
Advanced QuantOps Methodologies: Streaming PGD, Graph Regularization, and Regime Switching.

This script implements the cutting-edge extensions to the Dynamic Alpha Operator:
1. Online / Streaming PGD: Recursive updates using exponential forgetting (gamma).
2. Turnover Penalty: Smooths operator transitions to optimize net-of-cost capacity.
3. Graph Laplacian Regularization: Shares alpha across economically linked assets (sectors).
4. Regime-Switching: Adapts SVT and BST hyperparameters based on market volatility.
5. Alt-Data Simulation: Robustness check against high-dimensional, low-signal alternative data.

Designed for Institutional Applicability, Methodological Novelty, and Publishability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2026)

# --- 1. Online Dynamic Alpha Operator Estimator ---

class OnlineDynamicAlphaEstimator:
    def __init__(self, N, P, 
                 lambda_star_base=0.01, lambda_grp_base=0.01, 
                 lambda_turnover=0.1, lambda_graph=0.005,
                 gamma=0.99, lr=0.001, max_iter=50, tol=1e-5):
        """
        Args:
            gamma: Exponential forgetting factor (e.g., 0.99).
            lambda_turnover: Penalty on Euclidean distance from previous day's operator.
            lambda_graph: Penalty coefficient for Graph Laplacian regularization.
        """
        self.N = N
        self.P = P
        self.NP = N * P
        self.lambda_star_base = lambda_star_base
        self.lambda_grp_base = lambda_grp_base
        self.lambda_turnover = lambda_turnover
        self.lambda_graph = lambda_graph
        self.gamma = gamma
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        
        # Sufficient Statistics for Online Gradient
        self.R_xx = np.zeros((self.NP, self.NP))
        self.R_yx = np.zeros((self.N, self.NP))
        
        # Operator State
        self.A_hat_ = np.zeros((self.N, self.NP))
        self.A_prev_ = np.zeros((self.N, self.NP))
        
        # Structural Information
        self.L_graph = None # Graph Laplacian
        self.current_regime = "Normal"
        
    def set_graph_laplacian(self, L):
        self.L_graph = L
        
    def update_online(self, y_t, x_t, market_vol_t):
        """
        Processes a single new observation (y_t, x_t) and updates the operator A online.
        """
        # 1. Update Sufficient Statistics with Exponential Forgetting
        x_t_col = x_t.reshape(-1, 1) # NP x 1
        y_t_col = y_t.reshape(-1, 1) # N x 1
        
        self.R_xx = self.gamma * self.R_xx + (x_t_col @ x_t_col.T)
        self.R_yx = self.gamma * self.R_yx + (y_t_col @ x_t_col.T)
        
        # 2. Regime-Switching Hyperparameters
        if market_vol_t > 0.02: # High volatility regime (e.g. > 30% annualized)
            self.current_regime = "High_Vol"
            # In crises, correlations spike (rank drops), so we penalize rank more heavily
            lambda_star = self.lambda_star_base * 2.0
            lambda_grp = self.lambda_grp_base * 1.5
        else:
            self.current_regime = "Normal"
            lambda_star = self.lambda_star_base
            lambda_grp = self.lambda_grp_base
            
        # 3. Proximal Gradient Descent Iteration
        A = self.A_hat_.copy()
        best_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # --- Gradient Step ---
            # Data gradient: A * R_xx - R_yx
            grad_data = (A @ self.R_xx) - self.R_yx
            
            # Turnover Penalty Gradient
            grad_turnover = self.lambda_turnover * (A - self.A_prev_)
            
            # Graph Laplacian Gradient
            if self.L_graph is not None:
                grad_graph = self.lambda_graph * (self.L_graph @ A)
            else:
                grad_graph = 0.0
                
            total_grad = grad_data + grad_turnover + grad_graph
            
            # Normalized gradient step
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm > 1e-8:
                A_next = A - self.lr * (total_grad / grad_norm)
            else:
                A_next = A - self.lr * total_grad
                
            # --- Proximal Step: Group Lasso (BST) ---
            for p in range(self.P):
                indices = np.arange(p, self.NP, self.P)
                A_sub = A_next[:, indices]
                norm_val = np.linalg.norm(A_sub, ord='fro')
                if norm_val > 1e-8:
                    scale = np.maximum(1.0 - self.lr * lambda_grp / norm_val, 0.0)
                    A_next[:, indices] = A_sub * scale
                else:
                    A_next[:, indices] = 0.0
                    
            # --- Proximal Step: Nuclear Norm (SVT) ---
            U, S, Vt = np.linalg.svd(A_next, full_matrices=False)
            S_thresh = np.maximum(S - self.lr * lambda_star, 0.0)
            A_next = U @ np.diag(S_thresh) @ Vt
            
            # Convergence Check based on Frobenius change
            change = np.linalg.norm(A_next - A, ord='fro')
            if iteration > 5 and change < self.tol:
                A = A_next
                break
            A = A_next
            
        # Update state
        self.A_prev_ = self.A_hat_.copy()
        self.A_hat_ = A
        return self.A_hat_
        
    def predict(self, x_t):
        return self.A_hat_ @ x_t

# --- 2. Simulation and Execution ---

def build_sector_laplacian(N, num_sectors=4):
    """
    Constructs a Graph Laplacian L = D - W based on sector peers.
    Assets in the same sector are linked (w_ij = 1).
    """
    assets_per_sector = N // num_sectors
    W = np.zeros((N, N))
    for s in range(num_sectors):
        start = s * assets_per_sector
        end = start + assets_per_sector
        W[start:end, start:end] = 1.0
        
    np.fill_diagonal(W, 0.0) # No self-loops
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Normalize Laplacian for stability
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    L_norm = d_inv_sqrt @ L @ d_inv_sqrt
    return L_norm

def generate_alt_data_environment(T=1000, N=20, P=20):
    """
    Simulates a high-dimensional, low-SNR Alternative Data environment.
    N=20 assets, P=20 signal families (e.g., NLP embeddings, satellite features).
    Total features NP = 400.
    """
    print(f"Generating Alternative Data Environment ({N} assets, {P} signals = {N*P} features)...")
    
    # Benchmark Risk Factors (Market + Sector)
    market_factor = np.random.normal(0.0, 0.012, (T, 1))
    betas = np.random.uniform(0.5, 1.5, (N, 1))
    
    # Sector shocks
    num_sectors = 4
    assets_per_sector = N // num_sectors
    sector_factors = np.random.normal(0.0, 0.008, (T, num_sectors))
    
    returns = market_factor @ betas.T + np.random.normal(0.0, 0.015, (T, N))
    for s in range(num_sectors):
        start = s * assets_per_sector
        end = start + assets_per_sector
        returns[:, start:end] += sector_factors[:, s:s+1]
        
    # High-Dimensional Signals (Alt-Data)
    signals = np.random.normal(0.0, 1.0, (T, N, P))
    
    # Only 2 out of 20 signal families are actually predictive (Causal Alpha)
    # The rest are spurious noise
    true_alpha_dir_1 = np.linspace(-0.5, 0.5, N)
    true_alpha_dir_2 = np.random.choice([-0.2, 0.2], size=N)
    
    alpha_returns = (signals[:, :, 0] @ true_alpha_dir_1.reshape(-1, 1)) * 0.005 + \
                    (signals[:, :, 1] @ true_alpha_dir_2.reshape(-1, 1)) * 0.003
                    
    returns += alpha_returns
    
    # Flatten signals to size T x NP
    signals_flat = np.zeros((T, N * P))
    for t in range(T):
        signals_flat[t] = signals[t].flatten()
        
    return returns, market_factor, betas, signals_flat

def main():
    N, P = 20, 20
    NP = N * P
    
    returns, factors, betas, signals = generate_alt_data_environment(T=1000, N=N, P=P)
    
    # DML Step: Orthogonalize Returns (Causal Inference Isolation)
    print("Executing Double Machine Learning (DML) Phase 1: Orthogonalizing Returns...")
    Y_perp = np.zeros_like(returns)
    I_N = np.eye(N)
    for t in range(len(returns)):
        B_t = betas
        B_inv = np.linalg.inv(B_t.T @ B_t)
        M_B = I_N - B_t @ B_inv @ B_t.T
        Y_perp[t] = M_B @ returns[t]
        
    # Setup Graph Laplacian
    L_graph = build_sector_laplacian(N, num_sectors=4)
    
    # Initialize Online Estimator
    estimator = OnlineDynamicAlphaEstimator(
        N=N, P=P, 
        lambda_star_base=0.01, 
        lambda_grp_base=0.01,
        lambda_turnover=0.05, # Institutional capacity tuning
        lambda_graph=0.002,   # Sector peer propagation
        gamma=0.98            # Memory half-life of ~34 days
    )
    estimator.set_graph_laplacian(L_graph)
    
    # Pre-train / Burn-in on first 200 days
    burn_in = 200
    print(f"Executing Online PGD Burn-in Phase (T=0 to {burn_in})...")
    volatility_window = []
    
    for t in range(burn_in):
        y_t, x_t = Y_perp[t], signals[t]
        # Trailing volatility estimate
        volatility_window.append(np.std(Y_perp[max(0, t-20):t+1]))
        estimator.update_online(y_t, x_t, volatility_window[-1])
        
    # --- Online Backtest (T=200 to 1000) ---
    print(f"Executing Live Streaming Backtest with Regime Switching (T={burn_in} to 1000)...")
    
    pnl_advanced = []
    pnl_baseline = []
    turnover_advanced = []
    
    w_prev_advanced = np.zeros(N)
    
    for t in range(burn_in, 1000):
        y_t, x_t = Y_perp[t], signals[t]
        vol_t = np.std(Y_perp[t-20:t])
        
        # 1. Generate Prediction using current operator (before seeing y_t)
        pred_advanced = estimator.predict(x_t)
        
        # Simple EW Baseline Prediction (mean of signals)
        pred_baseline = np.mean(x_t.reshape(N, P), axis=1)
        
        # 2. Allocate Capital (Long/Short Adapter)
        w_adv = pred_advanced - np.mean(pred_advanced)
        if np.sum(np.abs(w_adv)) > 1e-8: w_adv /= np.sum(np.abs(w_adv))
            
        w_base = pred_baseline - np.mean(pred_baseline)
        if np.sum(np.abs(w_base)) > 1e-8: w_base /= np.sum(np.abs(w_base))
            
        # Compute Returns
        ret_adv = np.sum(w_adv * y_t)
        ret_base = np.sum(w_base * y_t)
        
        pnl_advanced.append(ret_adv)
        pnl_baseline.append(ret_base)
        
        # Track Turnover
        turnover_advanced.append(np.sum(np.abs(w_adv - w_prev_advanced)))
        w_prev_advanced = w_adv
        
        # 3. Update Operator Online (Observing y_t)
        estimator.update_online(y_t, x_t, vol_t)
        
    # --- Metrics & Plotting ---
    pnl_advanced = np.array(pnl_advanced)
    pnl_baseline = np.array(pnl_baseline)
    
    cum_adv = np.cumsum(pnl_advanced) * 100.0
    cum_base = np.cumsum(pnl_baseline) * 100.0
    
    sharpe_adv = np.mean(pnl_advanced) / np.std(pnl_advanced) * np.sqrt(252) if np.std(pnl_advanced) > 0 else 0
    sharpe_base = np.mean(pnl_baseline) / np.std(pnl_baseline) * np.sqrt(252) if np.std(pnl_baseline) > 0 else 0
    
    avg_turnover = np.mean(turnover_advanced) * 100.0
    
    # --- Strategy Capacity ($AUM) Calculator ---
    # Assume a market environment with a Portfolio Average Daily Volume (ADV) of $5 Billion
    # and a maximum safe participation rate of 5% (to prevent market impact).
    portfolio_adv_usd = 5_000_000_000
    max_participation_rate = 0.05
    daily_traded_allowance = portfolio_adv_usd * max_participation_rate
    
    turnover_decimal = np.mean(turnover_advanced)
    if turnover_decimal > 1e-6:
        strategy_capacity_usd = daily_traded_allowance / turnover_decimal
    else:
        strategy_capacity_usd = float('inf')
        
    capacity_str = f"${strategy_capacity_usd / 1e6:.2f} Million" if strategy_capacity_usd < 1e9 else f"${strategy_capacity_usd / 1e9:.2f} Billion"

    print("\n" + "="*80)
    print("Advanced Methodologies Backtest Results (Institutional Capacity Mode)")
    print("="*80)
    print(f"Online Dynamic Operator | Sharpe: {sharpe_adv:.2f} | Avg Daily Turnover: {avg_turnover:.2f}% | Max AUM Capacity: {capacity_str}")
    print(f"EW Naive Baseline       | Sharpe: {sharpe_base:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(cum_adv, label=f"Online Dynamic Operator (Sharpe: {sharpe_adv:.2f})", color="#2980b9", linewidth=2.5)
    plt.plot(cum_base, label=f"EW Naive Baseline (Sharpe: {sharpe_base:.2f})", color="#7f8c8d", linestyle="--", linewidth=1.5)
    
    plt.title("Streaming PGD Performance in High-Dimension Alt-Data")
    plt.xlabel("Out-of-Sample Trading Day")
    plt.ylabel("Cumulative Excess Return (%)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("advanced_methodology_performance.png", dpi=150)
    print("Chart saved to advanced_methodology_performance.png")
    
if __name__ == "__main__":
    main()
