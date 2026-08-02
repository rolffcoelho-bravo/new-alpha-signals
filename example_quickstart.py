"""
Beyond Backtesting: Dynamic Alpha Operator Quick-Start Demonstration.

This is a self-contained, high-profile example illustrating the mathematical 
and empirical execution of the regularized Dynamic Alpha Operator framework.

It demonstrates:
1. Benchmark-orthogonalization of asset returns.
2. PyTorch-based optimization of the operator subject to:
   - Nuclear Norm regularizations (inducing low-rank structures).
   - Group Lasso regularizations (inducing signal-family sparsity).
3. Singular Value Decomposition (SVD) analysis of predictive modes.
4. Capital allocation via a long-short portfolio adapter.
5. Visualization of strategy performance.

Designed for systematic quants, portfolio managers, and academic reviewers.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for exact reproducibility
np.random.seed(2025)
torch.manual_seed(2025)

# --- 1. Define the Dynamic Alpha Operator Estimator ---

class DynamicAlphaEstimator:
    """
    PyTorch-based regularized estimator for the high-dimensional Dynamic Alpha Operator.
    """
    def __init__(self, lambda_star=1e-4, lambda_grp=1e-4, lr=0.01, max_iter=150, tol=1e-6, verbose=True):
        self.lambda_star = lambda_star
        self.lambda_grp = lambda_grp
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        self.A_hat_ = None  # Estimated operator matrix N x NP
        self.U_ = None      # Left singular vectors (Return-space directions)
        self.S_ = None      # Singular values (Mode strengths)
        self.V_ = None      # Right singular vectors (Signal-space directions)
        
    def fit(self, Y_perp, X_vec, N, P):
        """
        Fits the regularized operator A.
        
        Parameters:
        - Y_perp: numpy array of size T x N (benchmark-orthogonal returns)
        - X_vec: numpy array of size T x NP (vectorized standardized signals)
        - N: int (number of assets)
        - P: int (number of signal families)
        """
        T = Y_perp.shape[0]
        NP = N * P
        
        # Compute Ridge OLS initialization for a warm, predictive start
        gamma = 1.0
        X_cov = X_vec.T @ X_vec + gamma * np.eye(NP)
        A_ols_T = np.linalg.solve(X_cov, X_vec.T @ Y_perp)  # NP x N
        A_init_val = A_ols_T.T  # N x NP
        
        # Convert to PyTorch tensors
        y = torch.tensor(Y_perp, dtype=torch.float32)  # T x N
        x = torch.tensor(X_vec, dtype=torch.float32)  # T x NP
        
        # Estimate residual covariance weighting matrix Omega
        diag_var = torch.var(y, dim=0) + 1e-6
        omega_inv = torch.diag(1.0 / diag_var)  # N x N
        
        # Initialize A from Ridge OLS
        A = torch.tensor(A_init_val, dtype=torch.float32, requires_grad=True)
        
        # Optimize using Adam
        optimizer = torch.optim.Adam([A], lr=self.lr)
        
        best_loss = float('inf')
        best_A = None
        
        for iteration in range(self.max_iter):
            optimizer.zero_grad()
            
            # 1. Data Loss: 1/T * sum_t (y_t - A x_t)' Omega_inv (y_t - A x_t)
            predictions = torch.matmul(x, A.t())  # T x N
            residuals = y - predictions  # T x N
            
            weighted_res = torch.matmul(residuals, omega_inv)  # T x N
            data_loss = torch.sum(residuals * weighted_res) / T
            
            # 2. Nuclear Norm (L1-like penalty on singular values)
            s_vals = torch.linalg.svdvals(A)
            nuclear_loss = torch.sum(s_vals)
            
            # 3. Group Lasso (grouped by signal family across all assets)
            group_loss = torch.tensor(0.0, dtype=torch.float32)
            for p in range(P):
                indices = torch.arange(p, NP, P)
                A_sub = A[:, indices]  # N x N
                group_loss += torch.linalg.matrix_norm(A_sub, ord='fro')
                
            # Total Loss
            loss = data_loss + self.lambda_star * nuclear_loss + self.lambda_grp * group_loss
            
            loss_val = loss.item()
            if iteration > 10 and abs(loss_val - best_loss) < self.tol:
                break
                
            if loss_val < best_loss:
                best_loss = loss_val
                best_A = A.detach().clone()
                
            loss.backward()
            optimizer.step()
            
        self.A_hat_ = best_A.cpu().numpy()
        
        # SVD Mode Decomposition
        U, S, Vh = np.linalg.svd(self.A_hat_, full_matrices=False)
        self.U_ = U          # Left singular vectors (N x r)
        self.S_ = S          # Singular values (r,)
        self.V_ = Vh.T       # Right singular vectors (NP x r)
        
        return self
    
    def predict(self, X_vec):
        return X_vec @ self.A_hat_.T

# --- 2. Simulation and Execution ---

def generate_synthetic_data(T=1000, N=5, P=3):
    """
    Generates realistic asset pricing dataset with a low-rank predictive signal subspace.
    """
    print("Generating synthetic asset returns and high-dimensional signals...")
    
    # Generate systematic benchmark factors (e.g. market factor)
    market_factor = np.random.normal(0.0, 0.01, (T, 1))
    
    # Asset exposures to the benchmark (betas)
    betas = np.array([[1.0], [0.8], [1.2], [0.5], [1.1]]) # N x 1
    
    # Idiosyncratic returns
    idiosyncratic = np.random.normal(0.0, 0.015, (T, N))
    
    # Total returns
    returns = market_factor @ betas.T + idiosyncratic # T x N
    
    # Generate P signal families (e.g., Mom, Rev, Vol)
    # The true predictive mapping is low-rank: only the first signal is predictive
    signals = np.random.normal(0.0, 1.0, (T, N, P))
    
    # Insert predictable component (orthogonal alpha)
    true_alpha_direction = np.array([0.5, -0.5, 0.2, -0.2, 0.0]) # N
    predictive_signal = signals[:, :, 0] # Use first signal family
    
    alpha_returns = (predictive_signal @ true_alpha_direction.reshape(-1, 1)) * 0.002
    returns += alpha_returns
    
    # Flatten signals to size T x NP
    signals_flat = np.zeros((T, N * P))
    for t in range(T):
        signals_flat[t] = signals[t].flatten()
        
    return returns, market_factor, betas, signals_flat

def main():
    N, P = 5, 3
    NP = N * P
    
    # 1. Ingest Data
    returns, factors, betas, signals = generate_synthetic_data(T=1000, N=N, P=P)
    
    # 2. Compute Benchmark-Orthogonal Returns
    print("Projecting returns onto the orthogonal complement of the benchmark...")
    Y_perp = np.zeros_like(returns)
    I_N = np.eye(N)
    for t in range(len(returns)):
        B_t = betas # Static betas in this example
        B_inv = np.linalg.inv(B_t.T @ B_t)
        M_B = I_N - B_t @ B_inv @ B_t.T
        Y_perp[t] = M_B @ returns[t]
        
    # Split into Train and Test (50/50)
    split = 500
    Y_train, Y_test = Y_perp[:split], Y_perp[split:]
    X_train, X_test = signals[:split], signals[split:]
    
    # 3. Fit the Dynamic Alpha Operator
    print("\nFitting regularized Dynamic Alpha Operator...")
    estimator = DynamicAlphaEstimator(
        lambda_star=1e-5, 
        lambda_grp=1e-5, 
        lr=0.01, 
        max_iter=200
    )
    estimator.fit(Y_train, X_train, N, P)
    
    print("\n" + "="*50)
    print("SVD Predictive Mode Analysis")
    print("="*50)
    print(f"Empirical Singular Values (Mode Strengths):\n{estimator.S_.round(6)}")
    
    # Norm of submatrices for signal families
    family_names = ["Momentum", "Reversal", "Volatility"]
    for p in range(P):
        indices = np.arange(p, NP, P)
        A_sub = estimator.A_hat_[:, indices]
        norm = np.linalg.norm(A_sub, ord='fro')
        print(f"Signal Family: {family_names[p]:<10} | Submatrix Norm: {norm:.6f}")
    print("="*50)
    
    # 4. Out-of-Sample Portfolio Adapter
    print("\nBacktesting out-of-sample performance...")
    predictions = estimator.predict(X_test) # T_test x N
    
    # Long-Short capital allocator
    strategy_returns = []
    for t in range(len(predictions)):
        pred = predictions[t]
        w = pred - np.mean(pred) # Zero-net exposure
        w = w / (np.sum(np.abs(w)) + 1e-8) # Leverage limit
        
        # Portfolio return
        ret = np.sum(w * Y_test[t])
        strategy_returns.append(ret)
        
    strategy_returns = np.array(strategy_returns)
    cum_returns = np.cumsum(strategy_returns) * 100.0
    
    # Calculate performance metrics
    ann_return = np.mean(strategy_returns) * 252.0 * 100.0
    ann_vol = np.std(strategy_returns) * np.sqrt(252.0) * 100.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    print(f"Annualized Return: {ann_return:.2f}%")
    print(f"Annualized Vol:    {ann_vol:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.3f}")
    
    # Plot performance
    plt.figure(figsize=(10, 5))
    plt.plot(cum_returns, label="Dynamic Alpha Long-Short", color="#2980b9", linewidth=2)
    plt.title("Out-of-Sample Cumulative Returns (Quick-Start Example)")
    plt.xlabel("Trading Day")
    plt.ylabel("Cumulative Excess Return (%)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("quickstart_performance.png", dpi=150)
    print("\nPerformance chart saved to quickstart_performance.png")
    print("Quickstart demo completed successfully!")

if __name__ == "__main__":
    main()
