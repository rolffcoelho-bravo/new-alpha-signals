"""
Beyond Backtesting: Dynamic Alpha Operator Quick-Start Demonstration.

This is a self-contained, high-profile example illustrating the mathematical 
and empirical execution of the regularized Dynamic Alpha Operator framework.

It uses a custom, high-performance Proximal Gradient Descent (PGD) solver 
in pure NumPy, making it 100% independent of heavy DL libraries and highly stable.

It demonstrates:
1. Benchmark-orthogonalization of asset returns.
2. Proximal Gradient Descent (PGD) optimization of the operator subject to:
   - Singular Value Thresholding (inducing low-rank structures).
   - Block Soft-Thresholding (inducing signal-family sparsity).
3. SVD Mode Decomposition and signal family importance attribution.
4. Capital allocation via a long-short portfolio adapter.
5. Out-of-sample backtesting comparison against OLS, Ridge, and Naive benchmarks.
6. Generation of performance comparison charts.

Designed for systematic quants, portfolio managers, and academic reviewers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set seed for exact reproducibility
np.random.seed(2025)

# --- 1. Define the Dynamic Alpha Operator Estimator ---

class DynamicAlphaEstimator:
    """
    NumPy-based proximal gradient solver for the high-dimensional Dynamic Alpha Operator.
    Enforces nuclear norm and group lasso penalties exactly via SVT and BST.
    """
    def __init__(self, lambda_star=0.001, lambda_grp=0.001, lr=0.001, max_iter=250, tol=1e-6):
        self.lambda_star = lambda_star
        self.lambda_grp = lambda_grp
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        
        self.A_hat_ = None  # Estimated operator matrix N x NP
        self.U_ = None      # Left singular vectors (Return-space directions)
        self.S_ = None      # Singular values (Mode strengths)
        self.V_ = None      # Right singular vectors (Signal-space directions)
        
    def fit(self, Y_perp, X_vec, N, P):
        """
        Fits the regularized operator A using Proximal Gradient Descent.
        """
        T = Y_perp.shape[0]
        NP = N * P
        
        # Compute Ridge OLS initialization for a warm, predictive start
        gamma = 1.0
        X_cov = X_vec.T @ X_vec + gamma * np.eye(NP)
        A_ols_T = np.linalg.solve(X_cov, X_vec.T @ Y_perp)  # NP x N
        A = A_ols_T.T  # N x NP
        
        # Estimate residual covariance weighting matrix Omega
        diag_var = np.var(Y_perp, axis=0) + 1e-6
        omega_inv = np.diag(1.0 / diag_var)  # N x N
        
        best_loss = float('inf')
        best_A = A.copy()
        
        for iteration in range(self.max_iter):
            # 1. Gradient Step on Data Loss
            predictions = X_vec @ A.T
            residuals = Y_perp - predictions  # T x N
            
            # Gradient: N x NP
            grad_data = -2.0 / T * (omega_inv @ residuals.T @ X_vec)
            
            # Use a normalized gradient step to prevent overflow/divergence
            grad_norm = np.linalg.norm(grad_data)
            if grad_norm > 1e-8:
                A_next = A - self.lr * (grad_data / grad_norm)
            else:
                A_next = A - self.lr * grad_data
            
            # 2. Proximal Step for Group Lasso (Block Soft-Thresholding)
            for p in range(P):
                indices = np.arange(p, NP, P)
                A_sub = A_next[:, indices]  # N x N
                norm_val = np.linalg.norm(A_sub, ord='fro')
                if norm_val > 1e-8:
                    scale = np.maximum(1.0 - self.lr * self.lambda_grp / norm_val, 0.0)
                    A_next[:, indices] = A_sub * scale
                else:
                    A_next[:, indices] = 0.0
                    
            # 3. Proximal Step for Nuclear Norm (Singular Value Thresholding - SVT)
            U, S, Vt = np.linalg.svd(A_next, full_matrices=False)
            S_thresh = np.maximum(S - self.lr * self.lambda_star, 0.0)
            A_next = U @ np.diag(S_thresh) @ Vt
            
            # Calculate objective function value
            pred = X_vec @ A_next.T
            res = Y_perp - pred
            weighted_res = res @ omega_inv
            data_loss = np.sum(res * weighted_res) / T
            nuclear_loss = np.sum(S_thresh)
            
            group_loss = 0.0
            for p in range(P):
                indices = np.arange(p, NP, P)
                group_loss += np.linalg.norm(A_next[:, indices], ord='fro')
                
            loss_val = data_loss + self.lambda_star * nuclear_loss + self.lambda_grp * group_loss
            
            if iteration > 10 and abs(loss_val - best_loss) < self.tol:
                break
                
            if loss_val < best_loss:
                best_loss = loss_val
                best_A = A_next.copy()
                
            A = A_next
            
        self.A_hat_ = best_A
        
        # Final SVD Mode Decomposition
        U, S, Vh = np.linalg.svd(self.A_hat_, full_matrices=False)
        self.U_ = U
        self.S_ = S
        self.V_ = Vh.T
        
        return self
    
    def predict(self, X_vec):
        return X_vec @ self.A_hat_.T

# --- 2. Simulation and Execution ---

def generate_synthetic_data(T=600, N=10, P=8):
    """
    Generates high-dimensional, noisy signal panel.
    N=10 assets, P=8 signal families (80 parameters total).
    Only the first signal family (Momentum) contains true alpha.
    The other 7 signal families are pure Gaussian noise.
    """
    print(f"Generating synthetic returns and high-dimensional noisy signals ({N} assets, {P} signals)...")
    
    # Generate systematic benchmark factor (market)
    market_factor = np.random.normal(0.0, 0.01, (T, 1))
    betas = np.random.uniform(0.5, 1.5, (N, 1)) # N x 1
    
    # Idiosyncratic returns
    idiosyncratic = np.random.normal(0.0, 0.015, (T, N))
    returns = market_factor @ betas.T + idiosyncratic # T x N
    
    # Generate P signals (80 total variables)
    signals = np.random.normal(0.0, 1.0, (T, N, P))
    
    # Target predictive direction (N,)
    true_alpha_direction = np.array([0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
    predictive_signal = signals[:, :, 0] # Signal 0 (Momentum) is the predictive one
    
    # Add alpha to returns
    alpha_returns = (predictive_signal @ true_alpha_direction.reshape(-1, 1)) * 0.008
    returns += alpha_returns
    
    # Flatten signals to size T x NP
    signals_flat = np.zeros((T, N * P))
    for t in range(T):
        signals_flat[t] = signals[t].flatten()
        
    return returns, market_factor, betas, signals_flat

def main():
    N, P = 10, 8
    NP = N * P
    
    # 1. Ingest Data
    returns, factors, betas, signals = generate_synthetic_data(T=600, N=N, P=P)
    
    # 2. Compute Benchmark-Orthogonal Returns
    print("Projecting returns onto the orthogonal complement of the benchmark risk factors...")
    Y_perp = np.zeros_like(returns)
    I_N = np.eye(N)
    for t in range(len(returns)):
        B_t = betas
        B_inv = np.linalg.inv(B_t.T @ B_t)
        M_B = I_N - B_t @ B_inv @ B_t.T
        Y_perp[t] = M_B @ returns[t]
        
    # Split into Train and Test (Train=100 days, Test=500 days)
    # High-dimensional: 80 parameters to estimate on only 100 observations!
    split = 100
    Y_train, Y_test = Y_perp[:split], Y_perp[split:]
    X_train, X_test = signals[:split], signals[split:]
    
    # 3. Fit Proposed Estimator (Dynamic Alpha Operator)
    print("\nFitting regularized Dynamic Alpha Operator via Proximal Gradient Descent...")
    estimator = DynamicAlphaEstimator(
        lambda_star=0.0001, 
        lambda_grp=0.0001, 
        lr=0.001, 
        max_iter=300
    )
    estimator.fit(Y_train, X_train, N, P)
    
    # 4. Fit Baseline Benchmark Models
    # Classical OLS (massively overfits noise because T_train is close to NP!)
    print("Fitting Benchmark 1: Classical OLS model...")
    A_ols = np.linalg.solve(X_train.T @ X_train + 1e-4 * np.eye(NP), X_train.T @ Y_train).T
    
    # Regularized Ridge (L2 penalty)
    print("Fitting Benchmark 2: Regularized Ridge model...")
    A_ridge = np.linalg.solve(X_train.T @ X_train + 80.0 * np.eye(NP), X_train.T @ Y_train).T
    
    # --- Print SVD Mode Attribution ---
    print("\n" + "="*50)
    print("SVD Predictive Mode Analysis (Proposed)")
    print("="*50)
    print(f"Empirical Singular Values (Mode Strengths):\n{estimator.S_.round(6)}")
    
    family_names = ["Momentum", "Reversal_W", "Reversal_D", "Vol_21", "Vol_252", "Skew", "Kurtosis", "Char"]
    for p in range(P):
        indices = np.arange(p, NP, P)
        A_sub = estimator.A_hat_[:, indices]
        norm = np.linalg.norm(A_sub, ord='fro')
        print(f"Signal Family: {family_names[p]:<10} | Submatrix Norm: {norm:.6f}")
    print("="*50)
    
    # 5. Out-of-Sample Portfolio Backtest Loop
    print("\nExecuting out-of-sample backtest comparison...")
    
    # Get predictions for all models
    preds = {
        "EW_Naive": np.zeros((len(Y_test), N)),
        "OLS": X_test @ A_ols.T,
        "Ridge": X_test @ A_ridge.T,
        "Dynamic_Alpha": estimator.predict(X_test)
    }
    
    # Fill EW Naive predictions (average signals across all families)
    for t in range(len(Y_test)):
        X_t_mat = X_test[t].reshape(N, P)
        preds["EW_Naive"][t] = np.mean(X_t_mat, axis=1)
        
    # Standardize weights and calculate returns for each strategy
    strategy_returns = {k: [] for k in preds.keys()}
    
    for t in range(len(Y_test)):
        y_perp_t = Y_test[t]
        
        for k in preds.keys():
            pred = preds[k][t]
            # Zero-net exposure adapter
            w = pred - np.mean(pred)
            norm_val = np.sum(np.abs(w))
            if norm_val > 1e-8:
                w = w / norm_val
            else:
                w = np.zeros_like(w)
                
            ret = np.sum(w * y_perp_t)
            strategy_returns[k].append(ret)
            
    # Calculate performance metrics
    print("\n" + "="*65)
    print("Out-of-Sample Portfolio Metrics Comparison")
    print("="*65)
    
    plt.figure(figsize=(10, 6))
    colors = {
        "EW_Naive": "#7f8c8d",       # Gray
        "OLS": "#e74c3c",            # Red
        "Ridge": "#e67e22",          # Orange
        "Dynamic_Alpha": "#2980b9"   # Blue
    }
    line_styles = {
        "EW_Naive": "--",
        "OLS": ":",
        "Ridge": "-.",
        "Dynamic_Alpha": "-"
    }
    labels = {
        "EW_Naive": "Benchmark 0: EW Naive Signals",
        "OLS": "Benchmark 1: Classical OLS (Overfits)",
        "Ridge": "Benchmark 2: Regularized Ridge",
        "Dynamic_Alpha": "Proposed: Dynamic Alpha Operator"
    }
    
    for k in strategy_returns.keys():
        ret_series = np.array(strategy_returns[k])
        cum_ret = np.cumsum(ret_series) * 100.0
        
        ann_return = np.mean(ret_series) * 252.0 * 100.0
        ann_vol = np.std(ret_series) * np.sqrt(252.0) * 100.0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        
        # Max drawdown
        peaks = np.maximum.accumulate(cum_ret)
        mdd = np.max(peaks - cum_ret)
        
        print(f"Strategy: {k:<15} | Ann Return: {ann_return:>6.2f}% | Ann Vol: {ann_vol:>5.2f}% | Sharpe: {sharpe:>5.3f} | Max DD: {mdd:>5.2f}%")
        
        # Plot path
        plt.plot(
            cum_ret, 
            label=labels[k], 
            color=colors[k], 
            linestyle=line_styles[k],
            linewidth=2.5 if k == "Dynamic_Alpha" else 1.5
        )
        
    print("="*65)
    
    plt.title("Out-of-Sample Performance Comparison (High-Dimensional Noisy Signals)")
    plt.xlabel("Trading Day")
    plt.ylabel("Cumulative Excess Return (%)")
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig("quickstart_performance.png", dpi=150)
    print("\nPerformance comparison chart saved to quickstart_performance.png")
    print("Quickstart demo completed successfully!")

if __name__ == "__main__":
    main()
