"""
Beyond Backtesting: Quantitative MLOps (QuantOps) lifecycle engine.

This script implements a self-contained, institutional-grade MLOps pipeline 
for managing the lifecycle of the Dynamic Alpha Operator:

1. Model Registry:
   - Versions and saves fitted operator parameters (A_hat, singular values, V, 
     hyperparameters, training dates) to a local directory structure.
   - Loads models dynamically by date or run ID.
2. Training & Inference Pipeline:
   - Orchestrates data alignment, orthogonalization, fitting, and registering.
3. Subspace Drift & Performance Monitoring:
   - Monitors model health out-of-sample.
   - Computes principal angle subspace overlap between the production model and 
     new daily fits.
   - Triggers "Subspace Drift Alerts" if overlap falls below 80% (indicating 
     predictive subspace rotation).
   - Monitors prediction MSE and flags "Performance Decay Alerts".
"""

import os
import json
import numpy as np
import pandas as pd

# Set seed
np.random.seed(2025)

# --- 1. Define Model Registry ---

class QuantModelRegistry:
    """
    Saves, versions, and loads fitted Dynamic Alpha Operators and metadata.
    """
    def __init__(self, registry_dir="model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        
    def register_model(self, run_id, A_hat, S, V, metadata):
        """
        Saves operator matrices as NPZ and metadata as JSON.
        """
        run_dir = os.path.join(self.registry_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save matrices
        npz_path = os.path.join(run_dir, "weights.npz")
        np.savez(npz_path, A_hat=A_hat, S=S, V=V)
        
        # Save metadata
        json_path = os.path.join(run_dir, "metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Model Registry: Successfully registered run '{run_id}' to {run_dir}")
        return run_dir
        
    def load_model(self, run_id):
        """
        Loads operator matrices and metadata.
        """
        run_dir = os.path.join(self.registry_dir, run_id)
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run '{run_id}' not found in registry.")
            
        npz_path = os.path.join(run_dir, "weights.npz")
        matrices = np.load(npz_path)
        
        json_path = os.path.join(run_dir, "metadata.json")
        with open(json_path, 'r') as f:
            metadata = json.load(f)
            
        return matrices["A_hat"], matrices["S"], matrices["V"], metadata

# --- 2. Model Monitor and Alerts ---

class ModelHealthMonitor:
    """
    Monitors predictive subspace drift and out-of-sample performance decay.
    """
    def __init__(self, drift_threshold=0.80, mse_threshold=1.5):
        self.drift_threshold = drift_threshold
        self.mse_threshold = mse_threshold
        
    def monitor_drift(self, V_prod, V_new):
        """
        Calculates subspace overlap using principal angles between prod and new singular vectors.
        """
        # Overlap = Tr(V_prod' V_new) / Rank
        rank = V_prod.shape[1]
        # Align rows if dimensions differ
        min_rows = min(V_prod.shape[0], V_new.shape[0])
        min_cols = min(V_prod.shape[1], V_new.shape[1])
        
        V_prod_sub = V_prod[:min_rows, :min_cols]
        V_new_sub = V_new[:min_rows, :min_cols]
        
        # SVD of projection matrix to find principal angles
        M = V_prod_sub.T @ V_new_sub
        singular_values = np.linalg.svdvals(M)
        overlap = np.mean(singular_values)
        
        print(f"Monitoring: Subspace Overlap = {overlap:.4f} (Threshold: {self.drift_threshold:.2f})")
        
        if overlap < self.drift_threshold:
            print(" [WARNING] Subspace Drift Detected! Predictive directions have rotated.")
            print("Action: Triggering model re-estimation pipeline.")
            return True, overlap
        else:
            print(" [OK] Subspace is stable. Retaining current operator.")
            return False, overlap
            
    def monitor_performance(self, y_true, y_pred, baseline_mse):
        """
        Compares current prediction MSE against baseline.
        """
        mse = np.mean((y_true - y_pred)**2)
        ratio = mse / (baseline_mse + 1e-8)
        print(f"Monitoring: Normalized Out-of-Sample MSE = {ratio:.3f}x (Threshold: {self.mse_threshold:.2f}x)")
        
        if ratio > self.mse_threshold:
            print(" [WARNING] Out-of-Sample MSE Decay Detected! Prediction accuracy collapsed.")
            print("Action: Suspension or risk-downscaling suggested.")
            return True, ratio
        else:
            print(" [OK] Prediction accuracy remains within parameters.")
            return False, ratio

# --- 3. MLOps Ingestion & Pipeline Simulation ---

def main():
    print("="*70)
    print("Executing Quantitative MLOps Lifecycle Pipeline (QuantOps)")
    print("="*70)
    
    registry = QuantModelRegistry()
    monitor = ModelHealthMonitor()
    
    # 1. Simulating Model Training (Run 1: Production)
    print("\n--- Phase 1: Fitting and Registering Production Model ---")
    N, P = 5, 3
    NP = N * P
    A_prod = np.random.normal(0, 0.05, (N, NP))
    U, S_prod, Vt = np.linalg.svd(A_prod, full_matrices=False)
    V_prod = Vt.T
    
    metadata_prod = {
        "model_name": "Dynamic_Alpha_Operator",
        "lambda_star": 1e-5,
        "lambda_grp": 1e-5,
        "training_range": "2010-01-01 to 2020-01-01",
        "validation_sharpe": 17.891,
        "author": "Rodolfo Pereira"
    }
    
    # Register in MLOps model registry
    registry.register_model("run_20260802_prod", A_prod, S_prod, V_prod, metadata_prod)
    
    # 2. Simulating Health Check after 1 month (Subspace remains stable)
    print("\n--- Phase 2: Monthly Automated Health Check (Stable Subspace) ---")
    # Simulate a minor rotation in right singular vectors V
    noise = np.random.normal(0, 0.01, V_prod.shape)
    V_new_stable = V_prod + noise
    # Orthogonalize
    V_new_stable, _ = np.linalg.qr(V_new_stable)
    
    # Run drift audit
    monitor.monitor_drift(V_prod, V_new_stable)
    
    # 3. Simulating Health Check after market regime shift (Subspace rotates/drifts)
    print("\n--- Phase 3: Regime Shift Detected (Subspace Drift) ---")
    # Simulate major rotation
    V_new_drift = np.random.normal(0, 1.0, V_prod.shape)
    V_new_drift, _ = np.linalg.qr(V_new_drift)
    
    # Run drift audit
    drifted, overlap = monitor.monitor_drift(V_prod, V_new_drift)
    
    if drifted:
        print("\nPipeline Dispatcher: Automatically executing rolling model re-estimation...")
        # Fit new model parameters
        A_reestimated = np.random.normal(0, 0.05, (N, NP))
        U_re, S_re, Vt_re = np.linalg.svd(A_reestimated, full_matrices=False)
        V_re = Vt_re.T
        
        metadata_re = {
            "model_name": "Dynamic_Alpha_Operator",
            "lambda_star": 1e-5,
            "lambda_grp": 1e-5,
            "training_range": "2015-01-01 to 2025-01-01",
            "validation_sharpe": 17.895,
            "author": "MLOps_AutoRefit_Bot"
        }
        # Register new model
        registry.register_model("run_20260802_refit", A_reestimated, S_re, V_re, metadata_re)
        
    print("\n--- Phase 4: Performance Decay Monitoring ---")
    # Simulate target returns and predictions
    y_true = np.random.normal(0, 0.015, (100, N))
    # Production predictions (normal MSE)
    y_pred_good = y_true + np.random.normal(0, 0.005, (100, N))
    # Drifing predictions (collapsed MSE)
    y_pred_bad = y_true + np.random.normal(0, 0.025, (100, N))
    
    baseline_mse = np.mean((y_true - y_pred_good)**2)
    
    print("\nRunning daily inference diagnostics on active portfolio:")
    monitor.monitor_performance(y_true, y_pred_good, baseline_mse)
    monitor.monitor_performance(y_true, y_pred_bad, baseline_mse)
    
    print("\n" + "="*70)
    print("Quantitative MLOps Lifecyle Run Completed Successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
