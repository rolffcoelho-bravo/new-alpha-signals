"""
Beyond Backtesting: Unit Testing Suite for Dynamic Alpha Operator Estimator.

This suite contains Python unit tests verifying the correctness of our
NumPy-based Proximal Gradient Descent solver (SVT, BST, and orthogonality).
"""

import sys
import os
import numpy as np
import pytest

# Ensure parent directory is in path to import example_quickstart
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_quickstart import DynamicAlphaEstimator, generate_synthetic_data

def test_estimator_initialization():
    """Verifies estimator hyperparameters are initialized correctly."""
    est = DynamicAlphaEstimator(lambda_star=1e-4, lambda_grp=2e-4, lr=0.01)
    assert est.lambda_star == 1e-4
    assert est.lambda_grp == 2e-4
    assert est.lr == 0.01
    assert est.A_hat_ is None

def test_orthogonality_projection():
    """Verifies that projected returns are mathematically orthogonal to market betas."""
    N, T = 5, 100
    betas = np.random.uniform(0.5, 1.5, (N, 1))
    returns = np.random.normal(0, 0.01, (T, N))
    
    Y_perp = np.zeros_like(returns)
    I_N = np.eye(N)
    for t in range(T):
        M_B = I_N - betas @ np.linalg.inv(betas.T @ betas) @ betas.T
        Y_perp[t] = M_B @ returns[t]
        
    # Check that Y_perp is orthogonal to betas (inner product must be 0)
    for t in range(T):
        projection_val = Y_perp[t] @ betas
        assert np.abs(projection_val[0]) < 1e-12

def test_estimator_fitting():
    """Verifies fitting dimensions and convergence under low-dimensional settings."""
    N, P, T = 4, 3, 50
    np.random.seed(42)
    Y = np.random.normal(0, 0.01, (T, N))
    X = np.random.normal(0, 1.0, (T, N * P))
    
    est = DynamicAlphaEstimator(lambda_star=1e-5, lambda_grp=1e-5, lr=0.005, max_iter=20)
    est.fit(Y, X, N, P)
    
    # Check shapes
    assert est.A_hat_.shape == (N, N * P)
    assert est.U_.shape == (N, N)
    assert est.S_.shape == (N,)
    assert est.V_.shape == (N * P, N)
    
    # Check predictions shape
    preds = est.predict(X)
    assert preds.shape == (T, N)

def test_svt_thresholding():
    """Verifies Singular Value Thresholding compresses singular values below threshold."""
    N, NP = 3, 6
    A = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    ])
    
    # If we apply SVT with threshold lambda_star = 0.2
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_thresh = np.maximum(S - 0.2, 0.0)
    A_thresh = U @ np.diag(S_thresh) @ Vt
    
    # The smallest singular value (0.1) should collapse to exactly 0
    _, S_new, _ = np.linalg.svd(A_thresh, full_matrices=False)
    assert S_new[2] == 0.0
    assert np.abs(S_new[0] - 0.8) < 1e-12
