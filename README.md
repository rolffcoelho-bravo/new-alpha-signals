# Beyond Backtesting: A New Framework for Alpha Signals

[![ShockBridge Pulse](https://img.shields.io/badge/Research-ShockBridge%20Pulse-blue.svg)](http://www.shockbridgepulse.com)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Under%20Active%20Research-orange.svg)]()

A systematic alpha-signal research and execution platform developing a new statistical framework for identifying, decomposing, validating, and conditionally deploying high-dimensional predictive signals in systematic financial markets.

---

## 1. Project Overview

Hedge funds and systematic investment firms face a common set of challenges when managing large signal libraries:
*   **Signal Redundancy:** Determining if new signals contain incremental information or if they are merely noise-diluted expressions of existing risk exposures.
*   **Subspace Decay:** Measuring the physical rotation and decay of predictive subspaces across different market regimes and forecasting horizons.
*   **Conditional Deployment:** Deciding when a signal should be activated, scaled down, or suspended based on diagnostic state indicators.

This repository implements the reference architecture for the **Dynamic Alpha Operator** framework, which treats high-dimensional signal libraries as a unified statistical operator. It isolates benchmark-orthogonal alpha structures, conducts multi-horizon persistence audits, and provides a systematic portfolio interface for conditional capital allocation.

---

## 2. Academic Manuscript Availability

The compiled PDF of our academic manuscript detailing the methodology, optimization constraints, and 100-year daily empirical results is publicly available in the root of this repository:
*   **Manuscript PDF:** [Alpha_Signals_manuscript.pdf](Alpha_Signals_manuscript.pdf)

Review the PDF directly for the complete mathematical derivation, out-of-sample backtest diagnostics, and empirical findings.

---

## 3. Standalone Quickstart Example

For professional transparency and immediate testing, we provide a fully functional, self-contained demonstration script in the root:
*   **Quickstart Script:** [example_quickstart.py](example_quickstart.py)

This script generates a realistic synthetic dataset representing high-dimensional signals and asset returns, projects returns onto the orthogonal complement of the benchmark risk exposures, fits our regularized PyTorch operator using OLS warm start, performs SVD mode decomposition, and backtests a long-short portfolio strategy.

### Installation
Ensure you have the required dependencies installed:
```bash
pip install torch numpy pandas matplotlib
```

### Running the Demo
Execute the quickstart script directly from your terminal:
```bash
python example_quickstart.py
```

### Expected Output
The script will print the SVD mode strengths (singular values), identify the relative importance of signal families, and output out-of-sample portfolio statistics:
```text
==================================================
SVD Predictive Mode Analysis (Proposed)
==================================================
Empirical Singular Values (Mode Strengths):
[3.8024e-02 3.5594e-02 3.4498e-02 2.8064e-02 2.5212e-02 2.1010e-02
 1.9473e-02 1.8679e-02 1.4648e-02 7.6000e-05]
==================================================
Annualized Return: 16.88%
Annualized Vol:    9.90%
Sharpe Ratio:      1.706
Performance comparison chart saved to quickstart_performance.png
```
It will also save a performance comparison chart to `quickstart_performance.png`, plotting the cumulative returns of all baseline models.

---

## 4. Benchmark Comparison & Strategic Edge

As demonstrated in the quickstart script and validated in the 100-year daily empirical results of our paper, the proposed **Dynamic Alpha Operator** achieves a distinct strategic edge over traditional predictive models under high-dimensional noisy settings:

1.  **Naive Signal Combinations (EW Naive):** Simply averaging signal scores fails because it is blind to signal noise. In our high-dimensional test, this leads to capital loss and high drawdowns (negative Sharpe ratio of -1.029).
2.  **Classical OLS Regression:** When the parameter space is large relative to the training sample, unregularized OLS overfits the local noise of the signal library, causing high variance out-of-sample.
3.  **Regularized Ridge Regression:** While Ridge (L2 penalty) reduces overfitting by shrinking coefficients, it treats all assets and signals independently. It is blind to the systematic co-movement of signal loadings.
4.  **Dynamic Alpha Operator (Proposed):** By applying a **nuclear norm** (L1 on singular values) and **group lasso** penalty directly to the return-signal matrix operator, our framework projects signal loadings onto a stable, low-rank predictive subspace. This filters out the non-systematic noise of the signal library, resulting in the highest risk-adjusted performance (Sharpe ratio of 1.706) and complete capital protection.

---

## 5. Platform Architecture (Local Workspace)

While the public repository provides the open-source Quickstart example and academic manuscript PDF, the proprietary 100-year historical database and empirical rolling backtest suite are maintained locally:
*   `code/data_prep.py`: Downloads daily data from Kenneth French's Dartmouth library, normalizes assets, and builds flat daily matrices.
*   `code/estimator.py`: The production PyTorch solver implementing warm-started nuclear norm and group lasso penalties.
*   `code/empirical.py`: Handles out-of-sample daily rolling backtests, re-estimating the operator monthly and computing persistence metrics.
*   `code/benchmarks.py`: Fits OLS, Ridge, and Random Forest ML state-gating classifiers.
*   `code/utils.py` & `code/visualization.py`: Computing HAC Newey-West standard errors, block bootstrapped p-values, and publication-ready charts.

---

## 6. License & Disclaimers

All research, code, and configurations are the intellectual property of the **ShockBridge Pulse Research Lab**. 
The repository is provided for academic review and research replication purposes. For licensing and commercial inquiries, contact: `rolffcoelho@hotmail.com`.
