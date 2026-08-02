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

## 2. LaTeX Manuscript Availability

The complete LaTeX source of our academic manuscript detailing the methodology, optimization constraints, and 100-year daily empirical results is publicly available in the root of this repository:
*   **Manuscript:** [Alpha_Signals_manuscript.tex](Alpha_Signals_manuscript.tex)

Feel free to compile the document or review it directly for the complete mathematical derivation and backtest diagnostics.

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
SVD Predictive Mode Analysis
==================================================
Empirical Singular Values (Mode Strengths):
[0.003692 0.003488 0.002382 0.001004 0.      ]
Signal Family: Momentum   | Submatrix Norm: 0.003794
Signal Family: Reversal   | Submatrix Norm: 0.003375
Signal Family: Volatility | Submatrix Norm: 0.002588
==================================================
Annualized Return: 5.63%
Annualized Vol:    13.80%
Sharpe Ratio:      0.408
Performance chart saved to quickstart_performance.png
```
It will also save a performance visualization chart to `quickstart_performance.png`.

---

## 4. Platform Architecture (Local Workspace)

While the public repository provides the open-source Quickstart example and LaTeX manuscript, the proprietary 100-year historical database and empirical rolling backtest suite are maintained locally:
*   `code/data_prep.py`: Downloads daily data from Kenneth French's Dartmouth library, normalizes assets, and builds flat daily matrices.
*   `code/estimator.py`: The production PyTorch solver implementing warm-started nuclear norm and group lasso penalties.
*   `code/empirical.py`: Handles out-of-sample daily rolling backtests, re-estimating the operator monthly and computing persistence metrics.
*   `code/benchmarks.py`: Fits OLS, Ridge, and Random Forest ML state-gating classifiers.
*   `code/utils.py` & `code/visualization.py`: Computing HAC Newey-West standard errors, block bootstrapped p-values, and publication-ready charts.

---

## 5. License & Disclaimers

All research, code, and configurations are the intellectual property of the **ShockBridge Pulse Research Lab**. 
The repository is provided for academic review and research replication purposes. For licensing and commercial inquiries, contact: `rolffcoelho@hotmail.com`.
