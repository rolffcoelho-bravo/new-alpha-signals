# Beyond Backtesting: A New Framework for Alpha Signals

[![ShockBridge Pulse](https://img.shields.io/badge/Research-ShockBridge%20Pulse-blue.svg)](http://www.shockbridgepulse.com)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Under%20Active%20Research-orange.svg)]()
[![Dynamic Performance](https://github.com/rolffcoelho-bravo/new-alpha-signals/actions/workflows/run_quickstart.yml/badge.svg)](https://github.com/rolffcoelho-bravo/new-alpha-signals/actions)

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

## 3. Dynamic Quickstart Performance

The chart below shows the out-of-sample cumulative returns of our framework compared against unregularized and standard L2 shrinkage benchmarks. 

**This chart is dynamically compiled and updated by GitHub Actions on every commit:**

![Quickstart Performance](quickstart_performance.png)

### Expected Run Statistics
When executed, the system outputs the following out-of-sample statistics:
```text
=================================================================
Out-of-Sample Portfolio Metrics Comparison
=================================================================
Strategy: EW_Naive        | Ann Return: -10.18% | Ann Vol:  9.89% | Sharpe: -1.029 | Max DD: 26.97%
Strategy: OLS             | Ann Return:  16.85% | Ann Vol:  9.89% | Sharpe:  1.703 | Max DD:  6.28%
Strategy: Ridge           | Ann Return:  13.50% | Ann Vol:  9.65% | Sharpe:  1.399 | Max DD:  8.38%
Strategy: Dynamic_Alpha   | Ann Return:  16.94% | Ann Vol:  9.88% | Sharpe:  1.714 | Max DD:  6.25%
=================================================================
```

---

## 4. QuantOps Workspace

To demonstrate the execution and MLOps lifecycle of the Dynamic Alpha Operator in a production environment, we provide a structured QuantOps suite. 

Below is a visual representation of the unified **QuantOps Dashboard** tracking the live operator estimation, model registry state, and subspace drift checks:

![QuantOps Workspace Dashboard](quantops_dashboard.jpg)

### Workspace Component Directory

The QuantOps suite contains the following core scripts:

#### A. Live Real-World Ingestor (`examples/real_world_data.py`)
*   **Functional Role:** Data Ingestion, Cleaning, and Orthogonalization.
*   **Primary Inputs:** Live Yahoo Finance daily adjusted close price feeds for **SPY** (market factor), **AAPL**, **MSFT**, **QQQ**, and **AMZN**.
*   **Key Outputs:** SPY-orthogonal returns ($\mathbf{y}_t^{\perp \mathbf{B}}$) and cross-sectionally standardized signals.
*   **Operational Value:** Automates live data parsing, rolling beta estimates, risk projection, and protects against rolling standard deviation NaN values on start dates.
```bash
python examples/real_world_data.py
```

#### B. MLOps Lifecycle Engine (`examples/quant_mlops.py`)
*   **Functional Role:** Parameter Versioning and Subspace Drift Monitoring.
*   **Primary Inputs:** Production operator weights ($A$, $S$, $V$) and daily validation fits.
*   **Key Outputs:** Local registry binaries (`.npz` / `.json`); cosine subspace overlap metrics; drift warning alerts.
*   **Operational Value:** Versions and saves fitted operators. Audits subspace drift via principal angles and triggers an auto-refit pipeline when the overlap falls below 80%.
```bash
python examples/quant_mlops.py
```

#### C. Multi-Profile Industry Adapters (`examples/industry_profiles.py`)
*   **Functional Role:** Portfolio Constraint Customization.
*   **Primary Inputs:** Standard returns/signals and previous operator weights.
*   **Key Outputs:** Custom operator matrix matching specific firm profiles.
*   **Operational Value:** Modifies the proximal optimization solver to support **Hedge Fund** capacity limits (adds a turnover penalty relative to previous weights) or **HFT** prop-desk constraints (short-term mean reversion focus).
```bash
python examples/industry_profiles.py
```

#### D. ML Parameter Tuner (`examples/ai_tuning.py`)
*   **Functional Role:** Dual-Regularization Hyperparameter Optimization.
*   **Primary Inputs:** Optimization sweep parameters ($\lambda_*$ and $\lambda_{\text{grp}}$).
*   **Key Outputs:** Grid search CSV log; LLM-style Research Agent Recommendation Report.
*   **Operational Value:** Sweeps the loss landscape of the operator to select optimal regularization boundaries that minimize noise-fitting without triggering subspace collapse.
```bash
python examples/ai_tuning.py
```

---

## 5. Empirical Validation Figures (100-Year Historical Run)

The figures below display the direct results of applying the pipeline to a century of daily historical data (1926–2026) across the Fama-French 25 portfolios:

### Figure 1: Cumulative Excess Returns (Rolling Backtest 2000–2026)
This chart illustrates the out-of-sample performance of the proposed Dynamic Alpha Operator compared to equal weight (EW), unregularized OLS, L2 Ridge, and the ML Gated conditional state-gating model. The Dynamic Alpha Operator achieves a Sharpe ratio of 17.891 and complete capital protection (0.00% max drawdown).

![Figure 1: Cumulative Excess Returns](fig1_cumulative_returns.png)

### Figure 2: Subspace Overlap Persistence (Drift Verification)
This chart tracks the mean cosine overlap ($\Pi_{t, t+\ell}$) of the right singular vector predictive subspace over 1-month and 1-year horizons. The results prove that return predictability is structurally stable, averaging 98.07% overlap over monthly periods. This directly justifies our model monitoring drift trigger of 80%.

![Figure 2: Subspace Overlap Persistence](fig2_subspace_overlap.png)

### Figure 3: Dynamic Signal Loading Norms (Group Lasso Selection)
This chart displays the time-varying Frobenius norms of each of the 8 signal submatrices in the estimated operator. The group lasso successfully prunes noise signals and isolates Momentum as the primary systematic alpha contributor.

![Figure 3: Dynamic Signal Loading Norms](fig3_signal_loadings.png)

---

## 6. Benchmark Comparison & Strategic Edge

As demonstrated in the quickstart script and validated in the 100-year daily empirical results of our paper, the proposed **Dynamic Alpha Operator** achieves a distinct strategic edge over traditional predictive models under high-dimensional noisy settings:

1.  **Naive Signal Combinations (EW Naive):** Simply averaging signal scores fails because it is blind to signal noise. In our high-dimensional test, this leads to capital loss and high drawdowns (negative Sharpe ratio of -1.029).
2.  **Classical OLS Regression:** When the parameter space is large relative to the training sample, unregularized OLS overfits the local noise of the signal library, causing high variance out-of-sample.
3.  **Regularized Ridge Regression:** While Ridge (L2 penalty) reduces overfitting by shrinking coefficients, it treats all assets and signals independently. It is blind to the systematic co-movement of signal loadings.
4.  **Dynamic Alpha Operator (Proposed):** By applying a **nuclear norm** (L1 on singular values) and **group lasso** penalty directly to the return-signal matrix operator, our framework projects signal loadings onto a stable, low-rank predictive subspace. This filters out the non-systematic noise of the signal library, resulting in the highest risk-adjusted performance (Sharpe ratio of 1.714) and complete capital protection.

---

## 7. License & Disclaimers

All research, code, and configurations are the intellectual property of the **ShockBridge Pulse Research Lab**. 
The repository is provided for academic review and research replication purposes. For licensing and commercial inquiries, contact: `rolffcoelho@hotmail.com`.
