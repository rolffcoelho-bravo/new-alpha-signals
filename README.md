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

## 2. Platform Architecture

The repository is organized into a modular, production-ready python library:

```text
new-alpha-signals/
├── code/
│   ├── data_prep.py            # Data ingestion, normalization, and alignment pipeline
│   ├── estimator.py            # Reference PyTorch solver for the Alpha Operator
│   ├── empirical.py            # Out-of-sample evaluation and persistence diagnostics
│   ├── benchmarks.py           # Benchmark models and conditional state-gating modules
│   ├── utils.py                # High-performance matrix calculus and statistical utilities
│   └── visualization.py        # Publication-quality reporting and plotting suite
└── README.md
```

### Core Components:
*   **Data Preparation (`data_prep.py`):** Standardizes historical panels and constructs point-in-time signal libraries, ensuring no look-ahead leakage.
*   **Operator Estimator (`estimator.py`):** Solves the regularized loss minimization problem to extract the latent predictive directions.
*   **Out-of-Sample Suite (`empirical.py`):** Evaluates signal persistence and conducts non-parametric subspace overlap analysis.
*   **Benchmarks (`benchmarks.py`):** Compares performance against classical linear models, regularized regressions, and ML-based conditional deployment gates.

---

## 3. Targeted Applications

This framework is designed to deliver high-performance tools for:
*   **Portfolio Managers:** Quantifying hidden concentration in signal libraries and dynamically adjusting exposures.
*   **Quantitative Researchers:** Formulating rigorous rank and persistence tests to evaluate signal efficacy before capital allocation.
*   **Quantitative Developers:** Integrating structured signal packets into existing execution and risk management engines.

---

## 4. Academic Positioning

This platform serves as the replication codebase for the accompanying research paper:
> **Beyond Backtesting: A New Framework for Alpha Signals**  
> *Dynamic Statistical Identification, Risk-Orthogonalization, and Conditional Deployment*  
> **Author:** Rodolfo Pereira (rolffcoelho@hotmail.com)  
> **Institution:** ShockBridge Pulse Research Lab (www.shockbridgepulse.com)

The framework is positioned within the empirical asset pricing literature, proposing a novel methodology that optimizes signal compression directly under economic and portfolio constraints, rather than statistical variance.

---

## 5. License & Disclaimers

All research, code, and configurations are the intellectual property of the **ShockBridge Pulse Research Lab**. 
The repository is provided for academic review and research replication purposes. For licensing and commercial inquiries, contact: `rolffcoelho@hotmail.com`.
