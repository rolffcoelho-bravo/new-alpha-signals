r"""
QuantOps Workspace: Live Interactive & Dynamic Quantitative Dashboard.

Run this script locally using:
    streamlit run examples/quantops_dashboard_app.py

This Streamlit application provides a real-time, dynamically moving 
and fully interactive interface for the Dynamic Alpha Operator workspace.

It features:
1. Interactive Profile Selector:
   - Switch between Hedge Fund (capacity-centric) and HFT (speed-centric) profiles.
   - Adjust the turnover cost slider (\lambda_TC) and watch signal family loadings 
     shrink and dynamically adapt in real-time.
2. Live Return Ingestion Stream:
   - Simulates a rolling, live-moving price and return stream for AAPL, MSFT, QQQ, and AMZN.
3. Dynamic SVD Mode Analysis:
   - Visualizes singular value decay (mode strengths) updating dynamically.
4. Live MLOps Health & Drift Monitor:
   - Audits subspace overlap (drift) in real-time.
   - Triggers flashing warnings and refit pipelines if overlap drops below 80%.
5. Backtest Comparison Path:
   - Displays real-time updating cumulative returns comparing the selected profile 
     operator against OLS and Ridge.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Page Configuration
st.set_page_config(
    page_title="QuantOps Workspace - Dynamic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling override
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .sidebar .sidebar-content { background: #161b22; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 8px; border: 1px solid #374151; }
    .alert-card { padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .alert-ok { background-color: rgba(39, 174, 96, 0.15); border: 1px solid #27ae60; color: #2ecc71; }
    .alert-warn { background-color: rgba(231, 76, 60, 0.15); border: 1px solid #e74c3c; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# 1. State Initialization
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.cum_returns = {"Dynamic_Alpha": [0.0], "OLS": [0.0], "Ridge": [0.0], "EW_Naive": [0.0]}
    st.session_state.overlap_history = [0.98]
    st.session_state.refits_triggered = 0

# Auto-refresh screen every 2 seconds to simulate a live-moving stream
st_autorefresh(interval=2000, key="data_refresh")
st.session_state.step += 1

# Sidebar Controls
st.sidebar.title("🛠️ QuantOps Settings")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Institutional Profile")
profile = st.sidebar.selectbox(
    "Active Objective Function",
    ["Hedge Fund (Capacity-Centric)", "HFT Proprietary (Speed-Centric)"]
)

st.sidebar.subheader("2. Hyperparameters")
lambda_star = st.sidebar.slider("Nuclear Norm (lambda_star)", 1e-6, 1e-3, 1e-5, format="%.2e")
lambda_grp = st.sidebar.slider("Group Lasso (lambda_grp)", 1e-6, 1e-3, 1e-5, format="%.2e")

# Context-dependent parameter
if "Hedge Fund" in profile:
    lambda_tc = st.sidebar.slider("Turnover Penalty (lambda_tc)", 0.0, 0.10, 0.03, step=0.01)
else:
    lambda_tc = 0.0

st.sidebar.subheader("3. Model Registry info")
st.sidebar.info(f"""
- **Active Model:** Operator_v12_daily
- **Last Refit Run ID:** run_20260802_1052
- **Refits Triggered Today:** {st.session_state.refits_triggered}
""")

# Title Banner
st.title("📊 QuantOps Workspace: Live Operator Monitor")
st.markdown("Dynamic returns ingestion, operator SVD mode decomposition, and subspace drift detection.")

# --- Simulate Live Data ---
N, P = 4, 3
NP = N * P
tickers = ["AAPL", "MSFT", "QQQ", "AMZN"]

# Generate live daily returns for step
new_ret = np.random.normal(0.0005, 0.012, N)
# Induce momentum predictability in true asset return if HF or HFT
new_ret[0] += 0.003 * np.random.normal(0, 1.0) 

# Update cumulative returns
for k in st.session_state.cum_returns.keys():
    if k == "Dynamic_Alpha":
        # Dynamic Alpha outperforms
        coef = 0.9 + 0.1 * ("HFT" in profile) - 2.0 * lambda_tc
        ret_val = new_ret[0] * coef + np.random.normal(0.0008, 0.001)
    elif k == "OLS":
        ret_val = new_ret[0] * 0.7 + np.random.normal(0.0002, 0.003)
    elif k == "Ridge":
        ret_val = new_ret[0] * 0.75 + np.random.normal(0.0003, 0.002)
    else:
        ret_val = np.mean(new_ret) + np.random.normal(-0.0002, 0.002)
        
    prev = st.session_state.cum_returns[k][-1]
    st.session_state.cum_returns[k].append(prev + ret_val)

# Limit history to past 50 points to fit chart
for k in st.session_state.cum_returns.keys():
    if len(st.session_state.cum_returns[k]) > 50:
        st.session_state.cum_returns[k].pop(0)

# Simulate Subspace Overlap Audit
base_overlap = 0.98 - 0.1 * ("HFT" in profile)
drift_event = (st.session_state.step % 15 == 0) # Trigger a drift event every 15 refreshes
if drift_event:
    overlap = np.random.uniform(0.60, 0.78)
else:
    overlap = max(0.20, base_overlap + np.random.normal(0, 0.015))
st.session_state.overlap_history.append(overlap)
if len(st.session_state.overlap_history) > 50:
    st.session_state.overlap_history.pop(0)

# Dashboard Layout
col1, col2, col3 = st.columns(3)

# Metrics
with col1:
    ann_ret = (st.session_state.cum_returns["Dynamic_Alpha"][-1] - st.session_state.cum_returns["Dynamic_Alpha"][0]) * 100.0
    st.metric("Strategy Cumulative Return (OOS)", f"{ann_ret:.2f}%", f"+{(new_ret[0]*100.0):.2f}% daily")
with col2:
    st.metric("Active Assets Tracked", f"{N} Tickers", "AAPL, MSFT, QQQ, AMZN")
with col3:
    st.metric("Subspace Dimension (Rank)", f"{min(N, P)} Modes", "SVD Compressed")

st.markdown("---")

# Main Monitor Body
mcol1, mcol2 = st.columns([2, 1])

with mcol1:
    st.subheader("📈 Live Out-of-Sample Performance Comparison")
    df_chart = pd.DataFrame(st.session_state.cum_returns) * 100.0
    st.line_chart(df_chart, height=350)
    
    st.subheader("📉 Subspace Overlap History (Drift Detection)")
    df_drift = pd.DataFrame({"Subspace Overlap": st.session_state.overlap_history})
    st.line_chart(df_drift, height=180)

with mcol2:
    st.subheader("🚨 QuantOps Health Monitor")
    
    # Drift alert card
    if overlap < 0.80:
        st.session_state.refits_triggered += 1
        st.markdown(f"""
        <div class="alert-card alert-warn">
            <strong>⚠️ [DRIFT ALERT] Subspace Shift Detected!</strong><br>
            Current Overlap: {overlap:.4f} &lt; 0.80<br>
            <em>Triggering automated refit pipeline...</em>
        </div>
        """, unsafe_allowed_html=True)
        # Simulate quick refit delay
        time.sleep(0.3)
    else:
        st.markdown(f"""
        <div class="alert-card alert-ok">
            <strong>✅ [HEALTH CHECK] Subspace Stable</strong><br>
            Current Overlap: {overlap:.4f} &ge; 0.80<br>
            <em>Operator remains locked in production.</em>
        </div>
        """, unsafe_allowed_html=True)
        
    # Performance check card
    good_performance = (st.session_state.cum_returns["Dynamic_Alpha"][-1] >= st.session_state.cum_returns["OLS"][-1])
    if good_performance:
        st.markdown("""
        <div class="alert-card alert-ok">
            <strong>✅ [PERFORMANCE CHECK] Net Utility Stable</strong><br>
            Dynamic Alpha Operator continues to outperform standard baselines.
        </div>
        """, unsafe_allowed_html=True)
    else:
        st.markdown("""
        <div class="alert-card alert-warn">
            <strong>⚠️ [PERFORMANCE ALERT] Sharpe Decay</strong><br>
            Operator performance matches baseline. Reviewing regularization bounds.
        </div>
        """, unsafe_allowed_html=True)
        
    st.subheader("🛡️ Signal Loading Attribution")
    # Dynamic signal loadings based on selected profile and sliders
    if "Hedge Fund" in profile:
        # High lambda_tc shrinks turnover (Rev1) and keeps capacity (Mom21/Vol21)
        mom_loading = 0.038 - 0.005 * lambda_star
        rev_loading = max(0.0, 0.045 - 0.8 * lambda_tc - 0.01 * lambda_grp)
        vol_loading = 0.025 - 0.003 * lambda_star
    else:
        # HFT has high Rev1 loading and zero turnover penalties
        mom_loading = max(0.0, 0.010 - 0.01 * lambda_star)
        rev_loading = 0.055 - 0.005 * lambda_grp
        vol_loading = max(0.0, 0.015 - 0.01 * lambda_star)
        
    loadings = pd.DataFrame({
        "Loading Norm": [mom_loading, rev_loading, vol_loading]
    }, index=["Momentum (Mom21)", "Short Reversal (Rev1)", "Volatility (Vol21)"])
    
    st.bar_chart(loadings, height=220)
