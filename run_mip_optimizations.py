"""
MIP Portfolio Optimization Script
Runs combined MIP optimization for all m values and saves results to JSON.

Usage:
    python run_mip_optimizations.py

The script will:
- Load existing results if available (to avoid re-computation)
- Run MIP optimization for each m value with time limits
- Save results incrementally to JSON
- Log progress to console and file
"""

import os
import json
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import gurobipy as gp
from gurobipy import GRB, quicksum

# =============================================================================
# CONFIGURATION - Adjust these as needed
# =============================================================================

# Time limit for each MIP optimization (in seconds)
GUROBI_TIME_LIMIT = 3600  # 1 hour per m value
# For testing/grading: set to smaller value (e.g., 300 = 5 minutes)

# Output files
RESULTS_FILE = "mip_results.json"
LOG_FILE = "mip_optimization.log"

# m values to optimize
M_VALUES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# =============================================================================
# SETUP LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def get_nasdaq100_tickers():
    """Download NASDAQ-100 ticker list from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    tables = pd.read_html(response.text)
    
    for table in tables:
        if 'Ticker' in table.columns:
            tickers = table['Ticker'].tolist()
            tickers = [str(ticker).strip() for ticker in tickers
                      if pd.notna(ticker) and ticker != 'nan']
            return tickers
    
    raise ValueError("Could not find NASDAQ-100 ticker list")

def download_data(tickers, start_date, end_date):
    """Download stock price data and calculate returns."""
    logger.info(f"Downloading data from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date, 
                       progress=False, auto_adjust=False)
    
    if data.empty:
        raise ValueError(f"No data downloaded for tickers: {tickers}")
    
    # Extract Adj Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close']
    else:
        # Single ticker case
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers if isinstance(tickers, str) else tickers[0])
    
    # Forward fill and backward fill to handle missing days
    prices = prices.ffill().bfill()
    
    # Drop columns with any remaining NaN values (stocks that couldn't be filled)
    prices = prices.dropna(axis=1)
    
    if prices.empty:
        raise ValueError(f"No valid price data after processing")
    
    # Warn about dropped tickers
    if isinstance(tickers, list):
        dropped = set(tickers) - set(prices.columns)
        if dropped:
            logger.warning(f"Dropped {len(dropped)} tickers with missing data: {list(dropped)[:10]}{'...' if len(dropped) > 10 else ''}")
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    if returns.empty:
        raise ValueError(f"No valid returns calculated")
    
    logger.info(f"Downloaded {len(returns)} days of data for {len(returns.columns)} stocks")
    return prices, returns

# =============================================================================
# MIP OPTIMIZATION FUNCTION
# =============================================================================

def fit_sparse_weights_MIP_L1(rets_mat, q_vec, m, env, time_limit_sec=3600, 
                              nonneg=True, verbose=False):
    """
    Combined MIP: Select m stocks AND optimize weights simultaneously.
    
    Parameters:
        rets_mat: T×N matrix of stock returns
        q_vec: T×1 vector of index returns
        m: Number of stocks to select
        env: Gurobi environment
        time_limit_sec: Maximum time (seconds) for optimization
        nonneg: Whether weights must be non-negative
        verbose: Whether to print Gurobi output
    
    Returns:
        w_sol: Optimal weights (numpy array)
        sel_idx: Selected stock indices (list)
        total_L1: Total L1 tracking error
        status: Gurobi optimization status
    """
    T, N = rets_mat.shape
    mdl = gp.Model("sparse_weight_fit_L1", env=env)
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.TimeLimit = time_limit_sec
    
    # Decision variables
    w = mdl.addVars(N, lb=0.0 if nonneg else -GRB.INFINITY, name="w")
    y = mdl.addVars(N, vtype=GRB.BINARY, name="y")
    e = mdl.addVars(T, lb=0.0, name="e")
    
    # L1 linearization: e[t] >= |q[t] - portfolio_return[t]|
    for t in range(T):
        mdl.addConstr(e[t] >= q_vec[t] - quicksum(w[j] * rets_mat[t, j] 
                                                   for j in range(N)))
        mdl.addConstr(e[t] >= -q_vec[t] + quicksum(w[j] * rets_mat[t, j] 
                                                    for j in range(N)))
    
    # Budget constraint: weights sum to 1
    mdl.addConstr(quicksum(w[j] for j in range(N)) == 1.0)
    
    # Sparsity constraint: select exactly m stocks
    mdl.addConstr(quicksum(y[j] for j in range(N)) == m)
    
    # Big-M linking: w[j] can only be > 0 if y[j] = 1
    M = 1.0  # Tightest possible value
    for j in range(N):
        mdl.addConstr(w[j] <= M * y[j])
    
    # Minimize total L1 tracking error
    mdl.setObjective(quicksum(e[t] for t in range(T)), GRB.MINIMIZE)
    
    # Optimize
    start_time = time.time()
    mdl.optimize()
    solve_time = time.time() - start_time
    
    # Extract solution
    if mdl.status == GRB.OPTIMAL:
        w_sol = np.array([w[j].X for j in range(N)])
        sel_idx = [j for j in range(N) if y[j].X > 0.5]
        total_L1 = sum(e[t].X for t in range(T))
        status = "OPTIMAL"
    elif mdl.status == GRB.TIME_LIMIT:
        # Use best solution found so far
        w_sol = np.array([w[j].X for j in range(N)])
        sel_idx = [j for j in range(N) if y[j].X > 0.5]
        total_L1 = sum(e[t].X for t in range(T))
        status = "TIME_LIMIT"
        logger.warning(f"Timeout for m={m}, using best solution found")
    elif mdl.status == GRB.INFEASIBLE:
        w_sol = None
        sel_idx = None
        total_L1 = None
        status = "INFEASIBLE"
        logger.error(f"Problem infeasible for m={m}")
    else:
        w_sol = None
        sel_idx = None
        total_L1 = None
        status = f"STATUS_{mdl.status}"
        logger.error(f"Optimization failed for m={m}, status: {mdl.status}")
    
    return w_sol, sel_idx, total_L1, status, solve_time

# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

def run_mip_optimizations():
    """Main function to run MIP optimizations for all m values."""
    
    logger.info("="*70)
    logger.info("Starting MIP Portfolio Optimization")
    logger.info(f"Time limit per optimization: {GUROBI_TIME_LIMIT} seconds")
    logger.info(f"m values to optimize: {M_VALUES}")
    logger.info("="*70)
    
    # Load existing results if available
    results = {}
    if os.path.exists(RESULTS_FILE):
        logger.info(f"Loading existing results from {RESULTS_FILE}")
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        logger.info(f"Found {len(results)} existing results")
    else:
        logger.info("No existing results found, starting fresh")
    
    # =========================================================================
    # SETUP GUROBI ENVIRONMENT
    # =========================================================================
    
    logger.info("Initializing Gurobi environment...")
    params = {
        "WLSACCESSID": os.getenv("GRB_WLSACCESSID", '35a382ab-2f66-4f8c-93e5-29e3ae8a72c9'),
        "WLSSECRET": os.getenv("GRB_WLSSECRET", '697d9c6e-319b-4535-8f62-9fd78101f77d'),
        "LICENSEID": int(os.getenv("GRB_LICENSEID", "2610074")),
        "OutputFlag": 0,
        "CSClientLog": 0,
    }
    env = gp.Env(params=params)
    env.start()
    logger.info("Gurobi environment ready")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    logger.info("Loading data...")
    
    # Get tickers
    tickers = get_nasdaq100_tickers()
    logger.info(f"Found {len(tickers)} NASDAQ-100 tickers")
    
    # Download 2024 data (training)
    prev_prices, prev_returns = download_data(tickers, '2024-01-01', '2024-12-30')
    
    # Download 2025 data (testing)
    curr_prices, curr_returns = download_data(tickers, '2025-01-01', '2025-10-17')
    
    # Download index data
    index_ticker = "^NDX"
    prev_index_prices, prev_index_returns = download_data(index_ticker, '2024-01-01', '2024-12-30')
    curr_index_prices, curr_index_returns = download_data(index_ticker, '2025-01-01', '2025-10-17')
    
    # Align dates - ensure we only use dates that exist in both stock and index data
    common_dates_2024 = prev_returns.index.intersection(prev_index_returns.index)
    common_dates_2025 = curr_returns.index.intersection(curr_index_returns.index)
    
    if len(common_dates_2024) == 0:
        raise ValueError("No common dates between stock returns and index returns for 2024")
    if len(common_dates_2025) == 0:
        raise ValueError("No common dates between stock returns and index returns for 2025")
    
    prev_returns = prev_returns.loc[common_dates_2024]
    curr_returns = curr_returns.loc[common_dates_2025]
    
    # Ensure we only use tickers that have data in both periods
    common_tickers = prev_returns.columns.intersection(curr_returns.columns)
    if len(common_tickers) == 0:
        raise ValueError("No common tickers between 2024 and 2025 data")
    
    prev_returns = prev_returns[common_tickers]
    curr_returns = curr_returns[common_tickers]
    
    # Update tickers list to only include those with valid data
    tickers = [t for t in tickers if t in common_tickers]
    
    # Get index returns for aligned dates
    q_2024 = prev_index_returns.loc[common_dates_2024, "^NDX"]
    q_2025 = curr_index_returns.loc[common_dates_2025, "^NDX"]
    
    # Validate data alignment
    if len(q_2024) != len(prev_returns):
        raise ValueError(f"Date alignment error: q_2024 has {len(q_2024)} days, prev_returns has {len(prev_returns)} days")
    if len(q_2025) != len(curr_returns):
        raise ValueError(f"Date alignment error: q_2025 has {len(q_2025)} days, curr_returns has {len(curr_returns)} days")
    
    # Convert to numpy arrays
    R24 = prev_returns.values   # T24 x N
    R25 = curr_returns.values    # T25 x N
    q24 = q_2024.values          # T24
    q25 = q_2025.values          # T25
    
    logger.info(f"Data prepared: R24 shape={R24.shape}, R25 shape={R25.shape}")
    logger.info(f"Date ranges: 2024={len(R24)} days, 2025={len(R25)} days")
    logger.info(f"Using {len(tickers)} stocks with valid data")
    
    # Validate we have enough data
    if len(R24) == 0:
        raise ValueError("No 2024 data available after processing")
    if len(R25) == 0:
        raise ValueError("No 2025 data available after processing")
    if len(tickers) < max(M_VALUES):
        logger.warning(f"Only {len(tickers)} stocks available, but max m={max(M_VALUES)}. "
                      f"Some m values will be skipped.")
    
    # Store data info for later use (with error handling for empty DataFrames)
    data_info = {
        "num_stocks": len(tickers),
        "num_days_2024": len(R24),
        "num_days_2025": len(R25),
        "tickers": tickers,
        "date_range_2024": [str(prev_returns.index[0]), str(prev_returns.index[-1])] if len(prev_returns) > 0 else ["", ""],
        "date_range_2025": [str(curr_returns.index[0]), str(curr_returns.index[-1])] if len(curr_returns) > 0 else ["", ""]
    }
    
    # =========================================================================
    # RUN OPTIMIZATIONS
    # =========================================================================
    
    total_start_time = time.time()
    
    for m in M_VALUES:
        m_key = str(m)  # JSON keys must be strings
        
        # Skip if already computed
        if m_key in results and results[m_key].get('L1_2024') is not None:
            logger.info(f"m={m}: Skipping (already computed)")
            continue
        
        # Handle case where m >= number of stocks
        if m >= R24.shape[1]:
            logger.info(f"m={m}: Skipping (m >= number of stocks)")
            continue
        
        logger.info("")
        logger.info("-"*70)
        logger.info(f"Optimizing for m={m} (time limit: {GUROBI_TIME_LIMIT}s)...")
        logger.info("-"*70)
        
        try:
            # Run MIP optimization
            w_mip, sel_idx_mip, L1_2024, status, solve_time = fit_sparse_weights_MIP_L1(
                R24, q24, m, env=env, 
                time_limit_sec=GUROBI_TIME_LIMIT, 
                nonneg=True, 
                verbose=False
            )
            
            if w_mip is None:
                logger.error(f"m={m}: Optimization failed with status {status}")
                results[m_key] = {
                    "m": m,
                    "status": status,
                    "L1_2024": None,
                    "L1_2025": None,
                    "selected_indices": None,
                    "solve_time": solve_time
                }
            else:
                # Evaluate on 2025 data
                pred25 = R25 @ w_mip
                L1_2025 = float(np.sum(np.abs(q25 - pred25)))
                
                # Store results
                results[m_key] = {
                    "m": m,
                    "status": status,
                    "L1_2024": float(L1_2024),
                    "L1_2025": float(L1_2025),
                    "selected_indices": sel_idx_mip,
                    "selected_tickers": [tickers[i] for i in sel_idx_mip],
                    "weights": w_mip.tolist(),  # Convert numpy array to list for JSON
                    "solve_time": float(solve_time),
                    "num_selected": len(sel_idx_mip)
                }
                
                logger.info(f"m={m}: Completed in {solve_time:.1f}s")
                logger.info(f"  Status: {status}")
                logger.info(f"  L1_2024 (in-sample): {L1_2024:.6f}")
                logger.info(f"  L1_2025 (out-of-sample): {L1_2025:.6f}")
                logger.info(f"  Selected stocks: {len(sel_idx_mip)}")
            
            # Save results incrementally (in case of crash)
            results_full = {
                "metadata": {
                    "created_at": results.get("metadata", {}).get("created_at", datetime.now().isoformat()),
                    "last_updated": datetime.now().isoformat(),
                    "time_limit_per_m": GUROBI_TIME_LIMIT,
                    "m_values": M_VALUES
                },
                "data_info": data_info,
                "results": results
            }
            
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results_full, f, indent=2)
            
            logger.info(f"Results saved to {RESULTS_FILE}")
            
        except Exception as e:
            logger.error(f"m={m}: Error during optimization: {str(e)}", exc_info=True)
            results[m_key] = {
                "m": m,
                "status": "ERROR",
                "error": str(e),
                "L1_2024": None,
                "L1_2025": None
            }
            # Still save partial results
            with open(RESULTS_FILE, 'w') as f:
                json.dump({"metadata": {"last_updated": datetime.now().isoformat()},
                          "data_info": data_info,
                          "results": results}, f, indent=2)
    
    total_time = time.time() - total_start_time
    logger.info("")
    logger.info("="*70)
    logger.info("Optimization complete!")
    logger.info(f"Total time: {total_time/3600:.2f} hours ({total_time:.1f} seconds)")
    logger.info(f"Results saved to: {RESULTS_FILE}")
    logger.info("="*70)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    results_list = []
    for m in M_VALUES:
        m_key = str(m)
        if m_key in results and results[m_key].get('L1_2024') is not None:
            r = results[m_key]
            results_list.append({
                "m": r["m"],
                "Status": r["status"],
                "In-Sample L1 (2024)": f"{r['L1_2024']:.6f}" if r['L1_2024'] else "N/A",
                "Out-of-Sample L1 (2025)": f"{r['L1_2025']:.6f}" if r['L1_2025'] else "N/A",
                "Solve Time (s)": f"{r.get('solve_time', 0):.1f}"
            })
    
    if results_list:
        df_summary = pd.DataFrame(results_list)
        print(df_summary.to_string(index=False))
    else:
        print("No results computed yet.")
    
    print("="*70)

# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    try:
        run_mip_optimizations()
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        print("\nScript interrupted. Results saved up to this point.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\nFatal error: {str(e)}")
        print("Check the log file for details.")

