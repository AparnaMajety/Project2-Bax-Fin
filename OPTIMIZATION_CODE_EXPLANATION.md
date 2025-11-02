# Optimization Code Explanation & Verification

## Overview

This notebook implements a two-step approach for constructing a sparse portfolio that tracks the NASDAQ-100 index:

1. **Stock Selection**: Uses Integer Programming (IP) to select m stocks based on correlation
2. **Weight Optimization**: Uses Linear Programming (LP) with L1 minimization to find optimal weights

---

## Part 1: Stock Selection (`select_m_stocks_by_IP`)

### Purpose
Selects m stocks from the universe of N stocks that best represent the correlation structure.

### Mathematical Formulation

```
Maximize: Σᵢⱼ ρᵢⱼ · xᵢⱼ

Subject to:
  Σⱼ yⱼ = m                    (select exactly m stocks)
  Σⱼ xᵢⱼ = 1, ∀i              (each stock assigned to one representative)
  xᵢⱼ ≤ yⱼ, ∀i,j              (can only assign to selected stocks)
  xᵢⱼ, yⱼ ∈ {0,1}             (binary variables)
```

### Variables:
- **x[i,j]**: Binary variable = 1 if stock i is represented by stock j
- **y[j]**: Binary variable = 1 if stock j is selected
- **ρ[i,j]**: Correlation between stocks i and j

### Intuition:
This is a clustering/assignment problem where:
- We want to select m "representative" stocks (centers)
- Each stock gets assigned to its most correlated representative
- We maximize the total correlation between stocks and their representatives

### Code Implementation:
```python
def select_m_stocks_by_IP(rho, m, tickers, env, verbose=False):
    n = rho.shape[0]
    mdl = gp.Model("stock_selection", env=env)
    
    # Decision variables
    x = mdl.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = mdl.addVars(n, vtype=GRB.BINARY, name="y")
    
    # Objective: maximize sum of correlations
    mdl.setObjective(quicksum(rho[i, j]*x[i, j] 
                    for i in range(n) for j in range(n)), 
                    GRB.MAXIMIZE)
    
    # Constraint: select exactly m stocks
    mdl.addConstr(quicksum(y[j] for j in range(n)) == m)
    
    # Constraint: each stock assigned to exactly one representative
    for i in range(n):
        mdl.addConstr(quicksum(x[i, j] for j in range(n)) == 1)
    
    # Constraint: can only assign to selected stocks
    for i in range(n):
        for j in range(n):
            mdl.addConstr(x[i, j] <= y[j])
    
    mdl.optimize()
    
    # Extract selected stocks
    selected_idx = [j for j in range(n) if y[j].X > 0.5]
    return selected_idx, selected_tickers, rep_of_i
```

---

## Part 2: Weight Optimization (`fit_weights_L1`)

### Purpose
Finds optimal portfolio weights wᵢ for the selected m stocks to minimize L1 tracking error.

### Mathematical Formulation

```
Minimize: Σₜ |qₜ - Σⱼ wⱼ·Rₜⱼ|

Subject to:
  Σⱼ wⱼ = 1              (weights sum to 1)
  wⱼ ≥ 0, ∀j ∈ selected   (non-negative weights)
```

### Variables:
- **w[j]**: Weight of stock j (only for selected stocks)
- **q[t]**: Index return at time t
- **R[t,j]**: Return of stock j at time t
- **e[t]**: Absolute error at time t (linearized)

### Linearization of Absolute Value:
The absolute value |qₜ - Σⱼ wⱼ·Rₜⱼ| is replaced with:
- e[t] ≥ qₜ - Σⱼ wⱼ·Rₜⱼ
- e[t] ≥ -qₜ + Σⱼ wⱼ·Rₜⱼ

This ensures e[t] ≥ |error|, and minimization will make it equal.

### Code Implementation:
```python
def fit_weights_L1(R, q, sel_idx, env, nonneg=True, verbose=False):
    T, N = R.shape
    J = sel_idx  # Only selected stocks
    
    mdl = gp.Model("weight_fit_L1", env=env)
    
    # Variables: weights only for selected stocks
    w = mdl.addVars(J, lb=0.0 if nonneg else -GRB.INFINITY, name="w")
    e = mdl.addVars(T, lb=0.0, name="e")  # Error variables
    
    # Linearize absolute value: e[t] >= |q[t] - Σw[j]*R[t,j]|
    for t in range(T):
        mdl.addConstr(e[t] >= q[t] - quicksum(w[j]*R[t, j] for j in J))
        mdl.addConstr(e[t] >= -q[t] + quicksum(w[j]*R[t, j] for j in J))
    
    # Budget constraint: weights sum to 1
    mdl.addConstr(quicksum(w[j] for j in J) == 1.0)
    
    # Minimize total L1 error
    mdl.setObjective(quicksum(e[t] for t in range(T)), GRB.MINIMIZE)
    mdl.optimize()
    
    # Return full weight vector (zeros for non-selected)
    w_full = np.zeros(N)
    for j in J:
        w_full[j] = w[j].X
    
    L1 = sum(e[t].X for t in range(T))
    return w_full, L1
```

---

## Part 3: Combined MIP Approach (`fit_sparse_weights_MIP_L1`)

### Quick Summary
**What**: Single optimization that selects m stocks AND finds optimal weights simultaneously  
**How**: Uses binary variables (y) + big-M constraints to link selection to weights  
**Why**: Directly optimizes tracking error (better than two-step correlation approach)  
**Cost**: Very slow - can take hours per m value, may need time limits  
**Best M**: M = 1.0 (smallest/tightest possible value)  

### Purpose
**Alternative method**: Instead of two separate optimizations (stock selection → weight fitting), combine everything into a **single Mixed-Integer Program (MIP)** that simultaneously selects m stocks AND optimizes their weights.

### Key Difference from Two-Step Approach

**Two-Step Method**:
1. Step 1: Select m stocks using correlation-based IP (ignores tracking error)
2. Step 2: Optimize weights for selected stocks using L1 minimization

**Combined MIP Method**:
1. One optimization: Directly minimizes L1 tracking error while selecting m stocks

### Mathematical Formulation

Instead of optimizing only over selected stocks, optimize over **ALL stocks**:

```
Minimize: Σₜ |qₜ - Σⱼ₌₁ⁿ wⱼ·Rₜⱼ|

Subject to:
  Σⱼ₌₁ⁿ wⱼ = 1              (weights sum to 1)
  Σⱼ₌₁ⁿ yⱼ = m             (select exactly m stocks)
  wⱼ ≤ M·yⱼ, ∀j ∈ {1,...,n}  (big-M: weight only if selected)
  wⱼ ≥ 0                     (non-negative weights)
  yⱼ ∈ {0,1}                 (binary selection variables)
```

**Key Variables**:
- **wⱼ**: Weight of stock j (continuous, for ALL n stocks, not just selected ones)
- **yⱼ**: Binary indicator = 1 if stock j is selected, 0 otherwise
- **M**: Big-M constant (we'll show why M = 1.0 is optimal)

### The Big-M Technique: Linking Selection to Weights

The big-M constraint `wⱼ ≤ M·yⱼ` creates a logical relationship:
- **If yⱼ = 0** (stock not selected):
  - Constraint: `wⱼ ≤ M·0 = 0`
  - Combined with `wⱼ ≥ 0`: Forces `wⱼ = 0` ✓

- **If yⱼ = 1** (stock selected):
  - Constraint: `wⱼ ≤ M·1 = M`
  - Stock can have positive weight (up to M)

### Why M = 1.0 is the Smallest (Tightest) Value

**Question**: What's the smallest value of M that works?

**Answer**: M = 1.0

**Proof**:
1. We have the constraint: `Σⱼ wⱼ = 1` (weights sum to 1)
2. We have: `wⱼ ≥ 0` (non-negative)
3. Therefore: Each individual weight must satisfy `wⱼ ≤ 1`
   - If any weight were > 1, and others are ≥ 0, the sum would exceed 1
4. So the tightest bound is: `wⱼ ≤ 1`

**Why Tight is Better**:
- Smaller M values = tighter constraints = faster optimization
- M = 1.0 is the **smallest valid value** (can't use M < 1.0)
- Using M > 1.0 (e.g., M = 100) would still work but is less efficient

**Visual Example**:
```
With M = 1.0 (tight):
  If y[j] = 1: w[j] ≤ 1.0  ← Exactly what we need (weights ≤ 1)
  
With M = 100 (loose):
  If y[j] = 1: w[j] ≤ 100  ← Too permissive, makes optimization harder
```

### Why This Problem is Computationally Hard

**Complexity Factors**:

1. **Binary Variables**: n binary variables (yⱼ) create 2ⁿ possible combinations
   - For n = 100 stocks, that's 2¹⁰⁰ ≈ 10³⁰ possibilities!

2. **Mixed-Integer Nature**: Combines continuous (weights) and discrete (selection) decisions
   - Solver must explore integer solutions, not just continuous relaxation

3. **No Closed-Form Solution**: Unlike pure linear programs, MIPs require branch-and-bound search

4. **Large Problem Size**: 
   - n stocks (100+)
   - T time periods (250+ trading days)
   - Creates O(n·T) constraints

**Real-World Performance**:
- Small problems (m ≤ 10, n ≤ 50): May solve in minutes
- Medium problems (m = 20-30, n = 100): May take hours
- Large problems (m ≥ 50, n = 100+): Can take 24+ hours or never converge!

### Code Implementation

```python
def fit_sparse_weights_MIP_L1(rets_mat: np.ndarray, q_vec: np.ndarray, 
                                m: int, env, time_limit_sec=3600, 
                                nonneg=True, verbose=False):
    """
    Combined MIP: Select m stocks AND optimize weights simultaneously.
    
    Parameters:
        rets_mat: T×N matrix of stock returns
        q_vec: T×1 vector of index returns
        m: Number of stocks to select
        time_limit_sec: Maximum time (seconds) for optimization
    """
    T, N = rets_mat.shape
    mdl = gp.Model("sparse_weight_fit_L1", env=env)
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.TimeLimit = time_limit_sec  # ⚠️ Critical: prevent infinite runtime
    
    # Decision variables
    w = mdl.addVars(N, lb=0.0 if nonneg else -GRB.INFINITY, name="w")
    y = mdl.addVars(N, vtype=GRB.BINARY, name="y")  # Binary selection
    e = mdl.addVars(T, lb=0.0, name="e")            # Absolute errors
    
    # L1 linearization (same as before)
    for t in range(T):
        mdl.addConstr(e[t] >= q_vec[t] - gp.quicksum(w[j] * rets_mat[t, j] 
                                                      for j in range(N)))
        mdl.addConstr(e[t] >= -q_vec[t] + gp.quicksum(w[j] * rets_mat[t, j] 
                                                       for j in range(N)))
    
    # Budget constraint: weights sum to 1
    mdl.addConstr(gp.quicksum(w[j] for j in range(N)) == 1.0)
    
    # Sparsity constraint: select exactly m stocks
    mdl.addConstr(gp.quicksum(y[j] for j in range(N)) == m)
    
    # Big-M linking: w[j] can only be > 0 if y[j] = 1
    M = 1.0  # Tightest possible value
    for j in range(N):
        mdl.addConstr(w[j] <= M * y[j])
    
    # Minimize total L1 tracking error
    mdl.setObjective(gp.quicksum(e[t] for t in range(T)), GRB.MINIMIZE)
    mdl.optimize()
    
    # Extract solution
    w_sol = np.array([w[j].X for j in range(N)])
    sel_idx = [j for j in range(N) if y[j].X > 0.5]
    total_L1 = sum(e[t].X for t in range(T))
    
    return w_sol, sel_idx, total_L1
```

### Time Limits: Essential for Practical Use

**The Problem**: Without time limits, Gurobi may run for days without finding a solution.

**The Solution**: Set `mdl.Params.TimeLimit = time_limit_sec`

**Recommendation from Instructions**:
- Set time limit to **3600 seconds (1 hour)** per m value
- Total runtime: Up to 10 hours for m = 5,10,20,...,90,100 (10 values × 1 hour)
- **Create a configurable constant** at the top of your code:

```python
# Configurable time limit (in seconds)
GUROBI_TIME_LIMIT = 3600  # 1 hour per optimization
# For grading: set to smaller value (e.g., 300 seconds = 5 minutes)
```

### Saving and Loading Results

**Why**: Avoid re-running expensive optimizations if results already exist.

**Implementation Strategy**:

```python
import pandas as pd
import os

TIME_LIMIT = 3600  # Configurable at top of file
RESULTS_FILE = "mip_results.csv"

def run_mip_optimizations(R24, q24, R25, q25, tickers, env):
    """Run MIP optimization for all m values, with caching."""
    
    # Check if results already exist
    if os.path.exists(RESULTS_FILE):
        print(f"Loading existing results from {RESULTS_FILE}")
        results = pd.read_csv(RESULTS_FILE).to_dict('records')
        return results
    
    # Otherwise, run optimizations
    results = []
    m_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for m in m_values:
        if m >= R24.shape[1]:
            # Use all stocks
            continue
            
        print(f"Optimizing for m={m} (time limit: {TIME_LIMIT}s)...")
        
        # Run MIP with time limit
        w_mip, sel_idx_mip, L1_2024 = fit_sparse_weights_MIP_L1(
            R24, q24, m, env=env, time_limit_sec=TIME_LIMIT, nonneg=True
        )
        
        # Evaluate on 2025 data
        L1_2025 = float(np.sum(np.abs(q25 - R25 @ w_mip)))
        
        results.append({
            "m": m,
            "L1_2024": L1_2024,
            "L1_2025": L1_2025
        })
        
        # Save incrementally (in case of crash)
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
        print(f"  Completed m={m}: L1_2024={L1_2024:.6f}, L1_2025={L1_2025:.6f}")
    
    return results
```

### Comparison: Two-Step vs Combined MIP

| Aspect | Two-Step Approach | Combined MIP Approach |
|-------|-------------------|----------------------|
| **Optimization Steps** | 2 separate problems | 1 integrated problem |
| **Stock Selection** | Based on correlation | Directly optimizes tracking error |
| **Speed** | Fast (minutes) | Very slow (hours) |
| **Solution Quality** | Suboptimal (local) | Potentially optimal (global) |
| **Convergence** | Always finds solution | May timeout |
| **Practical Use** | Suitable for real-time | Requires overnight runs |

### When to Use Each Approach

**Use Two-Step When**:
- You need results quickly
- Problem is large (n > 50 stocks)
- You want guaranteed convergence
- Good enough solution is acceptable

**Use Combined MIP When**:
- You need the best possible solution
- Problem is small-medium (n ≤ 100)
- You can wait hours for results
- You want to directly optimize tracking error

### Advantages of Combined MIP:
- ✅ **Directly optimizes tracking error** (the actual objective)
- ✅ **Potentially finds globally optimal solution**
- ✅ **No need for separate stock selection heuristic**

### Disadvantages of Combined MIP:
- ❌ **Computationally very expensive** (can take 24+ hours)
- ❌ **May not converge within time limit**
- ❌ **Requires careful time management**
- ❌ **Solution may be infeasible if timeout occurs**

---

## Comparing Results: Which Method Works Better?

### Expected Findings

**Two-Step Method (Correlation-based selection)**:
- ✅ Fast and reliable
- ✅ Always finds solution
- ⚠️ May not select stocks that best track the index
- ⚠️ Optimizes correlation (proxy), not actual tracking error

**Combined MIP Method**:
- ⚠️ Slow and may timeout
- ✅ Directly optimizes tracking error
- ✅ Potentially finds better solution (if it converges)
- ⚠️ May return suboptimal solution if time limit reached

### How to Compare

After running both methods for all m values:

1. **Compare Out-of-Sample L1 (2025)**:
   ```python
   # Two-step results
   df_two_step = pd.DataFrame(two_step_results)
   
   # MIP results
   df_mip = pd.DataFrame(mip_results)
   
   # Compare out-of-sample performance
   comparison = pd.merge(
       df_two_step[['m', 'L1_2025']], 
       df_mip[['m', 'L1_2025']], 
       on='m', 
       suffixes=('_two_step', '_mip')
   )
   comparison['improvement'] = comparison['L1_2025_two_step'] - comparison['L1_2025_mip']
   comparison['better_method'] = comparison['improvement'].apply(
       lambda x: 'MIP' if x > 0 else 'Two-Step'
   )
   ```

2. **Visualize Comparison**:
   ```python
   plt.figure(figsize=(12, 6))
   plt.plot(comparison['m'], comparison['L1_2025_two_step'], 
            marker='o', label='Two-Step Method', linewidth=2)
   plt.plot(comparison['m'], comparison['L1_2025_mip'], 
            marker='s', label='Combined MIP Method', linewidth=2)
   plt.xlabel('Number of Stocks (m)')
   plt.ylabel('Out-of-Sample L1 Tracking Error (2025)')
   plt.title('Method Comparison: Out-of-Sample Performance')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()
   ```

### What to Look For

**MIP is Better When**:
- L1_2025_MIP < L1_2025_TwoStep consistently
- Difference is significant (e.g., > 5-10% improvement)
- MIP solutions converged (didn't timeout)

**Two-Step is Better When**:
- L1_2025_TwoStep < L1_2025_MIP
- MIP solutions timed out (may be suboptimal)
- Results are similar (two-step is "good enough")

**Typical Pattern**:
- Small m (5-20): MIP may find better solutions (fewer combinations to search)
- Medium m (30-50): MIP struggles, may timeout or produce similar results
- Large m (60-100): Two-step likely faster and may perform similarly

### Handling Timeout Issues

If MIP times out (after 1 hour):
- Check `mdl.status`:
  ```python
  if mdl.status == GRB.TIME_LIMIT:
      print(f"⚠️  Timeout for m={m}, using best solution found")
  elif mdl.status == GRB.OPTIMAL:
      print(f"✅ Optimal solution found for m={m}")
  elif mdl.status == GRB.INFEASIBLE:
      print(f"❌ Problem infeasible for m={m}")
  ```
- The solution may still be usable (best solution found so far)
- Compare to two-step: if MIP's partial solution is worse, two-step is better

---

## Verification Against Instructions

### ✅ Requirement 1: Data Collection
**Instructions**: Download NASDAQ-100 data for 2024 and 2025 using yfinance, get Adjusted Close prices, calculate returns, use correlation matrix as ρ.

**Code Verification**:
```python
# ✅ Downloads tickers from Wikipedia
tickers = get_nasdaq100_tickers()

# ✅ Downloads data for 2024 and 2025
prev_prices, prev_returns = download_data(tickers, '2024-01-01', '2024-12-30')
curr_prices, curr_returns = download_data(tickers, '2025-01-01', '2025-10-17')

# ✅ Gets index data
prev_index_prices, prev_index_returns = download_data('^NDX', '2024-01-01', '2024-12-30')
curr_index_prices, curr_index_returns = download_data('^NDX', '2025-01-01', '2025-10-17')

# ✅ Uses Adj Close and calculates returns
data = yf.download(..., progress=False, auto_adjust=False)['Adj Close']
returns = data.pct_change().dropna()

# ✅ Calculates correlation matrix
rho = prev_returns.corr().fillna(0).clip(-1,1).values
```
**Status**: ✅ **COMPLIANT**

---

### ✅ Requirement 2: m=5 Portfolio
**Instructions**: Find best 5 stocks and weights using 2024 data, evaluate on 2025 data using L1 tracking error.

**Code Verification**:
```python
# ✅ Selects 5 stocks
sel_idx, sel_tickers, rep = select_m_stocks_by_IP(rho, m=5, tickers=tickers, env=env)

# ✅ Fits weights on 2024 data
w_full_2024, in_sample_L1 = fit_weights_L1(R24, q24, sel_idx, env=env, nonneg=True)

# ✅ Evaluates on 2025 data
pred25 = R25 @ w_full_2024
oos_L1 = np.sum(np.abs(q25 - pred25))
```
**Status**: ✅ **COMPLIANT**

---

### ✅ Requirement 3: Multiple Values of m
**Instructions**: Redo for m = 10, 20, ..., 90, 100. Analyze performance, check for diminishing returns.

**Code Verification**:
```python
# ✅ Loops through m values
for m in [5,10,20,30,40,50,60,70,80,90,100]:
    # ✅ Handles m >= N case
    if m < R24.shape[1]:
        sel_idx, sel_tickers, rep = select_m_stocks_by_IP(rho, m, tickers, env=env)
    else:
        sel_idx = list(range(R24.shape[1]))  # All stocks
    
    # ✅ Calculates in-sample and out-of-sample L1
    w_full_2024, L1_2024 = fit_weights_L1(R24, q24, sel_idx, env=env, nonneg=True)
    L1_2025 = float(np.sum(np.abs(q25 - R25 @ w_full_2024)))
    
    # ✅ Stores results for analysis
    results.append({"m": m, "L1_2024": L1_2024, "L1_2025": L1_2025})
```

**Analysis**:
- ✅ Calculates in-sample L1 (2024 performance)
- ✅ Calculates out-of-sample L1 (2025 performance)
- ✅ Creates visualization to analyze diminishing returns
- ✅ Handles cases where m ≥ number of stocks

**Status**: ✅ **COMPLIANT**

---

## Issues & Recommendations

### ⚠️ Potential Issue: Stock Selection Objective

**Problem**: The `select_m_stocks_by_IP` function maximizes correlation between stocks and representatives, but this may not directly minimize tracking error.

**Impact**: 
- Selected stocks might have high correlation with each other
- But may not track the index well if index correlation patterns differ

**Recommendation**: 
- Consider using the combined approach `fit_sparse_weights_MIP_L1` for comparison
- Or modify stock selection to use index-correlation instead of stock-stock correlation

---

### ✅ Code Quality Notes

1. **Proper error handling**: Uses intersection of dates to ensure alignment
2. **Flexible m values**: Handles m ≥ N correctly
3. **Clear separation**: Stock selection and weight optimization are separate, making debugging easier
4. **Efficient linearization**: L1 minimization properly linearized

---

## Summary

The code correctly implements:
1. ✅ Data collection from NASDAQ-100
2. ✅ Two-step optimization (selection → weights)
3. ✅ L1 tracking error minimization
4. ✅ Evaluation on both in-sample and out-of-sample data
5. ✅ Analysis across multiple m values

The approach is **mathematically sound** and **follows the instructions correctly**. The main optimization uses a correlation-based stock selection followed by L1 weight optimization, which is a reasonable heuristic for index tracking.

