# MIP Portfolio Optimization Scripts

This directory contains scripts to run the computationally expensive MIP (Mixed-Integer Programming) portfolio optimizations that can take many hours to complete.

## Files

- **`run_mip_optimizations.py`** - Main script to run MIP optimizations for all m values
- **`visualize_mip_results.py`** - Script to load and visualize results from JSON
- **`mip_results.json`** - Output file containing all results (created by run script)
- **`mip_optimization.log`** - Log file with detailed progress information

## Quick Start

### 1. Run Optimizations

```bash
python run_mip_optimizations.py
```

This script will:
- Load existing results if `mip_results.json` exists (skips already-computed m values)
- Run MIP optimization for each m in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
- Save results incrementally to JSON (so you don't lose progress if interrupted)
- Log progress to both console and `mip_optimization.log`

**Expected Runtime**: Up to 10 hours (10 m values × 1 hour each)

**Tip**: Run this overnight or when you don't need your computer!

### 2. Visualize Results

```bash
python visualize_mip_results.py
```

This will create plots and summary tables from the saved results.

## Configuration

### Time Limit

Edit the `GUROBI_TIME_LIMIT` variable at the top of `run_mip_optimizations.py`:

```python
GUROBI_TIME_LIMIT = 3600  # 1 hour per optimization
```

For testing or grading, set to a smaller value:
```python
GUROBI_TIME_LIMIT = 300  # 5 minutes per optimization
```

### M Values

Edit the `M_VALUES` list to change which m values to optimize:

```python
M_VALUES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
```

## Output Format

The JSON file contains:

```json
{
  "metadata": {
    "created_at": "2025-01-15T10:30:00",
    "last_updated": "2025-01-15T18:45:00",
    "time_limit_per_m": 3600,
    "m_values": [5, 10, 20, ...]
  },
  "data_info": {
    "num_stocks": 100,
    "num_days_2024": 252,
    "num_days_2025": 200,
    "tickers": ["AAPL", "MSFT", ...],
    ...
  },
  "results": {
    "5": {
      "m": 5,
      "status": "OPTIMAL",
      "L1_2024": 0.961257,
      "L1_2025": 0.879060,
      "selected_indices": [0, 5, 12, ...],
      "selected_tickers": ["AAPL", "MSFT", ...],
      "weights": [0.2, 0.15, 0.3, ...],
      "solve_time": 3542.5,
      "num_selected": 5
    },
    "10": { ... },
    ...
  }
}
```

## Features

### Incremental Saving
- Results are saved after each m value completes
- If the script crashes or is interrupted, you can restart and it will skip already-computed values
- No need to start over!

### Error Handling
- If one m value fails, the script continues with the next
- Errors are logged but don't stop the entire process
- Partial results are always saved

### Status Tracking
- Each result includes optimization status:
  - `OPTIMAL`: Found optimal solution
  - `TIME_LIMIT`: Time limit reached (using best solution found)
  - `INFEASIBLE`: Problem has no solution
  - `ERROR`: An error occurred

### Logging
- All progress is logged to `mip_optimization.log`
- Console output shows real-time progress
- Timestamps on all log entries

## Example Usage

### Run overnight:
```bash
# Start at 10 PM
python run_mip_optimizations.py

# Check progress in the morning
cat mip_optimization.log | tail -20

# Visualize results
python visualize_mip_results.py
```

### Resume after interruption:
```bash
# Script will automatically detect existing results
python run_mip_optimizations.py

# It will skip m=5, m=10 if already computed
# Only runs remaining m values
```

## Troubleshooting

### Script stops running
- Check `mip_optimization.log` for errors
- Results up to the crash are saved in JSON
- Restart script - it will resume from where it left off

### Gurobi license issues
- Make sure your Gurobi license is valid
- Check environment variables for license credentials
- See the script for license configuration

### Out of memory
- Reduce the number of m values in one run
- Close other applications
- Consider running with smaller time limits first

## Comparison with Notebook

This script runs the **combined MIP approach** which is much slower than the two-step approach in the notebook, but potentially finds better solutions by directly optimizing tracking error.

To compare results:
1. Run the notebook's two-step approach
2. Run this MIP script
3. Compare L1_2025 (out-of-sample) values
4. See which method gives better tracking performance

