# examples/optimize/example_optimize_strategy.py

from simple_trade import download_data, compute_indicator
from simple_trade import run_cross_trade, custom_optimizer

# --- Configuration ---
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-12-31"
initial_capital = 100000
commission_fee = 0.001 # 0.1%

# --- Download Data ---
print(f"Downloading data for {ticker}...")
data = download_data(ticker, start_date, end_date)

# Define a wrapper function to handle computing indicators and running the backtest
def run_cross_trade_with_windows(data, short_window, long_window, **kwargs):
    # Work on a copy of the data
    df = data.copy()
    
    # Compute the EMA indicators
    df, _, _ = compute_indicator(df, indicator='ema', parameters={'window': short_window}, columns={'close_col': 'Close'})
    df, _, _ = compute_indicator(df, indicator='ema', parameters={'window': long_window}, columns={'close_col': 'Close'})
    
    # Get the indicator column names
    short_window_indicator = f"EMA_{short_window}"
    long_window_indicator = f"EMA_{long_window}"
    
    # Run the backtest using function-based API
    return run_cross_trade(
        data=df,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        **kwargs
    )

# --- Optimization Parameters ---
# Define the parameter grid to search
param_grid = {
    'short_window': [10, 20, 30],
    'long_window': [50, 100, 150],
}

# Define constant parameters for the backtester
constant_params = {
    'initial_cash': initial_capital, 
    'commission_long': commission_fee,
    'price_col': 'Close'
}

# Define the metric to optimize and whether to maximize or minimize
metric_to_optimize = 'total_return_pct'
maximize_metric = True

# --- Instantiate and Run Optimizer ---
print("Initializing Optimizer...")
print("\nRunning Optimization (Parallel)...")
# Run optimization with parallel processing (adjust n_jobs as needed)
results = custom_optimizer(
    data=data,
    backtest_func=run_cross_trade_with_windows,
    param_grid=param_grid,
    metric_to_optimize=metric_to_optimize,
    maximize_metric=maximize_metric,
    constant_params=constant_params,
    parallel=True,
    n_jobs=-1  # n_jobs=-1 uses all available cores
)

# --- Display Results ---
print("\n--- Optimization Results ---")

if results is None:
    print("No valid optimization results found. All combinations failed or returned invalid metrics.")
    print("Possible reasons:")
    print("  - Check if metric_to_optimize ('total_return_pct') is valid and available in the backtest results")
    print("  - Ensure parameter ranges are appropriate")
    print("  - Consider potential data issues (missing columns, etc.)")
    exit(0)

# Unpack results only if we have valid results
best_params, best_metric_value, all_results = results

print(f"Best Parameters for '{metric_to_optimize}': {best_params}")
print(f"Best Metric Value: {best_metric_value:.4f}")

print("\n--- Top 5 Parameter Combinations ---")
# Sort results for display
sorted_results = sorted(all_results, key=lambda x: x[1], reverse=maximize_metric)
for i, (params, metric_val) in enumerate(sorted_results[:5]):
    print(f"{i+1}. Params: {params}, Metric: {metric_val:.4f}")

# --- Optional: Run Backtest with Best Parameters ---
print("\n--- Running Backtest with Best Parameters ---")

# Make a copy of the data for the final backtest
final_data = data.copy()

# Compute the EMA indicators for the best parameters
short_window = best_params['short_window']
long_window = best_params['long_window']
final_data, _, _ = compute_indicator(final_data, indicator='ema', parameters={'window': short_window})
final_data, _, _ = compute_indicator(final_data, indicator='ema', parameters={'window': long_window})

# Run backtest with the best parameters using function-based API
results, portfolio_df = run_cross_trade(
    data=final_data,
    short_window_indicator=f"EMA_{short_window}",
    long_window_indicator=f"EMA_{long_window}",
    price_col='Close',
    config=None  # Uses default config
)

print("\n--- Performance Metrics (Best Parameters) ---")
for metric, value in results.items():
    if isinstance(value, (int, float)):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
# Optional: Display some of the trade history
print("\n--- Trade History (Sample) ---")
if 'Action' in portfolio_df.columns:
    # Filter to only show actual trades (not HOLDs)
    trades_df = portfolio_df[portfolio_df['Action'] != 'HOLD']
    if not trades_df.empty:
        print(trades_df.head(5))  # Show first 5 trades
    else:
        print("No trades executed.")

print(f"\nOptimization complete for {ticker}.")
