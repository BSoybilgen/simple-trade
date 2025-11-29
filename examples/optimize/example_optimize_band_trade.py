#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for optimizing BandTrade strategy parameters.
"""

from simple_trade import download_data, compute_indicator
from simple_trade import run_band_trade, custom_optimizer

# --- Load Data ---
ticker = 'SPY'
start_date = '2020-01-01'
end_date = '2024-12-31'

# --- Download Data ---
print(f"Downloading data for {ticker}...")
data = download_data(ticker, start_date, end_date)

# --- Optimization Parameters ---
# Define the parameter grid to search
param_grid = {
    'rsi_window': [12, 14, 16],
    'rsi_lower_threshold': [20, 30],
    'rsi_upper_threshold': [70, 80],
}

# Define constant parameters for the backtester
initial_capital = 100000
commission_fee = 0.001  # 0.1%
constant_params = {
    'initial_cash': initial_capital, 
    'commission_long': commission_fee,
    'commission_short': commission_fee,
    'price_col': 'Close'
}

# Define the metric to optimize and whether to maximize or minimize
metric_to_optimize = 'total_return_pct'
maximize_metric = True

# --- Define Wrapper Function ---
def run_band_trade_with_windows(data, rsi_upper_threshold, rsi_lower_threshold, rsi_window, **kwargs):
    """
    Wrapper function to compute indicators and run the band trade backtest.
    """
    # Work on a copy of the data
    df = data.copy()
    
    # Set up threshold columns
    upper_threshold_col = 'RSI_Upper'
    lower_threshold_col = 'RSI_Lower'
    df[upper_threshold_col] = rsi_upper_threshold
    df[lower_threshold_col] = rsi_lower_threshold

    # Compute RSI indicator
    df, _, _ = compute_indicator(df, indicator='rsi', parameters={'window': rsi_window}, columns={'close_col': 'Close'})
    indicator_col = f'RSI_{rsi_window}'
    
    # Run the backtest using the function-based API
    return run_band_trade(
        data=df,
        indicator_col=indicator_col,
        upper_band_col=upper_threshold_col,
        lower_band_col=lower_threshold_col,
        **kwargs
    )

# --- Start Optimizer ---
print("Initializing Optimizer...")
print("\nRunning Optimization (Parallel)...")
# Run optimization with parallel processing
results = custom_optimizer(
    data=data,
    backtest_func=run_band_trade_with_windows,
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
    print("  - Check for any errors in the BandTrade implementation or wrapper function")
    print("  - Consider potential data issues (missing columns, etc.)")
else:
    # Unpack results
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

    # Extract best parameters
    rsi_window = best_params['rsi_window']
    rsi_lower_threshold = best_params['rsi_lower_threshold']
    rsi_upper_threshold = best_params['rsi_upper_threshold']

    # Prepare data for final backtest
    final_data = data.copy()
    final_data['RSI_Upper'] = rsi_upper_threshold
    final_data['RSI_Lower'] = rsi_lower_threshold
    final_data, _, _ = compute_indicator(final_data, indicator='rsi', parameters={'window': rsi_window}, columns={'close_col': 'Close'})
    
    # Run backtest with best parameters using function-based API
    results, portfolio_df = run_band_trade(
        data=final_data,
        indicator_col=f'RSI_{rsi_window}',
        upper_band_col='RSI_Upper',
        lower_band_col='RSI_Lower',
        price_col='Close',
        config=None  # Uses default config
    )
    
    print("\n--- Performance Metrics (Best Parameters) ---")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
            
    # Display trade history sample
    print("\n--- Trade History (Sample) ---")
    if 'Action' in portfolio_df.columns:
        # Filter to only show actual trades (not HOLDs)
        trades_df = portfolio_df[portfolio_df['Action'] != 'HOLD']
        if not trades_df.empty:
            print(trades_df.head(5))
        else:
            print("No trades executed.")

print(f"\nOptimization complete for {ticker}.") 