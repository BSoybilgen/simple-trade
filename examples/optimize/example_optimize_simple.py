"""
Example script demonstrating the premade_optimizer for various trading strategies.
This example shows how to optimize different strategies using the simple_trade library.
"""

from simple_trade import download_data
from simple_trade import premade_optimizer
import pandas as pd

# --- Configuration ---
ticker = 'SPY'
start_date = '2020-01-01'
end_date = '2024-12-31'

print(f"=== Simple Trade Optimizer Example ===")
print(f"Ticker: {ticker} | Period: {start_date} to {end_date}")

# --- Download Data ---
print(f"\nDownloading data for {ticker}...")
data = download_data(ticker, start_date, end_date)
print(f"Data shape: {data.shape}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# --- Example 1: RSI Strategy Optimization ---
print("\n" + "="*60)
print("EXAMPLE 1: RSI Strategy Optimization")
print("="*60)

# Define parameter grid for RSI strategy
rsi_param_grid = {
    'window': [10, 14, 20],           # RSI window periods
    'upper': [70, 75, 80],            # Upper threshold values
    'lower': [20, 25, 30]             # Lower threshold values
}

# Base parameters for RSI strategy
rsi_base_params = {
    'initial_cash': 100000.0,
    'commission_long': 0.001,         # 0.1% commission
    'commission_short': 0.001,
    'trading_type': 'long',           # Only long trades
    'day1_position': 'none',
    'risk_free_rate': 0.02,
    'metric': 'total_return_pct',     # Metric to optimize
    'maximize': True,                 # Maximize the metric
    'parallel': False,                # Sequential execution for this example
    'fig_control': 0                  # No plotting during optimization
}

print("RSI Strategy Parameters:")
print(f"  Parameter grid: {rsi_param_grid}")
print(f"  Base parameters: {rsi_base_params}")
print(f"  Total combinations: {len(rsi_param_grid['window']) * len(rsi_param_grid['upper']) * len(rsi_param_grid['lower'])}")

# Run RSI optimization
print("\nRunning RSI optimization...")
rsi_best_results, rsi_best_params, rsi_all_results = premade_optimizer(
    data=data,
    strategy_name='rsi',
    parameters=rsi_base_params,
    param_grid=rsi_param_grid
)

if rsi_best_results is not None:
    print(f"\nRSI Optimization Results:")
    print(f"Best Parameters: {rsi_best_params}")
    print(f"Best Total Return: {rsi_best_results['total_return_pct']:.2f}%")
    print(f"Best Sharpe Ratio: {rsi_best_results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {rsi_best_results['max_drawdown_pct']:.2f}%")
    
    # Show top 3 parameter combinations
    print(f"\nTop 3 RSI Parameter Combinations:")
    sorted_rsi = sorted(rsi_all_results, key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_rsi[:3]):
        print(f"  {i+1}. {result['params']} -> {result['score']:.2f}%")

# --- Example 2: SMA Cross Strategy Optimization ---
print("\n" + "="*60)
print("EXAMPLE 2: SMA Cross Strategy Optimization")
print("="*60)

# Define parameter grid for SMA cross strategy
sma_param_grid = {
    'short_window': [10, 20, 30],     # Short SMA periods
    'long_window': [50, 100, 150]     # Long SMA periods
}

# Base parameters for SMA strategy
sma_base_params = {
    'initial_cash': 100000.0,
    'commission_long': 0.001,
    'commission_short': 0.001,
    'trading_type': 'long',
    'day1_position': 'none',
    'risk_free_rate': 0.02,
    'metric': 'sharpe_ratio',         # Optimize for Sharpe ratio
    'maximize': True,
    'parallel': True,                 # Parallel execution for faster results
    'n_jobs': -1,                     # Use all available cores
    'fig_control': 0
}

print("SMA Cross Strategy Parameters:")
print(f"  Parameter grid: {sma_param_grid}")
print(f"  Optimizing metric: {sma_base_params['metric']}")
print(f"  Parallel processing: {sma_base_params['parallel']}")

# Run SMA optimization
print("\nRunning SMA cross optimization...")
sma_best_results, sma_best_params, sma_all_results = premade_optimizer(
    data=data,
    strategy_name='sma',
    parameters=sma_base_params,
    param_grid=sma_param_grid
)

if sma_best_results is not None:
    print(f"\nSMA Cross Optimization Results:")
    print(f"Best Parameters: {sma_best_params}")
    print(f"Best Sharpe Ratio: {sma_best_results['sharpe_ratio']:.4f}")
    print(f"Total Return: {sma_best_results['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {sma_best_results['max_drawdown_pct']:.2f}%")

# --- Example 3: MACD Strategy Optimization ---
print("\n" + "="*60)
print("EXAMPLE 3: MACD Strategy Optimization")
print("="*60)

# Define parameter grid for MACD strategy
macd_param_grid = {
    'window_fast': [8, 12, 16],       # Fast EMA periods
    'window_slow': [21, 26, 31],      # Slow EMA periods
    'window_signal': [7, 9, 11]       # Signal line periods
}

# Base parameters for MACD strategy
macd_base_params = {
    'initial_cash': 100000.0,
    'commission_long': 0.001,
    'commission_short': 0.001,
    'trading_type': 'long',
    'day1_position': 'none',
    'risk_free_rate': 0.02,
    'metric': 'total_return_pct',
    'maximize': True,
    'parallel': False,
    'fig_control': 0
}

print("MACD Strategy Parameters:")
print(f"  Parameter grid: {macd_param_grid}")
print(f"  Total combinations: {len(macd_param_grid['window_fast']) * len(macd_param_grid['window_slow']) * len(macd_param_grid['window_signal'])}")

# Run MACD optimization
print("\nRunning MACD optimization...")
macd_best_results, macd_best_params, macd_all_results = premade_optimizer(
    data=data,
    strategy_name='mac',
    parameters=macd_base_params,
    param_grid=macd_param_grid
)

if macd_best_results is not None:
    print(f"\nMACD Optimization Results:")
    print(f"Best Parameters: {macd_best_params}")
    print(f"Best Total Return: {macd_best_results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {macd_best_results['sharpe_ratio']:.4f}")

# --- Strategy Comparison ---
print("\n" + "="*60)
print("STRATEGY COMPARISON")
print("="*60)

strategies_results = []

if rsi_best_results is not None:
    strategies_results.append({
        'Strategy': 'RSI',
        'Parameters': rsi_best_params,
        'Total Return (%)': rsi_best_results['total_return_pct'],
        'Sharpe Ratio': rsi_best_results['sharpe_ratio'],
        'Max Drawdown (%)': rsi_best_results['max_drawdown_pct']
    })

if sma_best_results is not None:
    strategies_results.append({
        'Strategy': 'SMA Cross',
        'Parameters': sma_best_params,
        'Total Return (%)': sma_best_results['total_return_pct'],
        'Sharpe Ratio': sma_best_results['sharpe_ratio'],
        'Max Drawdown (%)': sma_best_results['max_drawdown_pct']
    })

if macd_best_results is not None:
    strategies_results.append({
        'Strategy': 'MACD',
        'Parameters': macd_best_params,
        'Total Return (%)': macd_best_results['total_return_pct'],
        'Sharpe Ratio': macd_best_results['sharpe_ratio'],
        'Max Drawdown (%)': macd_best_results['max_drawdown_pct']
    })

if strategies_results:
    comparison_df = pd.DataFrame(strategies_results)
    print("\nStrategy Performance Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Find best performing strategy by total return
    best_strategy = max(strategies_results, key=lambda x: x['Total Return (%)'])
    print(f"\nBest Performing Strategy: {best_strategy['Strategy']}")
    print(f"Parameters: {best_strategy['Parameters']}")
    print(f"Total Return: {best_strategy['Total Return (%)']:.2f}%")