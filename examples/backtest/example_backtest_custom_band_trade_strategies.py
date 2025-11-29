"""Example usage of run_band_trade with manual indicator computation."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import (
    download_data,
    compute_indicator,
    run_band_trade,
    print_results,
    plot_backtest_results,
    BacktestConfig,
)

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# --- Global Parameters ---
initial_cash = 10000.0
commission = 0.0
long_entry_pct_cash = 1.0
short_entry_pct_cash = 0.5

# --- Backtest Configuration ---
symbol = 'SPY'
start_date = '2020-01-01'
end_date = '2024-12-31'
interval = '1d'

# RSI Parameters
rsi_window = 14
rsi_upper_threshold = 80
rsi_lower_threshold = 20

print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start_date, end_date, interval=interval)

# --- Compute RSI Indicator ---
parameters = {'window': rsi_window}
columns = {'close_col': 'Close'}
data, _, _ = compute_indicator(data, indicator='rsi', parameters=parameters, columns=columns)

# Add constant columns for the fixed RSI thresholds
indicator_col = f'RSI_{rsi_window}'
upper_threshold_col = 'RSI_Upper'
lower_threshold_col = 'RSI_Lower'
data[upper_threshold_col] = rsi_upper_threshold
data[lower_threshold_col] = rsi_lower_threshold

indicator_cols_to_plot = [indicator_col, upper_threshold_col, lower_threshold_col]

# ### Long Only Trading Backtest
print("\n" + "="*80)
print("BAND TRADE - LONG ONLY (RSI Mean Reversion)")
print("="*80)

config = BacktestConfig(
    initial_cash=initial_cash,
    commission_long=commission,
    commission_short=commission,
    long_entry_pct_cash=long_entry_pct_cash,
    short_entry_pct_cash=short_entry_pct_cash,
)

results, portfolio = run_band_trade(
    data=data,
    indicator_col=indicator_col,
    upper_band_col=upper_threshold_col,
    lower_band_col=lower_threshold_col,
    config=config,
    price_col='Close',
    trading_type='long',
)

print_results(results)

fig = plot_backtest_results(
    data_df=data,
    history_df=portfolio,
    price_col='Close',
    indicator_cols=indicator_cols_to_plot,
    title=f"RSI Threshold (Long Only) (RSI-{rsi_window} {rsi_lower_threshold}/{rsi_upper_threshold})"
)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Short Only Trading Backtest
print("\n" + "="*80)
print("BAND TRADE - SHORT ONLY (RSI Mean Reversion)")
print("="*80)

config = BacktestConfig(
    initial_cash=initial_cash,
    commission_long=commission,
    commission_short=commission,
    long_entry_pct_cash=long_entry_pct_cash,
    short_entry_pct_cash=short_entry_pct_cash,
)

results, portfolio = run_band_trade(
    data=data,
    indicator_col=indicator_col,
    upper_band_col=upper_threshold_col,
    lower_band_col=lower_threshold_col,
    config=config,
    price_col='Close',
    trading_type='short',
)

print_results(results)

fig = plot_backtest_results(
    data_df=data,
    history_df=portfolio,
    price_col='Close',
    indicator_cols=indicator_cols_to_plot,
    title=f"RSI Threshold (Short Only) (RSI-{rsi_window} {rsi_lower_threshold}/{rsi_upper_threshold})"
)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Mixed Trading Backtest
print("\n" + "="*80)
print("BAND TRADE - MIXED (RSI Mean Reversion)")
print("="*80)

config = BacktestConfig(
    initial_cash=initial_cash,
    commission_long=commission,
    commission_short=commission,
    long_entry_pct_cash=long_entry_pct_cash,
    short_entry_pct_cash=short_entry_pct_cash,
)

results, portfolio = run_band_trade(
    data=data,
    indicator_col=indicator_col,
    upper_band_col=upper_threshold_col,
    lower_band_col=lower_threshold_col,
    config=config,
    price_col='Close',
    trading_type='mixed',
)

print_results(results)

fig = plot_backtest_results(
    data_df=data,
    history_df=portfolio,
    price_col='Close',
    indicator_cols=indicator_cols_to_plot,
    title=f"RSI Threshold (Mixed Trading) (RSI-{rsi_window} {rsi_lower_threshold}/{rsi_upper_threshold})"
)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

print("\n" + "="*80)
print("ALL BAND TRADE BACKTESTS COMPLETED")
print("="*80)
