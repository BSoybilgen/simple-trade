"""Example usage of volume indicator backtesting strategies."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data, run_premade_trade, print_results

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# --- Global Parameters ---
global_parameters = {
    'initial_cash': 10000,
    'commission_long': 0.001,
    'commission_short': 0.001,
    'short_borrow_fee_inc_rate': 0.0,
    'long_borrow_fee_inc_rate': 0.0,
    'trading_type': 'long',
    'day1_position': 'none',
    'risk_free_rate': 0.0,
}

# --- Backtest Configuration ---
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-12-31'
interval = '1d'

print(f"\nDownloading data for {symbol}...")
data_day = download_data(symbol, start_date, end_date, interval=interval)

# --- Backtest Configuration ---
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-12-31'
interval = '1wk'

print(f"\nDownloading data for {symbol}...")
data_week = download_data(symbol, start_date, end_date, interval=interval)

# ### Backtest with ADL
print("\n" + "="*80)
print("BACKTEST WITH ADL (Accumulation/Distribution Line)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'adl'
specific_parameters = {
    'sma_period': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ADO
print("\n" + "="*80)
print("BACKTEST WITH ADO (Accumulation/Distribution Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ado'
specific_parameters = {
    'period': 14,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with BWM
print("\n" + "="*80)
print("BACKTEST WITH BWM (Bill Williams Market Facilitation Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'bwm'
specific_parameters = {
    'upper_pct': 80,
    'lower_pct': 20,
    'lookback': 100,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with CMF
print("\n" + "="*80)
print("BACKTEST WITH CMF (Chaikin Money Flow)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'cmf'
specific_parameters = {
    'period': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with EMV
print("\n" + "="*80)
print("BACKTEST WITH EMV (Ease of Movement)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'emv'
specific_parameters = {
    'period': 14,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with FOI
print("\n" + "="*80)
print("BACKTEST WITH FOI (Force Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'foi'
specific_parameters = {
    'period': 13,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with FVE
print("\n" + "="*80)
print("BACKTEST WITH FVE (Finite Volume Elements)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'fve'
specific_parameters = {
    'period': 22,
    'factor': 0.3,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with KVO
print("\n" + "="*80)
print("BACKTEST WITH KVO (Klinger Volume Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'kvo'
specific_parameters = {
    'fast_period': 34,
    'slow_period': 55,
    'signal_period': 13,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with MFI
print("\n" + "="*80)
print("BACKTEST WITH MFI (Money Flow Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'mfi'
specific_parameters = {
    'period': 14,
    'upper': 80,
    'lower': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with NVI
print("\n" + "="*80)
print("BACKTEST WITH NVI (Negative Volume Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'nvi'
specific_parameters = {
    'sma_period': 25,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_week, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with OBV
print("\n" + "="*80)
print("BACKTEST WITH OBV (On-Balance Volume)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'obv'
specific_parameters = {
    'sma_period': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PVI
print("\n" + "="*80)
print("BACKTEST WITH PVI (Positive Volume Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'pvi'
specific_parameters = {
    'sma_period': 25,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_week, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PVO
print("\n" + "="*80)
print("BACKTEST WITH PVO (Percentage Volume Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'pvo'
specific_parameters = {
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VFI
print("\n" + "="*80)
print("BACKTEST WITH VFI (Volume Flow Indicator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vfi'
specific_parameters = {
    'period': 25,
    'coef': 0.2,
    'vcoef': 2.5,
    'smoothing_period': 3,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_week, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VMA
print("\n" + "="*80)
print("BACKTEST WITH VMA (Volume Moving Average)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vma'
specific_parameters = {
    'window': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VOO
print("\n" + "="*80)
print("BACKTEST WITH VOO (Volume Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'voo'
specific_parameters = {
    'fast_period': 5,
    'slow_period': 10,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VPT
print("\n" + "="*80)
print("BACKTEST WITH VPT (Volume Price Trend)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vpt'
specific_parameters = {
    'sma_period': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VRO
print("\n" + "="*80)
print("BACKTEST WITH VRO (Volume Rate of Change)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vro'
specific_parameters = {
    'period': 14,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VWA
print("\n" + "="*80)
print("BACKTEST WITH VWA (Volume Weighted Average Price)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vwa'
specific_parameters = {
    'window': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with WAD
print("\n" + "="*80)
print("BACKTEST WITH WAD (Williams Accumulation/Distribution)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'wad'
specific_parameters = {
    'sma_period': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

print("\n" + "="*80)
print("ALL BACKTESTS COMPLETED")
print("="*80)
