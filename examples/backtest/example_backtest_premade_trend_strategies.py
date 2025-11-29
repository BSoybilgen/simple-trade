"""Example usage of trend indicator backtesting strategies."""

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

# ### Backtest with ADS
print("\n" + "="*80)
print("BACKTEST WITH ADS")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ads'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ADX
print("\n" + "="*80)
print("BACKTEST WITH ADX")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'adx'
specific_parameters = {
    'window': 14,
    'adx_threshold': 25,
    'ma_window': 40,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_week, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ALM
print("\n" + "="*80)
print("BACKTEST WITH ALM")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'alm'
specific_parameters = {
    'short_window': 25,
    'long_window': 150,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with AMA
print("\n" + "="*80)
print("BACKTEST WITH AMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ama'
specific_parameters = {
    'short_window': 10,
    'long_window': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with AROON
print("\n" + "="*80)
print("BACKTEST WITH AROON")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'aro'
specific_parameters = {
    'period': 50,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with DEM
print("\n" + "="*80)
print("BACKTEST WITH DEM")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'dem'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with EAC
print("\n" + "="*80)
print("BACKTEST WITH EAC")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'eac'
specific_parameters = {
    'short_alpha': 0.03,
    'long_alpha': 0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_week, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with EIT
print("\n" + "="*80)
print("BACKTEST WITH EIT")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'eit'
specific_parameters = {
    'short_alpha': 0.07,
    'long_alpha': 0.14,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with EMA
print("\n" + "="*80)
print("BACKTEST WITH EMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ema'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with FMA
print("\n" + "="*80)
print("BACKTEST WITH FMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'fma'
specific_parameters = {
    'short_window': 0,
    'long_window': 100,
    'use_adx_filter': True,      # Enable ADX filter
    'adx_window': 14,             # ADX calculation window
    'adx_threshold': 25,          # Only trade when ADX > 25
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_week, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with GMA
print("\n" + "="*80)
print("BACKTEST WITH GMA (Guppy Multiple Moving Average)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'gma'
specific_parameters = {
    'short_windows': (25, 30),  # Short-term EMAs
    'long_windows': (75, 80),  # Long-term EMAs
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with HMA
print("\n" + "="*80)
print("BACKTEST WITH HMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'hma'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with HTT
print("\n" + "="*80)
print("BACKTEST WITH HTT (Hilbert Transform Trendline)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'htt'
specific_parameters = {
    'short_window': 25,         # Fast HTT
    'long_window': 75,         # Slow HTT
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ICHIMOKU
print("\n" + "="*80)
print("BACKTEST WITH ICHIMOKU")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ich'
specific_parameters = {
    'tenkan_period': 9,
    'kijun_period': 26,
    'senkou_b_period': 52,
    'displacement': 26,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with JMA
print("\n" + "="*80)
print("BACKTEST WITH JMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'jma'
specific_parameters = {
    'short_length': 6,
    'long_length': 12,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with KMA
print("\n" + "="*80)
print("BACKTEST WITH KMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'kma'
specific_parameters = {
    'short_window': 10,
    'long_window': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with LSM
print("\n" + "="*80)
print("BACKTEST WITH LSM")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'lsm'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with MGD
print("\n" + "="*80)
print("BACKTEST WITH MGD")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'mgd'
specific_parameters = {
    'short_window': 10,
    'long_window': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PSAR
print("\n" + "="*80)
print("BACKTEST WITH PSAR")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'psa'
specific_parameters = {
    'af_initial': 0.003,
    'af_step': 0.003,
    'af_max': 0.03,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with SMA
print("\n" + "="*80)
print("BACKTEST WITH SMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'sma'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with SOA
print("\n" + "="*80)
print("BACKTEST WITH SOA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'soa'
specific_parameters = {
    'short_window': 10,
    'long_window': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with SWM
print("\n" + "="*80)
print("BACKTEST WITH SWM")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'swm'
specific_parameters = {
    'short_window': 10,
    'long_window': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with SUPERTREND
print("\n" + "="*80)
print("BACKTEST WITH SUPERTREND")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'str'
specific_parameters = {
    'period': 7,
    'multiplier': 3.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with TMA
print("\n" + "="*80)
print("BACKTEST WITH TMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'tma'
specific_parameters = {
    'short_window': 10,
    'long_window': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with TEM
print("\n" + "="*80)
print("BACKTEST WITH TEM")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'tem'
specific_parameters = {
    'short_window': 50,
    'long_window': 100,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with TRIX
print("\n" + "="*80)
print("BACKTEST WITH TRIX")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'tri'
specific_parameters = {
    'window': 40,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VID
print("\n" + "="*80)
print("BACKTEST WITH VID")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vid'
specific_parameters = {
    'short_window': 14,
    'long_window': 42,
    'cmo_window': 9,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with WMA
print("\n" + "="*80)
print("BACKTEST WITH WMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'wma'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = run_premade_trade(data_day, strategy_name, parameters)
print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ZMA
print("\n" + "="*80)
print("BACKTEST WITH ZMA")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'zma'
specific_parameters = {
    'short_window': 25,
    'long_window': 75,
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
