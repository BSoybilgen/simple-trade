"""Example usage of volatility indicator backtesting strategies."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data, premade_backtest, BandTradeBacktester

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
end_date = '2022-12-31'
interval = '1wk'

print(f"\nDownloading data for {symbol}...")
data_week = download_data(symbol, start_date, end_date, interval=interval)

# Initialize backtester for printing results
backtester = BandTradeBacktester()

# ### Backtest with ACB (Acceleration Bands)
print("\n" + "="*80)
print("BACKTEST WITH ACB (Acceleration Bands)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'acb'
specific_parameters = {
    'period': 20,
    'factor': 0.001,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ATP (Average True Range Percent)
print("\n" + "="*80)
print("BACKTEST WITH ATP (Average True Range Percent)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'atp'
specific_parameters = {
    'window': 14,
    'upper': 4.0,
    'lower': 2.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ATR (Average True Range)
print("\n" + "="*80)
print("BACKTEST WITH ATR (Average True Range)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'atr'
specific_parameters = {
    'window': 14,
    'upper_pct': 80,  # Upper percentile threshold
    'lower_pct': 20,  # Lower percentile threshold
    'lookback': 100,  # Lookback period for percentile calculation
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with BBW (Bollinger Band Width)
print("\n" + "="*80)
print("BACKTEST WITH BBW (Bollinger Band Width)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'bbw'
specific_parameters = {
    'window': 20,
    'num_std': 2.0,
    'upper': 10.0,
    'lower': 4.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with BOL (Bollinger Bands)
print("\n" + "="*80)
print("BACKTEST WITH BOL (Bollinger Bands)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'bol'
specific_parameters = {
    'window': 20,
    'num_std': 2,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with CHA (Chaikin Volatility)
print("\n" + "="*80)
print("BACKTEST WITH CHA (Chaikin Volatility)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'cha'
specific_parameters = {
    'ema_window': 10,
    'roc_window': 10,
    'upper': 20.0,
    'lower': -20.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with CHO (Choppiness Index)
print("\n" + "="*80)
print("BACKTEST WITH CHO (Choppiness Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'cho'
specific_parameters = {
    'period': 14,
    'upper': 61.8,
    'lower': 38.2,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with DON (Donchian Channels)
print("\n" + "="*80)
print("BACKTEST WITH DON (Donchian Channels)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'don'
specific_parameters = {
    'window': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with DVI (Dynamic Volatility Index)
print("\n" + "="*80)
print("BACKTEST WITH DVI (Dynamic Volatility Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'dvi'
specific_parameters = {
    'magnitude_period': 5,
    'stretch_period': 100,
    'smooth_period': 5,
    'upper': 70,
    'lower': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with EFR (Efficiency Ratio)
print("\n" + "="*80)
print("BACKTEST WITH EFR (Efficiency Ratio)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'efr'
specific_parameters = {
    'period': 10,
    'upper': 0.7,
    'lower': 0.3,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with FDI (Fractal Dimension Index)
print("\n" + "="*80)
print("BACKTEST WITH FDI (Fractal Dimension Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'fdi'
specific_parameters = {
    'period': 20,
    'upper': 1.6,
    'lower': 1.4,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with GRV (Garman-Klass Volatility)
print("\n" + "="*80)
print("BACKTEST WITH GRV (Garman-Klass Volatility)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'grv'
specific_parameters = {
    'period': 20,
    'upper': 30.0,
    'lower': 15.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with HAV (Heikin-Ashi Volatility)
print("\n" + "="*80)
print("BACKTEST WITH HAV (Heikin-Ashi Volatility)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'hav'
specific_parameters = {
    'period': 14,
    'method': 'atr',
    'upper_pct': 80,  # Upper percentile threshold
    'lower_pct': 20,  # Lower percentile threshold
    'lookback': 100,  # Lookback period for percentile calculation
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with HIV (Historical Volatility)
print("\n" + "="*80)
print("BACKTEST WITH HIV (Historical Volatility)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'hiv'
specific_parameters = {
    'period': 20,
    'upper': 40.0,
    'lower': 20.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with KEL (Keltner Channels)
print("\n" + "="*80)
print("BACKTEST WITH KEL (Keltner Channels)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'kel'
specific_parameters = {
    'ema_window': 20,
    'atr_window': 10,
    'atr_multiplier': 2.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with MAD (Median Absolute Deviation)
print("\n" + "="*80)
print("BACKTEST WITH MAD (Median Absolute Deviation)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'mad'
specific_parameters = {
    'period': 20,
    'upper': 3.0,
    'lower': 1.5,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with MAI (Mass Index)
print("\n" + "="*80)
print("BACKTEST WITH MAI (Mass Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'mai'
specific_parameters = {
    'ema_period': 9,
    'sum_period': 25,
    'upper': 26.0,
    'lower': 24.5,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with NAT (Normalized ATR)
print("\n" + "="*80)
print("BACKTEST WITH NAT (Normalized ATR)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'nat'
specific_parameters = {
    'window': 14,
    'upper': 3.5,
    'lower': 2.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PAV (Parkinson Volatility)
print("\n" + "="*80)
print("BACKTEST WITH PAV (Parkinson Volatility)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'pav'
specific_parameters = {
    'period': 20,
    'upper': 30.0,
    'lower': 20.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PCW (Price Channel Width)
print("\n" + "="*80)
print("BACKTEST WITH PCW (Price Channel Width)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'pcw'
specific_parameters = {
    'period': 20,
    'upper': 17.5,
    'lower': 10.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PRO (Projection Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH PRO (Projection Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'pro'
specific_parameters = {
    'period': 10,
    'smooth_period': 3,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with RSV (Rogers-Satchell Volatility)
print("\n" + "="*80)
print("BACKTEST WITH RSV (Rogers-Satchell Volatility)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'rsv'
specific_parameters = {
    'period': 20,
    'upper': 30.0,
    'lower': 20.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with RVI (Relative Volatility Index)
print("\n" + "="*80)
print("BACKTEST WITH RVI (Relative Volatility Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'rvi'
specific_parameters = {
    'window': 10,
    'rvi_period': 14,
    'upper': 70,
    'lower': 30,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with STD (Standard Deviation)
print("\n" + "="*80)
print("BACKTEST WITH STD (Standard Deviation)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'std'
specific_parameters = {
    'window': 20,
    'upper': 5.0,
    'lower': 2.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with SVI (Stochastic Volatility Index)
print("\n" + "="*80)
print("BACKTEST WITH SVI (Stochastic Volatility Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'svi'
specific_parameters = {
    'atr_period': 14,
    'stoch_period': 14,
    'smooth_k': 3,
    'smooth_d': 3,
    'upper': 80,
    'lower': 20,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with TSV (TSI Volatility)
print("\n" + "="*80)
print("BACKTEST WITH TSV (TSI Volatility)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'tsv'
specific_parameters = {
    'atr_period': 14,
    'long_period': 25,
    'short_period': 13,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ULI (Ulcer Index)
print("\n" + "="*80)
print("BACKTEST WITH ULI (Ulcer Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'uli'
specific_parameters = {
    'period': 14,
    'upper': 5.0,
    'lower': 1.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VHF (Vertical Horizontal Filter)
print("\n" + "="*80)
print("BACKTEST WITH VHF (Vertical Horizontal Filter)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vhf'
specific_parameters = {
    'period': 28,
    'upper': 0.40,
    'lower': 0.25,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VQI (Volatility Quality Index)
print("\n" + "="*80)
print("BACKTEST WITH VQI (Volatility Quality Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vqi'
specific_parameters = {
    'period': 9,
    'smooth_period': 9,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VRA (Volatility Ratio)
print("\n" + "="*80)
print("BACKTEST WITH VRA (Volatility Ratio)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vra'
specific_parameters = {
    'short_period': 5,
    'long_period': 20,
    'upper': 1.25,
    'lower': 0.8,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VSI (Volatility Switch Index)
print("\n" + "="*80)
print("BACKTEST WITH VSI (Volatility Switch Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vsi'
specific_parameters = {
    'short_period': 10,
    'long_period': 50,
    'threshold': 1.2,
    'upper': 0.5,
    'lower': 0.5,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

print("\n" + "="*80)
print("ALL BACKTESTS COMPLETED")
print("="*80)
