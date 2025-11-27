"""Example usage of pre-made momentum indicator backtesting strategies."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data, premade_backtest, CrossTradeBacktester

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

# Initialize backtester for printing results
backtester = CrossTradeBacktester()

# ### Backtest with AWO (Awesome Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH AWO (Awesome Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'awo'
specific_parameters = {
    'fast_window': 5,
    'slow_window': 34,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with BOP (Balance of Power)
print("\n" + "="*80)
print("BACKTEST WITH BOP (Balance of Power)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'bop'
specific_parameters = {
    'window': 14,
    'smooth': True,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with CCI (Commodity Channel Index)
print("\n" + "="*80)
print("BACKTEST WITH CCI (Commodity Channel Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'cci'
specific_parameters = {
    'window': 20,
    'constant': 0.015,
    'upper': 150,
    'lower': -150,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with CMO (Chande Momentum Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH CMO (Chande Momentum Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'cmo'
specific_parameters = {
    'window': 14,
    'upper': 50,
    'lower': -50,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with COG (Center of Gravity)
print("\n" + "="*80)
print("BACKTEST WITH COG (Center of Gravity)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'cog'
specific_parameters = {
    'window': 20,
    'signal_window': 9,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with CRS (Connors RSI)
print("\n" + "="*80)
print("BACKTEST WITH CRS (Connors RSI)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'crs'
specific_parameters = {
    'rsi_window': 3,
    'streak_window': 2,
    'rank_window': 100,
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

# ### Backtest with DPO (Detrended Price Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH DPO (Detrended Price Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'dpo'
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

# ### Backtest with ERI (Elder-Ray Index)
print("\n" + "="*80)
print("BACKTEST WITH ERI (Elder-Ray Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'eri'
specific_parameters = {
    'window': 25,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with FIS (Fisher Transform)
print("\n" + "="*80)
print("BACKTEST WITH FIS (Fisher Transform)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'fis'
specific_parameters = {
    'window': 25,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with IMI (Intraday Momentum Index)
print("\n" + "="*80)
print("BACKTEST WITH IMI (Intraday Momentum Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'imi'
specific_parameters = {
    'window': 14,
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

# ### Backtest with KST (Know Sure Thing)
print("\n" + "="*80)
print("BACKTEST WITH KST (Know Sure Thing)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'kst'
specific_parameters = {
    'signal': 9,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with LSI (Laguerre RSI)
print("\n" + "="*80)
print("BACKTEST WITH LSI (Laguerre RSI)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'lsi'
specific_parameters = {
    'gamma': 0.5,
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

# ### Backtest with MACD (Moving Average Convergence Divergence)
print("\n" + "="*80)
print("BACKTEST WITH MACD (Moving Average Convergence Divergence)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'mac'
specific_parameters = {
    'window_fast': 12,
    'window_slow': 26,
    'window_signal': 9,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with MSI (Momentum Strength Index)
print("\n" + "="*80)
print("BACKTEST WITH MSI (Momentum Strength Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'msi'
specific_parameters = {
    'window': 14,
    'power': 1.0,
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

# ### Backtest with PGO (Pretty Good Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH PGO (Pretty Good Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'pgo'
specific_parameters = {
    'window': 14,
    'upper': 3.0,
    'lower': -2.0,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PPO (Percentage Price Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH PPO (Percentage Price Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ppo'
specific_parameters = {
    'fast_window': 12,
    'slow_window': 26,
    'signal_window': 9,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with PSY (Psychological Line)
print("\n" + "="*80)
print("BACKTEST WITH PSY (Psychological Line)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'psy'
specific_parameters = {
    'window': 12,
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

# ### Backtest with QST (Qstick)
print("\n" + "="*80)
print("BACKTEST WITH QST (Qstick)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'qst'
specific_parameters = {
    'window': 10,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with RMI (Relative Momentum Index)
print("\n" + "="*80)
print("BACKTEST WITH RMI (Relative Momentum Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'rmi'
specific_parameters = {
    'window': 20,
    'momentum_period': 5,
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

# ### Backtest with ROC (Rate of Change)
print("\n" + "="*80)
print("BACKTEST WITH ROC (Rate of Change)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'roc'
specific_parameters = {
    'window': 12,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with RSI (Relative Strength Index)
print("\n" + "="*80)
print("BACKTEST WITH RSI (Relative Strength Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'rsi'
specific_parameters = {
    'window': 14,
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

# ### Backtest with RVG (Relative Vigor Index)
print("\n" + "="*80)
print("BACKTEST WITH RVG (Relative Vigor Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'rvg'
specific_parameters = {
    'window': 10,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with SRI (Stochastic RSI)
print("\n" + "="*80)
print("BACKTEST WITH SRI (Stochastic RSI)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'sri'
specific_parameters = {
    'rsi_window': 14,
    'stoch_window': 14,
    'k_window': 3,
    'd_window': 3,
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

# ### Backtest with STC (Schaff Trend Cycle)
print("\n" + "="*80)
print("BACKTEST WITH STC (Schaff Trend Cycle)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'stc'
specific_parameters = {
    'window_fast': 23,
    'window_slow': 50,
    'cycle': 10,
    'smooth': 3,
    'upper': 75,
    'lower': 25,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with STOCH (Stochastic Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH STOCH (Stochastic Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'sto'
specific_parameters = {
    'k_period': 14,
    'd_period': 3,
    'smooth_k': 3,
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

# ### Backtest with TSI (True Strength Index)
print("\n" + "="*80)
print("BACKTEST WITH TSI (True Strength Index)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'tsi'
specific_parameters = {
    'slow': 25,
    'fast': 13,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with TTM (TTM Squeeze)
print("\n" + "="*80)
print("BACKTEST WITH TTM (TTM Squeeze)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ttm'
specific_parameters = {
    'length': 20,
    'std_dev': 2.0,
    'atr_length': 20,
    'atr_multiplier': 1.5,
    'smooth': 3,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with ULT (Ultimate Oscillator)
print("\n" + "="*80)
print("BACKTEST WITH ULT (Ultimate Oscillator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'ult'
specific_parameters = {
    'short_window': 7,
    'medium_window': 14,
    'long_window': 28,
    'upper': 65,
    'lower': 35,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with VOR (Vortex Indicator)
print("\n" + "="*80)
print("BACKTEST WITH VOR (Vortex Indicator)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'vor'
specific_parameters = {
    'window': 14,
    'fig_control': 1,
}

parameters = {**global_parameters, **specific_parameters}

results, portfolio, fig = premade_backtest(data_day, strategy_name, parameters)
backtester.print_results(results)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Backtest with WIL (Williams %R)
print("\n" + "="*80)
print("BACKTEST WITH WIL (Williams %R)")
print("="*80)

# --- Strategy Parameters ---
strategy_name = 'wil'
specific_parameters = {
    'window': 14,
    'upper': -20,
    'lower': -80,
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
