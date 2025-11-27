"""Example usage of CrossTradeBacktester with manual indicator computation."""

import matplotlib.pyplot as plt
import pandas as pd

from simple_trade import download_data, compute_indicator
from simple_trade import CrossTradeBacktester
from simple_trade import BacktestPlotter

# Set pandas display options for better output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# --- Global Parameters ---
initial_cash = 10000.0
commission = 0.01
short_borrow_fee_inc_rate = 0.00001

# --- Backtest Configuration ---
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-12-31'
interval = '1d'

print(f"\nDownloading data for {symbol}...")
data = download_data(symbol, start_date, end_date, interval=interval)

# --- Compute Indicators ---
short_window = 25
long_window = 75

parameters = {'window': short_window}
columns = {'close': 'Close'}
data, _, _ = compute_indicator(data, indicator='sma', parameters=parameters, columns=columns)

parameters = {'window': long_window}
columns = {'close': 'Close'}
data, _, _ = compute_indicator(data, indicator='sma', parameters=parameters, columns=columns)

# Initialize plotter
plotter = BacktestPlotter()

# ### Long Only Trading Backtest
print("\n" + "="*80)
print("CROSS TRADE - LONG ONLY (SMA Cross)")
print("="*80)

backtester = CrossTradeBacktester(
    initial_cash=initial_cash,
    commission_long=commission,
    commission_short=commission
)

results, portfolio = backtester.run_cross_trade(
    data=data,
    short_window_indicator=f"SMA_{short_window}",
    long_window_indicator=f"SMA_{long_window}",
    price_col='Close',
    trading_type='long'
)

backtester.print_results(results)

indicator_cols_to_plot = [f'SMA_{short_window}', f'SMA_{long_window}']
fig = plotter.plot_results(
    data_df=data,
    history_df=portfolio,
    price_col='Close',
    indicator_cols=indicator_cols_to_plot,
    title=f"Cross Trade (Long Only) (SMA-{short_window} vs SMA-{long_window})"
)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Short Only Trading Backtest
print("\n" + "="*80)
print("CROSS TRADE - SHORT ONLY (SMA Cross)")
print("="*80)

backtester = CrossTradeBacktester(
    initial_cash=initial_cash,
    commission_long=commission,
    commission_short=commission,
    short_borrow_fee_inc_rate=short_borrow_fee_inc_rate
)

results, portfolio = backtester.run_cross_trade(
    data=data,
    short_window_indicator=f"SMA_{short_window}",
    long_window_indicator=f"SMA_{long_window}",
    price_col='Close',
    trading_type='short'
)

backtester.print_results(results)

fig = plotter.plot_results(
    data_df=data,
    history_df=portfolio,
    price_col='Close',
    indicator_cols=indicator_cols_to_plot,
    title=f"Cross Trade (Short Only) (SMA-{short_window} vs SMA-{long_window})"
)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Mixed Trading Backtest (Scenario 1: 25/75 SMA)
print("\n" + "="*80)
print("CROSS TRADE - MIXED (SMA 25/75)")
print("="*80)

backtester = CrossTradeBacktester(
    initial_cash=initial_cash,
    commission_long=commission,
    commission_short=commission,
    short_borrow_fee_inc_rate=short_borrow_fee_inc_rate
)

results, portfolio = backtester.run_cross_trade(
    data=data,
    short_window_indicator=f"SMA_{short_window}",
    long_window_indicator=f"SMA_{long_window}",
    price_col='Close',
    trading_type='mixed'
)

backtester.print_results(results)

fig = plotter.plot_results(
    data_df=data,
    history_df=portfolio,
    price_col='Close',
    indicator_cols=indicator_cols_to_plot,
    title=f"Cross Trade (Mixed Trading) (SMA-{short_window} vs SMA-{long_window})"
)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

# ### Mixed Trading Backtest (Scenario 2: 25/150 SMA)
print("\n" + "="*80)
print("CROSS TRADE - MIXED (SMA 25/150)")
print("="*80)

# Reload data and compute new indicators
data2 = download_data(symbol, start_date, end_date, interval=interval)

short_window_2 = 25
long_window_2 = 150

parameters = {'window': short_window_2}
columns = {'close': 'Close'}
data2, _, _ = compute_indicator(data2, indicator='sma', parameters=parameters, columns=columns)

parameters = {'window': long_window_2}
columns = {'close': 'Close'}
data2, _, _ = compute_indicator(data2, indicator='sma', parameters=parameters, columns=columns)

backtester = CrossTradeBacktester(
    initial_cash=initial_cash,
    commission_long=commission,
    commission_short=commission,
    short_borrow_fee_inc_rate=short_borrow_fee_inc_rate
)

results, portfolio = backtester.run_cross_trade(
    data=data2,
    short_window_indicator=f"SMA_{short_window_2}",
    long_window_indicator=f"SMA_{long_window_2}",
    price_col='Close',
    trading_type='mixed'
)

backtester.print_results(results)

indicator_cols_to_plot_2 = [f'SMA_{short_window_2}', f'SMA_{long_window_2}']
fig = plotter.plot_results(
    data_df=data2,
    history_df=portfolio,
    price_col='Close',
    indicator_cols=indicator_cols_to_plot_2,
    title=f"Cross Trade (Mixed Trading) (SMA-{short_window_2} vs SMA-{long_window_2})"
)

if fig is not None:
    plt.show(block=True)
    plt.close(fig)

print("\n" + "="*80)
print("ALL CROSS TRADE BACKTESTS COMPLETED")
print("="*80)
