"""
Example: Combination Strategy Example using premade_backtest

This example demonstrates how to combine trading signals using 
the premade_backtest function for cleaner, more maintainable code.
"""

from simple_trade import download_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simple_trade.premade_backtest import premade_backtest
from simple_trade.combine_trade import CombineTradeBacktester

# Download data
print("Downloading stock data...")
# --- Backtest Configuration ---
symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2022-12-31'
interval = '1d'
data = download_data(symbol, start_date, end_date, interval=interval)

# --- Global Parameters ---
global_parameters = {
    'initial_cash': 200,
    'commission_long': 0.001,
    'commission_short': 0.001,
    'short_borrow_fee_inc_rate': 0.0,
    'long_borrow_fee_inc_rate': 0.0,
    'trading_type': 'long',
    'day1_position': 'none',
    'risk_free_rate': 0.0,
}

# RSI Strategy Parameters
rsi_params = {
    'window': 14,
    'upper': 70,
    'lower': 30,
    'fig_control': 0
}

rsi_params = {**global_parameters, **rsi_params}
rsi_results, rsi_portfolio, _ = premade_backtest(data, "rsi", rsi_params)

print(f"RSI Strategy - Final Value: ${rsi_results['final_value']:.2f}")
print(f"RSI Strategy - Total Return: {rsi_results['total_return_pct']}%")
print(f"RSI Strategy - Number of Trades: {rsi_results['num_trades']}")

# SMA Strategy Parameters  
sma_params = {
    'short_window': 20,
    'long_window': 50,
    'fig_control': 0
}

sma_params = {**global_parameters, **sma_params}
sma_results, sma_portfolio, _ = premade_backtest(data, "sma", sma_params)

print(f"SMA Strategy - Final Value: ${sma_results['final_value']:.2f}")
print(f"SMA Strategy - Total Return: {sma_results['total_return_pct']}%")
print(f"SMA Strategy - Number of Trades: {sma_results['num_trades']}")

# MACD Strategy Parameters  
macd_params = {
    'window_fast': 12,
    'window_slow': 26,
    'window_signal': 9
}
macd_params = {**global_parameters, **macd_params}
macd_results, macd_portfolio, _ = premade_backtest(data, "mac", sma_params)

print(f"MACD Strategy - Final Value: ${macd_results['final_value']:.2f}")
print(f"MACD Strategy - Total Return: {macd_results['total_return_pct']}%")
print(f"MACD Strategy - Number of Trades: {macd_results['num_trades']}")

# Initialize the CombineTradeBacktester
combine_backtester = CombineTradeBacktester(
    initial_cash=200,
    commission_long=0.001,
    commission_short=0.001
)

# Build strategies dict for plotting individual strategies
strategies = {
    'RSI': {'results': rsi_results, 'portfolio': rsi_portfolio},
    'SMA': {'results': sma_results, 'portfolio': sma_portfolio}
}

print("2 Trading Strategy Combination")
# Run the combined backtest
combined_results, combined_portfolio, figures = combine_backtester.run_combined_trade(
    portfolio_dfs=[rsi_portfolio, sma_portfolio],
    price_data=data,
    price_col='Close',
    combination_logic='majority',
    trading_type='long',
    fig_control=2,
    strategies=strategies,
    strategy_name='Majority'
)

print(f"2 Trading Strategy Combination - Final Value: ${combined_results['final_value']:.2f}")
print(f"2 Trading Strategy Combination - Total Return: {combined_results['total_return_pct']}%")
print(f"2 Trading Strategy Combination - Number of Trades: {combined_results['num_trades']}")
print(f"2 Trading Strategy Combination - Sharpe Ratio: {combined_results['sharpe_ratio']:.3f}")

if figures is not None:
    plt.show(block=True)
    for fig in figures:
        if fig is not None:
            plt.close(fig)

# Build strategies dict for plotting individual strategies
strategies = {
    'RSI': {'results': rsi_results, 'portfolio': rsi_portfolio},
    'SMA': {'results': sma_results, 'portfolio': sma_portfolio},
    'MACD': {'results': macd_results, 'portfolio': macd_portfolio}
}

print("3 Trading Strategy Combination")
# Run the combined backtest
combined_results, combined_portfolio, figures = combine_backtester.run_combined_trade(
    portfolio_dfs=[rsi_portfolio, sma_portfolio, macd_portfolio],
    price_data=data,
    price_col='Close',
    combination_logic='majority',
    trading_type='long',
    fig_control=1,
    strategies=strategies,
    strategy_name='Majority'
)

print(f"3 Trading Strategy Combination - Final Value: ${combined_results['final_value']:.2f}")
print(f"3 Trading Strategy Combination - Total Return: {combined_results['total_return_pct']}%")
print(f"3 Trading Strategy Combination - Number of Trades: {combined_results['num_trades']}")
print(f"3 Trading Strategy Combination - Sharpe Ratio: {combined_results['sharpe_ratio']:.3f}")

if figures is not None:
    plt.show(block=True)
    for fig in figures:
        if fig is not None:
            plt.close(fig)