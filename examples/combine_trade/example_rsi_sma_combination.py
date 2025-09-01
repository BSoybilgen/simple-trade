"""
Example: RSI + SMA Combination Strategy Example using premade_backtest

This example demonstrates how to combine RSI and SMA trading signals using 
the premade_backtest function for cleaner, more maintainable code.

The strategy combines:
1. RSI band trading (oversold/overbought signals)
2. SMA crossover trading (trend following signals)
"""

from simple_trade import download_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simple_trade.premade_backtest import premade_backtest
from simple_trade.combine_trade import CombineTradeBacktester

def run_individual_strategies(data):
    """Run individual RSI and SMA strategies using premade_backtest."""

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
    
    return {
        'rsi': {'results': rsi_results, 'portfolio': rsi_portfolio},
        'sma': {'results': sma_results, 'portfolio': sma_portfolio}
    }


def run_combined_strategy(data, rsi_portfolio, sma_portfolio, combination_logic='unanimous'):
    """Run the combined RSI + SMA strategy."""
    print(f"\n=== Running Combined Strategy ({combination_logic}) ===")
    
    # Initialize the CombineTradeBacktester
    combine_backtester = CombineTradeBacktester(
        initial_cash=200,
        commission_long=0.001,
        commission_short=0.001
    )
    
    # Run the combined backtest
    combined_results, combined_portfolio = combine_backtester.run_combined_trade(
        portfolio_dfs=[rsi_portfolio, sma_portfolio],
        price_data=data,
        price_col='Close',
        combination_logic=combination_logic,
        trading_type='long'
    )
    
    print(f"Combined Strategy - Final Value: ${combined_results['final_value']:.2f}")
    print(f"Combined Strategy - Total Return: {combined_results['total_return_pct']}%")
    print(f"Combined Strategy - Number of Trades: {combined_results['num_trades']}")
    print(f"Combined Strategy - Sharpe Ratio: {combined_results['sharpe_ratio']:.3f}")
    
    return combined_results, combined_portfolio


def analyze_signal_agreement(rsi_portfolio, sma_portfolio):
    """Analyze how often the two strategies agree."""
    print("\nAnalyzing signal agreement...")
    
    # Align the portfolios by date
    aligned_data = pd.merge(
        rsi_portfolio[['PositionType']].rename(columns={'PositionType': 'RSI_Position'}),
        sma_portfolio[['PositionType']].rename(columns={'PositionType': 'SMA_Position'}),
        left_index=True, right_index=True, how='inner'
    )
    
    # Calculate agreement statistics
    total_days = len(aligned_data)
    both_long = ((aligned_data['RSI_Position'] == 'long') & 
                 (aligned_data['SMA_Position'] == 'long')).sum()
    both_short = ((aligned_data['RSI_Position'] == 'short') & 
                  (aligned_data['SMA_Position'] == 'short')).sum()
    both_none = ((aligned_data['RSI_Position'] == 'none') & 
                 (aligned_data['SMA_Position'] == 'none')).sum()
    
    agreement_days = both_long + both_short + both_none
    agreement_pct = (agreement_days / total_days) * 100
    
    print(f"Signal Agreement Analysis:")
    print(f"  Total Days: {total_days}")
    print(f"  Both Long: {both_long} days ({both_long/total_days*100:.1f}%)")
    print(f"  Both Short: {both_short} days ({both_short/total_days*100:.1f}%)")
    print(f"  Both None: {both_none} days ({both_none/total_days*100:.1f}%)")
    print(f"  Overall Agreement: {agreement_pct:.1f}%")
    
    return aligned_data


def plot_strategy_comparison(data, rsi_portfolio, sma_portfolio, combined_portfolio, 
                           rsi_results, sma_results, combined_results):
    """Plot comparison of all strategies."""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Price and Portfolio Values
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['Close'], label='Stock Price', alpha=0.7, color='black')
    plt.plot(rsi_portfolio.index, rsi_portfolio['PortfolioValue'], 
             label=f'RSI Strategy (Return: {rsi_results["total_return_pct"]}%)', 
             linewidth=2)
    plt.plot(sma_portfolio.index, sma_portfolio['PortfolioValue'], 
             label=f'SMA Strategy (Return: {sma_results["total_return_pct"]}%)', 
             linewidth=2)
    
    if not combined_portfolio.empty:
        plt.plot(combined_portfolio.index, combined_portfolio['PortfolioValue'], 
                 label=f'Combined Strategy (Return: {combined_results["total_return_pct"]}%)', 
                 linewidth=2, linestyle='--')
    
    plt.title('Strategy Performance Comparison')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Position Types
    plt.subplot(3, 1, 2)
    
    # Create position signals for plotting
    rsi_signals = rsi_portfolio['PositionType'].map({'long': 1, 'short': -1, 'none': 0})
    sma_signals = sma_portfolio['PositionType'].map({'long': 1, 'short': -1, 'none': 0})
    
    plt.plot(rsi_portfolio.index, rsi_signals, label='RSI Signals', alpha=0.7, linewidth=2)
    plt.plot(sma_portfolio.index, sma_signals, label='SMA Signals', alpha=0.7, linewidth=2)
    
    if not combined_portfolio.empty:
        combined_signals = combined_portfolio['PositionType'].map({'long': 1, 'short': -1, 'none': 0})
        plt.plot(combined_portfolio.index, combined_signals, 
                 label='Combined Signals', linewidth=2, linestyle='--')
    
    plt.title('Trading Signals Comparison')
    plt.ylabel('Position (1=Long, -1=Short, 0=None)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown Analysis
    plt.subplot(3, 1, 3)
    
    # Calculate drawdowns
    rsi_peak = rsi_portfolio['PortfolioValue'].expanding().max()
    rsi_drawdown = (rsi_portfolio['PortfolioValue'] - rsi_peak) / rsi_peak * 100
    
    sma_peak = sma_portfolio['PortfolioValue'].expanding().max()
    sma_drawdown = (sma_portfolio['PortfolioValue'] - sma_peak) / sma_peak * 100
    
    plt.plot(rsi_portfolio.index, rsi_drawdown, label='RSI Drawdown', alpha=0.7)
    plt.plot(sma_portfolio.index, sma_drawdown, label='SMA Drawdown', alpha=0.7)
    
    if not combined_portfolio.empty:
        combined_peak = combined_portfolio['PortfolioValue'].expanding().max()
        combined_drawdown = (combined_portfolio['PortfolioValue'] - combined_peak) / combined_peak * 100
        plt.plot(combined_portfolio.index, combined_drawdown, 
                 label='Combined Drawdown', linestyle='--')
    
    plt.title('Drawdown Analysis')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the RSI + SMA combination strategy example."""
    print("=" * 60)
    print("RSI + SMA Combination Strategy Example")
    print("=" * 60)
    
    # Download data
    print("Downloading stock data...")
    # --- Backtest Configuration ---
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    interval = '1d'
    data = download_data(symbol, start_date, end_date, interval=interval)
    
    # Run individual strategies
    individual_results = run_individual_strategies(data)
    rsi_results = individual_results['rsi']['results']
    rsi_portfolio = individual_results['rsi']['portfolio']
    sma_results = individual_results['sma']['results']
    sma_portfolio = individual_results['sma']['portfolio']
    
    # Analyze signal agreement
    signal_analysis = analyze_signal_agreement(rsi_portfolio, sma_portfolio)
    
    # Run combined strategy with unanimous logic
    combined_results_unanimous, combined_portfolio_unanimous = run_combined_strategy(
        data, rsi_portfolio, sma_portfolio, 'unanimous'
    )
    
    # Run combined strategy with majority logic (for comparison)
    combined_results_majority, combined_portfolio_majority = run_combined_strategy(
        data, rsi_portfolio, sma_portfolio, 'majority'
    )
    
    # Performance Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    strategies = [
        ("RSI Only", rsi_results),
        ("SMA Only", sma_results),
        ("Combined (Unanimous)", combined_results_unanimous),
        ("Combined (Majority)", combined_results_majority)
    ]
    
    print(f"{'Strategy':<20} {'Final Value':<12} {'Return %':<10} {'Trades':<8} {'Sharpe':<8}")
    print("-" * 60)
    
    for name, results in strategies:
        final_value = results['final_value']
        return_pct = results['total_return_pct']
        num_trades = results['num_trades']
        sharpe = results.get('sharpe_ratio', 'N/A')
        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        
        print(f"{name:<20} ${final_value:<11,.0f} {return_pct:<9.1f}% {num_trades:<7} {sharpe_str:<8}")
    
    # Plot comparison
    plot_strategy_comparison(
        data, rsi_portfolio, sma_portfolio, combined_portfolio_unanimous,
        rsi_results, sma_results, combined_results_unanimous
    )
    
    print("\n" + "=" * 60)
    print("STRATEGY INSIGHTS")
    print("=" * 60)
    print("1. Unanimous Logic: More conservative, only trades when both indicators agree")
    print("2. Majority Logic: More aggressive, trades when either indicator signals")
    print("3. Combined strategies often have fewer trades but potentially better risk-adjusted returns")
    print("4. Signal agreement analysis helps understand strategy correlation")


if __name__ == "__main__":
    main()
