"""
Example: Multi-Indicator Voting Strategy using premade_backtest

This example demonstrates combining multiple technical indicators (RSI, MACD, SMA, CCI)
using both unanimous and majority voting logic. This creates a robust trading system
that requires consensus from multiple indicators before taking positions.

Strategy Components:
- RSI (14): Momentum oscillator for overbought/oversold conditions
- MACD (12,26,9): Trend-following momentum indicator
- SMA Cross (20,50): Simple moving average crossover
- CCI (20): Commodity Channel Index for cyclical turns

Voting Logic:
- Unanimous: All 4 indicators must agree (very conservative)
- Majority: 3 out of 4 indicators must agree (balanced approach)
"""

from simple_trade import download_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simple_trade.premade_backtest import premade_backtest
from simple_trade.combine_trade import CombineTradeBacktester




def run_individual_strategies(data):
    """Run all individual strategies using premade_backtest."""
    print("Running individual indicator strategies...")
    
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
        'fig_control': 0
    }
    
    strategies = {}
    
    # 1. RSI Strategy
    print("  - RSI Strategy")
    rsi_params = {
        'window': 14,
        'upper': 70,
        'lower': 30
    }
    rsi_params = {**global_parameters, **rsi_params}
    rsi_results, rsi_portfolio, _ = premade_backtest(data, 'rsi', rsi_params)
    strategies['RSI'] = {'results': rsi_results, 'portfolio': rsi_portfolio}
    
    # 2. MACD Strategy
    print("  - MACD Strategy")
    macd_params = {
        'window_fast': 12,
        'window_slow': 26,
        'window_signal': 9
    }
    macd_params = {**global_parameters, **macd_params}
    macd_results, macd_portfolio, _ = premade_backtest(data, 'macd', macd_params)
    strategies['MACD'] = {'results': macd_results, 'portfolio': macd_portfolio}
    
    # 3. SMA Crossover Strategy
    print("  - SMA Crossover Strategy")
    sma_params = {
        'short_window': 20,
        'long_window': 50
    }
    sma_params = {**global_parameters, **sma_params}
    sma_results, sma_portfolio, _ = premade_backtest(data, 'sma', sma_params)
    strategies['SMA'] = {'results': sma_results, 'portfolio': sma_portfolio}
    
    # 4. CCI Strategy
    print("  - CCI Strategy")
    cci_params = {
        'window': 20,
        'constant': 0.015,
        'upper': 100,
        'lower': -100
    }
    cci_params = {**global_parameters, **cci_params}
    cci_results, cci_portfolio, _ = premade_backtest(data, 'cci', cci_params)
    strategies['CCI'] = {'results': cci_results, 'portfolio': cci_portfolio}
    
    # Print individual results
    print("\nIndividual Strategy Results:")
    print("-" * 60)
    for name, strategy in strategies.items():
        results = strategy['results']
        print(f"{name:<8} | Return: {results['total_return_pct']:>6.1f}% | "
              f"Trades: {results['num_trades']:>3} | "
              f"Sharpe: {results.get('sharpe_ratio', 'N/A'):>5}")
    
    return strategies


def analyze_signal_consensus(strategies):
    """Analyze how often indicators agree with each other."""
    print("\n" + "=" * 60)
    print("SIGNAL CONSENSUS ANALYSIS")
    print("=" * 60)
    
    # Align all portfolios
    portfolios = {}
    for name, strategy in strategies.items():
        portfolios[name] = strategy['portfolio']['PositionType']
    
    aligned_signals = pd.DataFrame(portfolios)
    
    # Convert position types to numeric for analysis
    signal_map = {'long': 1, 'short': -1, 'none': 0}
    numeric_signals = aligned_signals.applymap(lambda x: signal_map.get(x, 0))
    
    # Calculate correlation matrix
    correlation_matrix = numeric_signals.corr()
    
    print("Signal Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Calculate consensus statistics
    total_days = len(aligned_signals)
    
    # Count unanimous agreement
    unanimous_long = (aligned_signals == 'long').all(axis=1).sum()
    unanimous_short = (aligned_signals == 'short').all(axis=1).sum()
    unanimous_none = (aligned_signals == 'none').all(axis=1).sum()
    
    print(f"\nUnanimous Agreement:")
    print(f"  All Long: {unanimous_long} days ({unanimous_long/total_days*100:.1f}%)")
    print(f"  All Short: {unanimous_short} days ({unanimous_short/total_days*100:.1f}%)")
    print(f"  All None: {unanimous_none} days ({unanimous_none/total_days*100:.1f}%)")
    print(f"  Total Unanimous: {(unanimous_long + unanimous_short + unanimous_none)/total_days*100:.1f}%")
    
    # Count majority agreement (3 out of 4)
    def count_majority(row, position):
        return (row == position).sum() >= 3
    
    majority_long = aligned_signals.apply(lambda row: count_majority(row, 'long'), axis=1).sum()
    majority_short = aligned_signals.apply(lambda row: count_majority(row, 'short'), axis=1).sum()
    
    print(f"\nMajority Agreement (3+ out of 4):")
    print(f"  Majority Long: {majority_long} days ({majority_long/total_days*100:.1f}%)")
    print(f"  Majority Short: {majority_short} days ({majority_short/total_days*100:.1f}%)")
    
    return aligned_signals, correlation_matrix


def run_voting_strategies(data, strategies):
    """Run combined strategies with different voting mechanisms."""
    print("\n" + "=" * 60)
    print("RUNNING VOTING STRATEGIES")
    print("=" * 60)
    
    # Get portfolio DataFrames
    portfolio_dfs = [strategy['portfolio'] for strategy in strategies.values()]
    
    combine_backtester = CombineTradeBacktester(
        initial_cash=10000,
        commission_long=0.001,
        commission_short=0.001
    )
    
    voting_results = {}
    
    # Unanimous Voting (All 4 must agree)
    print("Running Unanimous Voting Strategy...")
    unanimous_results, unanimous_portfolio = combine_backtester.run_combined_trade(
        portfolio_dfs=portfolio_dfs,
        price_data=data,
        price_col='Close',
        combination_logic='unanimous',
        trading_type='long'
    )
    voting_results['Unanimous'] = {'results': unanimous_results, 'portfolio': unanimous_portfolio}
    
    # Majority Voting (3 out of 4 must agree)
    print("Running Majority Voting Strategy...")
    majority_results, majority_portfolio = combine_backtester.run_combined_trade(
        portfolio_dfs=portfolio_dfs,
        price_data=data,
        price_col='Close',
        combination_logic='majority',
        trading_type='long'
    )
    voting_results['Majority'] = {'results': majority_results, 'portfolio': majority_portfolio}
    
    # Print voting results
    print("\nVoting Strategy Results:")
    print("-" * 60)
    for name, strategy in voting_results.items():
        results = strategy['results']
        return_pct = results['total_return_pct']
        num_trades = results['num_trades']
        sharpe = results.get('sharpe_ratio', 'N/A')
        print(f"{name:<12} | Return: {return_pct:>6.1f}% | "
              f"Trades: {num_trades:>3} | "
              f"Sharpe: {sharpe:>5}")
    
    return voting_results


def create_comprehensive_visualization(data, strategies, voting_results, correlation_matrix):
    """Create comprehensive visualization of all strategies."""
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # Plot 1: Stock Price and Portfolio Values (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(data.index, data['Close'], label='Stock Price', color='black', alpha=0.7, linewidth=1)
    
    ax1_twin = ax1.twinx()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot individual strategies
    for i, (name, strategy) in enumerate(strategies.items()):
        portfolio = strategy['portfolio']
        results = strategy['results']
        return_pct = results['total_return_pct']
        ax1_twin.plot(portfolio.index, portfolio['PortfolioValue'], 
                     label=f"{name} ({return_pct:.1f}%)", 
                     color=colors[i], alpha=0.8, linewidth=1.5)
    
    # Plot voting strategies
    for i, (name, strategy) in enumerate(voting_results.items()):
        portfolio = strategy['portfolio']
        results = strategy['results']
        if not portfolio.empty:
            return_pct = results['total_return_pct']
            ax1_twin.plot(portfolio.index, portfolio['PortfolioValue'], 
                         label=f"{name} Voting ({return_pct:.1f}%)", 
                         color=colors[len(strategies) + i], linewidth=3, linestyle='--')
    
    ax1.set_title('Multi-Indicator Voting Strategy Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Stock Price ($)')
    ax1_twin.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1_twin.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation Heatmap
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Signal Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Plot 3: Individual Signals
    ax3 = fig.add_subplot(gs[1, :])
    signal_offset = 0
    for name, strategy in strategies.items():
        portfolio = strategy['portfolio']
        signals = portfolio['PositionType'].map({'long': 1, 'short': -1, 'none': 0})
        ax3.plot(portfolio.index, signals + signal_offset, label=f'{name} Signals', 
                alpha=0.8, linewidth=1.5)
        signal_offset += 3
    
    # Add voting signals
    for name, strategy in voting_results.items():
        portfolio = strategy['portfolio']
        if not portfolio.empty:
            signals = portfolio['PositionType'].map({'long': 1, 'short': -1, 'none': 0})
            ax3.plot(portfolio.index, signals + signal_offset, label=f'{name} Voting', 
                    linewidth=3, linestyle='--', alpha=0.9)
            signal_offset += 3
    
    ax3.set_title('Trading Signals Timeline', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Position Signals (offset for clarity)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling Sharpe Ratios (90-day)
    ax4 = fig.add_subplot(gs[2, :2])
    
    def calculate_rolling_sharpe(portfolio, window=90):
        returns = portfolio['PortfolioValue'].pct_change()
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        return rolling_sharpe
    
    for name, strategy in strategies.items():
        portfolio = strategy['portfolio']
        rolling_sharpe = calculate_rolling_sharpe(portfolio)
        ax4.plot(portfolio.index, rolling_sharpe, label=f'{name}', alpha=0.7)
    
    for name, strategy in voting_results.items():
        portfolio = strategy['portfolio']
        if not portfolio.empty:
            rolling_sharpe = calculate_rolling_sharpe(portfolio)
            ax4.plot(portfolio.index, rolling_sharpe, label=f'{name} Voting', 
                    linewidth=2, linestyle='--')
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=1, color='red', linestyle=':', alpha=0.5, label='Sharpe = 1')
    ax4.set_title('Rolling 90-Day Sharpe Ratios', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Trade Frequency Analysis
    ax5 = fig.add_subplot(gs[2, 2])
    
    strategy_names = list(strategies.keys()) + [f"{name} Voting" for name in voting_results.keys()]
    trade_counts = []
    
    for strategy in strategies.values():
        trade_counts.append(strategy['results']['num_trades'])
    
    for strategy in voting_results.values():
        trade_counts.append(strategy['results']['num_trades'])
    
    bars = ax5.bar(range(len(strategy_names)), trade_counts, 
                   color=['skyblue'] * len(strategies) + ['orange'] * len(voting_results))
    ax5.set_title('Number of Trades Comparison', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Number of Trades')
    ax5.set_xticks(range(len(strategy_names)))
    ax5.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=8)
    
    # Add value labels on bars
    for bar, count in zip(bars, trade_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=8)
    
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Performance Summary Table
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Create performance summary data
    summary_data = []
    all_strategies = {**strategies, **{f"{k} Voting": v for k, v in voting_results.items()}}
    
    for name, strategy in all_strategies.items():
        results = strategy['results']
        portfolio = strategy['portfolio']
        
        # Calculate additional metrics
        if not portfolio.empty:
            daily_returns = portfolio['PortfolioValue'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            max_dd = ((portfolio['PortfolioValue'] / portfolio['PortfolioValue'].expanding().max()) - 1).min() * 100
            win_rate = (daily_returns > 0).mean() * 100
        else:
            volatility = max_dd = win_rate = 0
        
        return_pct = results['total_return_pct']
        summary_data.append([
            name,
            f"{return_pct:.1f}%",
            f"{volatility:.1f}%",
            f"{max_dd:.1f}%",
            f"{win_rate:.1f}%",
            results['num_trades'],
            f"{results.get('sharpe_ratio', 0):.2f}" if isinstance(results.get('sharpe_ratio'), (int, float)) else 'N/A'
        ])
    
    # Create table
    table = ax6.table(cellText=summary_data,
                     colLabels=['Strategy', 'Return', 'Volatility', 'Max DD', 'Win Rate', 'Trades', 'Sharpe'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.1, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(7):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif 'Voting' in summary_data[i-1][0]:  # Voting strategies
                cell.set_facecolor('#FFF3E0')
            else:  # Individual strategies
                cell.set_facecolor('#E3F2FD')
    
    plt.suptitle('Multi-Indicator Voting Strategy Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.show()


def main():
    """Main function to run the multi-indicator voting strategy example."""
    print("=" * 80)
    print("MULTI-INDICATOR VOTING STRATEGY EXAMPLE")
    print("=" * 80)
    print("Combining RSI, MACD, SMA Cross, and CCI indicators")
    print("Testing both Unanimous and Majority voting logic")
    print("=" * 80)
    
    # Download data
    print("\nDownloading market data...")
    symbol = 'QQQ'
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    interval = '1d'
    data = download_data(symbol, start_date, end_date, interval=interval)
    print(f"Data loaded: {len(data)} days from {data.index[0].date()} to {data.index[-1].date()}")
    
    # Run individual strategies
    strategies = run_individual_strategies(data)
    
    # Analyze signal consensus
    aligned_signals, correlation_matrix = analyze_signal_consensus(strategies)
    
    # Run voting strategies
    voting_results = run_voting_strategies(data, strategies)
    
    # Create comprehensive visualization
    create_comprehensive_visualization(data, strategies, voting_results, correlation_matrix)
    
    # Final insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("1. UNANIMOUS VOTING:")
    print("   - Very conservative approach requiring all 4 indicators to agree")
    print("   - Typically results in fewer trades but higher conviction")
    print("   - May miss opportunities when indicators slightly disagree")
    
    print("\n2. MAJORITY VOTING:")
    print("   - Balanced approach requiring 3 out of 4 indicators to agree")
    print("   - More active than unanimous but still filtered")
    print("   - Good compromise between conviction and opportunity capture")
    
    print("\n3. INDICATOR CORRELATION:")
    print("   - Higher correlation = more agreement between indicators")
    print("   - Lower correlation = more diverse signal sources")
    print("   - Optimal portfolio benefits from diverse, uncorrelated signals")
    
    print("\n4. RISK CONSIDERATIONS:")
    print("   - More indicators can reduce false signals")
    print("   - But may also increase lag and miss quick opportunities")
    print("   - Balance between signal quality and responsiveness")
    
    unanimous_results = voting_results['Unanimous']['results']
    majority_results = voting_results['Majority']['results']
    unanimous_return = unanimous_results['total_return_pct']
    majority_return = majority_results['total_return_pct']
    
    print(f"\n5. PERFORMANCE SUMMARY:")
    print(f"   - Unanimous Voting: {unanimous_return:.1f}% return")
    print(f"   - Majority Voting: {majority_return:.1f}% return")
    print("   - Compare risk-adjusted metrics (Sharpe ratio) for true performance")


if __name__ == "__main__":
    main()
