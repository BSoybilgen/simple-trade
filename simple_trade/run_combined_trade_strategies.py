"""
Combined Trade backtesting functions.

This module provides function-based implementations for combining multiple
trading strategies, replacing the class-based CombineTradeBacktester approach.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

from .config import BacktestConfig
from .metrics import compute_benchmark_return, calculate_performance_metrics


def run_combined_trade(
    portfolio_dfs: List[pd.DataFrame],
    price_data: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    price_col: str = 'Close',
    trading_type: str = 'long',
    combination_logic: str = 'unanimous',
    fig_control: int = 0,
    strategies: Optional[Dict] = None,
    strategy_name: str = 'Combined',
    # Legacy parameters for backward compatibility
    initial_cash: Optional[float] = None,
    commission_long: Optional[float] = None,
    commission_short: Optional[float] = None,
    short_borrow_fee_inc_rate: Optional[float] = None,
    long_borrow_fee_inc_rate: Optional[float] = None,
    long_entry_pct_cash: Optional[float] = None,
    short_entry_pct_cash: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
) -> tuple:
    """
    Runs a backtest based on combined signals from multiple portfolio DataFrames.

    Args:
        portfolio_dfs: A list of portfolio DataFrames from other backtests.
                      Each DataFrame must contain a 'PositionType' column
                      and have a DatetimeIndex.
        price_data: DataFrame with historical price data, including the price_col.
                   Must have a DatetimeIndex.
        config: BacktestConfig object with all configuration parameters.
        price_col: Column name for trade execution prices.
        trading_type: Defines trading behavior ('long', 'short', 'mixed').
        combination_logic: Logic for combining signals ('unanimous' or 'majority').
        fig_control: Controls figure generation:
                    0 = No figures (default)
                    1 = Create and show figures
                    2 = Create figures but don't show (return only)
        strategies: Optional dict of individual strategies for plotting.
                   Structure: {'Name': {'results': dict, 'portfolio': DataFrame}}
        strategy_name: Name for this combined strategy in plots.
        
        # Legacy parameters (used if config is None):
        initial_cash: Starting cash balance for the backtest.
        commission_long: Commission rate for long trades.
        commission_short: Commission rate for short trades.
        short_borrow_fee_inc_rate: Time-based fee rate for holding short positions.
        long_borrow_fee_inc_rate: Time-based fee rate for holding long positions.
        long_entry_pct_cash: Pct of available cash to use for long entries.
        short_entry_pct_cash: Pct of available cash defining the value of short entries.
        risk_free_rate: Risk-free rate for Sharpe and Sortino ratios.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary with backtest summary results.
            - pd.DataFrame: DataFrame tracking daily portfolio evolution.
            - tuple or None: (fig_performance, fig_signals, fig_table) if fig_control > 0, else None.
            
    Example:
        >>> config = BacktestConfig(initial_cash=50000)
        >>> results, portfolio, figs = run_combined_trade(
        ...     [portfolio1, portfolio2], price_data,
        ...     config=config, combination_logic='unanimous'
        ... )
    """
    # Build config from individual parameters if not provided
    if config is None:
        config = BacktestConfig(
            initial_cash=initial_cash if initial_cash is not None else 10000.0,
            commission_long=commission_long if commission_long is not None else 0.001,
            commission_short=commission_short if commission_short is not None else 0.001,
            short_borrow_fee_inc_rate=short_borrow_fee_inc_rate if short_borrow_fee_inc_rate is not None else 0.0,
            long_borrow_fee_inc_rate=long_borrow_fee_inc_rate if long_borrow_fee_inc_rate is not None else 0.0,
            long_entry_pct_cash=long_entry_pct_cash if long_entry_pct_cash is not None else 0.9,
            short_entry_pct_cash=short_entry_pct_cash if short_entry_pct_cash is not None else 0.9,
            risk_free_rate=risk_free_rate if risk_free_rate is not None else 0.0,
        )
    
    # --- Input Validation ---
    if combination_logic not in ['unanimous', 'majority']:
        raise ValueError("combination_logic must be either 'unanimous' or 'majority'.")
    if not isinstance(portfolio_dfs, list) or not portfolio_dfs:
        raise ValueError("portfolio_dfs must be a non-empty list of DataFrames.")
    if not isinstance(price_data, pd.DataFrame) or not isinstance(price_data.index, pd.DatetimeIndex):
        raise TypeError("price_data must be a DataFrame with a DatetimeIndex.")
    if price_col not in price_data.columns:
        raise ValueError(f"Price column '{price_col}' not found in price_data.")

    # --- Signal Combination ---
    df = _combine_signals(portfolio_dfs, price_data, price_col, combination_logic)

    if df.empty:
        empty_results = _get_empty_combined_results(config)
        if fig_control > 0:
            return empty_results, pd.DataFrame(), (None, None, None)
        return empty_results, pd.DataFrame(), None

    # --- Run Backtest ---
    portfolio_log, end_state = _run_combined_backtest(
        signal_df=df,
        config=config,
        price_col=price_col,
        trading_type=trading_type,
    )

    # --- Prepare Results ---
    results, portfolio_df = _prepare_combined_results(
        portfolio_log=portfolio_log,
        original_data=price_data,
        config=config,
        price_col=price_col,
        trading_type=trading_type,
    )

    # --- Generate Figures if requested ---
    figures = None
    if fig_control > 0:
        voting_results = {
            strategy_name: {'results': results, 'portfolio': portfolio_df}
        }
        strat_dict = strategies if strategies is not None else {}
        
        figures = plot_combined_results(
            price_data=price_data,
            strategies=strat_dict,
            voting_results=voting_results,
            price_col=price_col,
            fig_control=fig_control
        )

    return results, portfolio_df, figures


def _combine_signals(
    portfolio_dfs: List[pd.DataFrame],
    price_data: pd.DataFrame,
    price_col: str,
    combination_logic: str
) -> pd.DataFrame:
    """
    Merges PositionType from multiple DataFrames and generates final buy/sell signals.
    """
    combined_df = price_data[[price_col]].copy()

    for i, portfolio_df in enumerate(portfolio_dfs):
        if 'PositionType' not in portfolio_df.columns:
            raise ValueError(f"DataFrame at index {i} is missing 'PositionType' column.")
        if not isinstance(portfolio_df.index, pd.DatetimeIndex):
            raise TypeError(f"Index of DataFrame at index {i} must be a DatetimeIndex.")

        position_col = portfolio_df[['PositionType']].rename(columns={'PositionType': f'PositionType_{i}'})
        combined_df = combined_df.join(position_col, how='left')

    position_cols = [f'PositionType_{i}' for i in range(len(portfolio_dfs))]
    combined_df[position_cols] = combined_df[position_cols].ffill()
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        return pd.DataFrame()

    def get_final_position(row):
        positions = [row[col] for col in position_cols]

        if combination_logic == 'unanimous':
            is_all_long = all(p == 'long' for p in positions)
            is_all_short = all(p == 'short' for p in positions)
            if is_all_long:
                return 'long'
            elif is_all_short:
                return 'short'
            else:
                return 'none'

        elif combination_logic == 'majority':
            long_votes = positions.count('long')
            short_votes = positions.count('short')
            if long_votes > short_votes:
                return 'long'
            elif short_votes > long_votes:
                return 'short'
            else:
                return 'none'

        return 'none'

    combined_df['PositionType'] = combined_df.apply(get_final_position, axis=1)
    combined_df['prev_PositionType'] = combined_df['PositionType'].shift(1).fillna('none')

    buy_entry = (combined_df['PositionType'] == 'long') & (combined_df['prev_PositionType'] != 'long')
    short_exit = (combined_df['PositionType'] == 'long') & (combined_df['prev_PositionType'] == 'short')
    combined_df['buy_signal'] = buy_entry | short_exit

    short_entry = (combined_df['PositionType'] == 'short') & (combined_df['prev_PositionType'] != 'short')
    long_exit = (combined_df['PositionType'] != 'long') & (combined_df['prev_PositionType'] == 'long')
    combined_df['sell_signal'] = short_entry | long_exit

    return combined_df


def _run_combined_backtest(
    signal_df: pd.DataFrame,
    config: BacktestConfig,
    price_col: str,
    trading_type: str,
) -> tuple:
    """Runs the backtest simulation loop."""
    portfolio_log = []
    cash = config.initial_cash
    position_size = 0
    position_value = 0.0
    position_type = 'none'

    for date, row in signal_df.iterrows():
        current_price = row[price_col]
        buy_signal = row['buy_signal']
        sell_signal = row['sell_signal']
        action = 'HOLD'
        commission_paid = 0.0

        start_of_day_position_type = position_type
        start_of_day_position_value = abs(position_size) * current_price

        short_fee = 0.0
        long_fee = 0.0
        if start_of_day_position_type == 'short':
            short_fee = start_of_day_position_value * config.short_borrow_fee_inc_rate
            cash -= short_fee
        elif start_of_day_position_type == 'long':
            long_fee = start_of_day_position_value * config.long_borrow_fee_inc_rate
            cash -= long_fee

        if position_type == 'long':
            position_value = position_size * current_price
        elif position_type == 'short':
            position_value = abs(position_size) * current_price
        else:
            position_value = 0.0

        if trading_type == 'long':
            if buy_signal and position_type != 'long':
                shares_to_buy = int((cash * config.long_entry_pct_cash) / current_price)
                if shares_to_buy > 0:
                    commission = shares_to_buy * current_price * config.commission_long
                    cash -= (shares_to_buy * current_price + commission)
                    position_size = shares_to_buy
                    position_value = shares_to_buy * current_price
                    position_type = 'long'
                    commission_paid += commission
                    action = 'BUY'
            elif sell_signal and position_type == 'long':
                commission = position_value * config.commission_long
                cash += (position_value - commission)
                position_size = 0
                position_value = 0.0
                position_type = 'none'
                commission_paid += commission
                action = 'SELL'

        elif trading_type == 'short':
            if sell_signal and position_type != 'short':
                shares_to_short = int((cash * config.short_entry_pct_cash) / current_price)
                if shares_to_short > 0:
                    commission = shares_to_short * current_price * config.commission_short
                    cash += (shares_to_short * current_price - commission)
                    position_size = -shares_to_short
                    position_value = abs(position_size) * current_price
                    position_type = 'short'
                    commission_paid += commission
                    action = 'SHORT'
            elif buy_signal and position_type == 'short':
                commission = position_value * config.commission_short
                cash -= (position_value + commission)
                position_size = 0
                position_value = 0.0
                position_type = 'none'
                commission_paid += commission
                action = 'COVER'

        elif trading_type == 'mixed':
            if buy_signal:
                prev_position_type = position_type
                if position_type == 'short':
                    commission = position_value * config.commission_short
                    cash -= (position_value + commission)
                    commission_paid += commission
                    position_size = 0
                    position_value = 0.0
                    position_type = 'none'

                if position_type != 'long':
                    shares_to_buy = int((cash * config.long_entry_pct_cash) / current_price)
                    if shares_to_buy > 0:
                        commission = shares_to_buy * current_price * config.commission_long
                        cash -= (shares_to_buy * current_price + commission)
                        position_size = shares_to_buy
                        position_value = shares_to_buy * current_price
                        position_type = 'long'
                        commission_paid += commission
                        action = 'COVER AND BUY' if prev_position_type == 'short' else 'BUY'

            elif sell_signal:
                prev_position_type = position_type
                if position_type == 'long':
                    commission = position_value * config.commission_long
                    cash += (position_value - commission)
                    commission_paid += commission
                    position_size = 0
                    position_value = 0.0
                    position_type = 'none'

                if position_type != 'short':
                    shares_to_short = int((cash * config.short_entry_pct_cash) / current_price)
                    if shares_to_short > 0:
                        commission = shares_to_short * current_price * config.commission_short
                        cash += (shares_to_short * current_price - commission)
                        position_size = -shares_to_short
                        position_value = abs(position_size) * current_price
                        position_type = 'short'
                        commission_paid += commission
                        action = 'SELL AND SHORT' if prev_position_type == 'long' else 'SHORT'

        portfolio_value = cash
        if position_type == 'long':
            portfolio_value += position_value
        elif position_type == 'short':
            portfolio_value -= position_value

        portfolio_log.append({
            'Date': date,
            'Close': current_price,
            'Cash': cash,
            'PositionSize': position_size,
            'PositionValue': position_value,
            'PositionType': position_type,
            'PortfolioValue': portfolio_value,
            'CommissionPaid': commission_paid,
            'ShortFee': short_fee,
            'LongFee': long_fee,
            'BuySignal': buy_signal,
            'SellSignal': sell_signal,
            'Action': action
        })

    end_state = pd.DataFrame(portfolio_log).set_index('Date') if portfolio_log else pd.DataFrame()
    return portfolio_log, end_state


def _prepare_combined_results(
    portfolio_log: list,
    original_data: pd.DataFrame,
    config: BacktestConfig,
    price_col: str,
    trading_type: str,
) -> tuple:
    """Prepares the final results dictionary and portfolio DataFrame."""
    if not portfolio_log:
        return _get_empty_combined_results(config), pd.DataFrame()

    portfolio_df = pd.DataFrame(portfolio_log).set_index('Date')
    num_trades = len(portfolio_df[portfolio_df['Action'].isin(
        ['BUY', 'SELL', 'SHORT', 'COVER', 'COVER AND BUY', 'SELL AND SHORT']
    )])

    performance_metrics = calculate_performance_metrics(portfolio_df, config.risk_free_rate)
    benchmark_metrics = compute_benchmark_return(original_data, config.initial_cash, config.commission_long, price_col)

    results = {
        "strategy": f"Combined Strategy ({trading_type})",
        "initial_cash": config.initial_cash,
        "final_value": portfolio_df['PortfolioValue'].iloc[-1],
        "num_trades": num_trades,
    }
    results.update(performance_metrics)
    results.update(benchmark_metrics)

    return results, portfolio_df


def _get_empty_combined_results(config: BacktestConfig) -> dict:
    """Returns a dictionary for an empty/failed backtest."""
    return {
        "error": "Could not run combined backtest. Check input DataFrames.",
        "strategy": "Combined Strategy",
        "initial_cash": config.initial_cash,
        "final_value": config.initial_cash,
        "total_return_pct": 0.0,
        "num_trades": 0,
    }


def plot_combined_results(
    price_data: pd.DataFrame,
    strategies: dict,
    voting_results: dict,
    price_col: str = 'Close',
    fig_control: int = 1
) -> tuple:
    """
    Create visualization figures for combined trading strategy results.
    
    Generates three separate figures:
    1. Performance figure: Stock price and portfolio values
    2. Signals timeline figure: Trading signals for each strategy
    3. Summary table figure: Performance metrics comparison
    
    Args:
        price_data: DataFrame with price data and DatetimeIndex.
        strategies: Dictionary of individual strategies with structure:
                   {'StrategyName': {'results': dict, 'portfolio': DataFrame}}
        voting_results: Dictionary of voting strategies with same structure.
        price_col: Column name for price data.
        fig_control: Controls figure display behavior:
                    0 = Don't create figures
                    1 = Create and show figures
                    2 = Create figures but don't show (return only)
    
    Returns:
        tuple: (fig_performance, fig_signals, fig_table) - Three matplotlib Figure objects.
               Returns (None, None, None) if fig_control is 0.
    """
    if fig_control == 0:
        return None, None, None
    
    strategy_colors = ['blue', 'red', 'green', 'orange', 'cyan', 'magenta', 'lime', 'pink']
    voting_colors = ['purple', 'brown', 'darkred', 'darkblue']
    
    # ==================== FIGURE 1: Performance ====================
    fig_performance, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.plot(price_data.index, price_data[price_col], 
             label='Stock Price', color='black', alpha=0.7, linewidth=1)
    ax1.set_ylabel('Stock Price ($)', fontsize=10)
    ax1.set_xlabel('')
    
    ax1_twin = ax1.twinx()
    
    for i, (name, strategy) in enumerate(strategies.items()):
        portfolio = strategy['portfolio']
        results = strategy['results']
        if not portfolio.empty:
            return_pct = results.get('total_return_pct', 0)
            color = strategy_colors[i % len(strategy_colors)]
            ax1_twin.plot(portfolio.index, portfolio['PortfolioValue'], 
                         label=f"{name} ({return_pct:.1f}%)", 
                         color=color, alpha=0.8, linewidth=1.5)
    
    for i, (name, strategy) in enumerate(voting_results.items()):
        portfolio = strategy['portfolio']
        results = strategy['results']
        if not portfolio.empty:
            return_pct = results.get('total_return_pct', 0)
            color = voting_colors[i % len(voting_colors)]
            ax1_twin.plot(portfolio.index, portfolio['PortfolioValue'], 
                         label=f"{name} Voting ({return_pct:.1f}%)", 
                         color=color, linewidth=2.5, linestyle='--')
    
    ax1_twin.set_ylabel('Portfolio Value ($)', fontsize=10)
    
    ax1.set_title('Multi-Indicator Voting Strategy Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1_twin.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ==================== FIGURE 2: Signals Timeline ====================
    fig_signals, ax2 = plt.subplots(figsize=(14, 5))
    
    signal_offset = 0
    offset_step = 3
    
    for i, (name, strategy) in enumerate(strategies.items()):
        portfolio = strategy['portfolio']
        if not portfolio.empty:
            signals = portfolio['PositionType'].map({'long': 1, 'short': -1, 'none': 0})
            color = strategy_colors[i % len(strategy_colors)]
            ax2.plot(portfolio.index, signals + signal_offset, 
                    label=f'{name} Signals', color=color, alpha=0.8, linewidth=1.5)
            signal_offset += offset_step
    
    for i, (name, strategy) in enumerate(voting_results.items()):
        portfolio = strategy['portfolio']
        if not portfolio.empty:
            signals = portfolio['PositionType'].map({'long': 1, 'short': -1, 'none': 0})
            color = voting_colors[i % len(voting_colors)]
            ax2.plot(portfolio.index, signals + signal_offset, 
                    label=f'{name} Voting', color=color, linewidth=2.5, linestyle='--', alpha=0.9)
            signal_offset += offset_step
    
    ax2.set_title('Trading Signals Timeline', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Position Signals (offset for clarity)', fontsize=10)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ==================== FIGURE 3: Summary Table ====================
    fig_table, ax3 = plt.subplots(figsize=(14, 4))
    ax3.axis('off')
    
    summary_data = []
    
    for name, strategy in strategies.items():
        results = strategy['results']
        portfolio = strategy['portfolio']
        
        if not portfolio.empty:
            daily_returns = portfolio['PortfolioValue'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            max_dd = ((portfolio['PortfolioValue'] / portfolio['PortfolioValue'].expanding().max()) - 1).min() * 100
            win_rate = (daily_returns > 0).mean() * 100
        else:
            volatility = max_dd = win_rate = 0
        
        return_pct = results.get('total_return_pct', 0)
        sharpe = results.get('sharpe_ratio', 0)
        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else 'N/A'
        
        summary_data.append([
            name,
            f"{return_pct:.1f}%",
            f"{volatility:.1f}%",
            f"{max_dd:.1f}%",
            f"{win_rate:.1f}%",
            results.get('num_trades', 0),
            sharpe_str
        ])
    
    for name, strategy in voting_results.items():
        results = strategy['results']
        portfolio = strategy['portfolio']
        
        if not portfolio.empty:
            daily_returns = portfolio['PortfolioValue'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            max_dd = ((portfolio['PortfolioValue'] / portfolio['PortfolioValue'].expanding().max()) - 1).min() * 100
            win_rate = (daily_returns > 0).mean() * 100
        else:
            volatility = max_dd = win_rate = 0
        
        return_pct = results.get('total_return_pct', 0)
        sharpe = results.get('sharpe_ratio', 0)
        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else 'N/A'
        
        summary_data.append([
            f"{name} Voting",
            f"{return_pct:.1f}%",
            f"{volatility:.1f}%",
            f"{max_dd:.1f}%",
            f"{win_rate:.1f}%",
            results.get('num_trades', 0),
            sharpe_str
        ])
    
    col_labels = ['Strategy', 'Return', 'Volatility', 'Max DD', 'Win Rate', 'Trades', 'Sharpe']
    table = ax3.table(
        cellText=summary_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.1, 0.12]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    num_strategies = len(strategies)
    for i in range(len(summary_data) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i > num_strategies:
                cell.set_facecolor('#FFF3E0')
            else:
                cell.set_facecolor('#E3F2FD')
    
    plt.tight_layout()
    
    if fig_control == 1:
        plt.show()
    
    return fig_performance, fig_signals, fig_table


