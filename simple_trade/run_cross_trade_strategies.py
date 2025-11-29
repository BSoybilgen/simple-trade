"""
Cross Trade backtesting functions.

This module provides function-based implementations for cross trade strategies,
replacing the class-based CrossTradeBacktester approach.
"""
import warnings
import pandas as pd
from typing import Optional

from .config import BacktestConfig
from .metrics import compute_benchmark_return, calculate_performance_metrics


def run_cross_trade(
    data: pd.DataFrame,
    short_window_indicator: str,
    long_window_indicator: str,
    config: Optional[BacktestConfig] = None,
    price_col: str = 'Close',
    trading_type: str = 'long',
    day1_position: str = 'none',
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
    Runs a backtest for a cross trading strategy.

    Trading behavior is determined by the `trading_type` argument:
    - 'long': Buys when short-term crosses above long-term, sells when below.
    - 'short': Shorts when short-term crosses below long-term, covers when above.
    - 'mixed': Allows both long and short positions with seamless transitions.

    Args:
        data: DataFrame containing price data. Must have a DatetimeIndex.
        short_window_indicator: Column name for the short-term indicator.
        long_window_indicator: Column name for the long-term indicator.
        config: BacktestConfig object with all configuration parameters.
        price_col: Column name to use for trade execution prices.
        trading_type: Defines the trading behavior ('long', 'short', 'mixed').
        day1_position: Specifies whether to take a position on day 1 ('none', 'long', 'short').
        
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
            
    Example:
        >>> config = BacktestConfig(initial_cash=50000, commission_long=0.002)
        >>> results, portfolio = run_cross_trade(
        ...     data, 'SMA_20', 'SMA_50',
        ...     config=config, trading_type='long'
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
    _validate_cross_trade_inputs(
        data, short_window_indicator, long_window_indicator,
        price_col, trading_type, day1_position
    )

    df = data.copy()

    # --- Signal Generation ---
    prev_short = df[short_window_indicator].shift(1)
    prev_long = df[long_window_indicator].shift(1)
    prev_prev_short = df[short_window_indicator].shift(2)
    prev_prev_long = df[long_window_indicator].shift(2)

    # Golden Cross (Buy/Cover Signal): Short crossed above Long on the previous day
    df['buy_signal'] = (prev_short > prev_long) & (prev_prev_short <= prev_prev_long)
    # Death Cross (Sell/Short Signal): Short crossed below Long on the previous day
    df['sell_signal'] = (prev_short < prev_long) & (prev_prev_short >= prev_prev_long)

    df = df.dropna(how='any')

    # Check if DataFrame is empty after signal generation
    if df.empty:
        warnings.warn(
            f"DataFrame is empty after generating signals for indicators "
            f"'{short_window_indicator}' and '{long_window_indicator}'. No trades executed."
        )
        return _get_empty_cross_results(
            config, short_window_indicator, long_window_indicator, trading_type, day1_position
        ), pd.DataFrame()

    # --- Run Backtest ---
    portfolio_log, num_trades = _run_cross_backtest(
        signal_df=df,
        config=config,
        price_col=price_col,
        trading_type=trading_type,
        day1_position=day1_position,
    )

    # --- Prepare Results ---
    portfolio_df = pd.DataFrame(portfolio_log)
    
    if portfolio_df.empty:
        return _get_empty_cross_results(
            config, short_window_indicator, long_window_indicator, trading_type, day1_position
        ), pd.DataFrame()
    
    portfolio_df.set_index('Date', inplace=True)
    if not isinstance(portfolio_df.index, pd.DatetimeIndex):
        portfolio_df.index = pd.to_datetime(portfolio_df.index)

    # Calculate final metrics
    final_value = portfolio_df['PortfolioValue'].iloc[-1] if not portfolio_df.empty else config.initial_cash
    total_return_pct = ((final_value / config.initial_cash) - 1) * 100 if config.initial_cash else 0

    strategy_name = _build_cross_strategy_name(
        short_window_indicator, long_window_indicator, trading_type, day1_position
    )
    
    results = {
        "strategy": strategy_name,
        "short_window_indicator": short_window_indicator,
        "long_window_indicator": long_window_indicator,
        "initial_cash": config.initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return_pct, 2),
        "num_trades": num_trades,
    }

    # Calculate benchmark and performance metrics
    data_traded = data.loc[portfolio_df.index]
    benchmark_results = compute_benchmark_return(data_traded, config.initial_cash, config.commission_long, price_col)
    performance_metrics = calculate_performance_metrics(portfolio_df.copy(), config.risk_free_rate)

    results.update(benchmark_results)
    results.update(performance_metrics)

    return results, portfolio_df


def _validate_cross_trade_inputs(
    data: pd.DataFrame,
    short_window_indicator: str,
    long_window_indicator: str,
    price_col: str,
    trading_type: str,
    day1_position: str
) -> None:
    """Validates all inputs for cross trade backtest."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
    if short_window_indicator not in data.columns:
        raise ValueError(f"Required column '{short_window_indicator}' for short window indicator is missing from the DataFrame.")
    if long_window_indicator not in data.columns:
        raise ValueError(f"Required column '{long_window_indicator}' for long window indicator is missing from the DataFrame.")

    valid_trading_types = ['long', 'short', 'mixed']
    if trading_type not in valid_trading_types:
        raise ValueError(f"Invalid trading_type '{trading_type}'. Must be one of {valid_trading_types}")
        
    valid_day1_positions = ['none', 'long', 'short']
    if day1_position not in valid_day1_positions:
        raise ValueError(f"Invalid day1_position '{day1_position}'. Must be one of {valid_day1_positions}")
    
    if day1_position == 'long' and trading_type == 'short':
        raise ValueError("Cannot use day1_position='long' with trading_type='short'")
    if day1_position == 'short' and trading_type == 'long':
        raise ValueError("Cannot use day1_position='short' with trading_type='long'")


def _run_cross_backtest(
    signal_df: pd.DataFrame,
    config: BacktestConfig,
    price_col: str,
    trading_type: str,
    day1_position: str,
) -> tuple:
    """Runs the backtest simulation based on the generated signals."""
    cash = config.initial_cash
    position_size = 0
    position_type = 'none'
    position_cost_basis = 0
    num_trades = 0
    portfolio_log = []
    first_day = True

    for idx, row in signal_df.iterrows():
        trade_price = row[price_col]
        
        # Special handling for first day if day1_position is specified
        if first_day and day1_position != 'none':
            buy_signal = day1_position == 'long'
            sell_signal = day1_position == 'short'
            first_day = False
        else:
            buy_signal = row['buy_signal']
            sell_signal = row['sell_signal']
        
        action_taken = "HOLD"
        commission_paid = 0.0
        short_fee = 0.0
        long_fee = 0.0
        
        # Calculate position value and apply fees
        position_value = 0.0
        if position_type == 'long':
            position_value = position_size * trade_price
            if config.long_borrow_fee_inc_rate > 0:
                long_fee = position_value * config.long_borrow_fee_inc_rate
                cash -= long_fee
        elif position_type == 'short':
            position_value = abs(position_size) * trade_price
            if config.short_borrow_fee_inc_rate > 0:
                short_fee = position_value * config.short_borrow_fee_inc_rate
                cash -= short_fee
        
        # Calculate portfolio value
        portfolio_value = cash
        if position_type == 'long':
            portfolio_value += position_value
        elif position_type == 'short':
            portfolio_value -= position_value
        
        # Execute trading logic
        if trading_type == 'long':
            cash, position_size, position_type, position_cost_basis, commission_paid, action_taken, num_trades = \
                _execute_cross_long_only(
                    buy_signal, sell_signal, position_type, position_size,
                    cash, trade_price, config, num_trades
                )
        
        elif trading_type == 'short':
            cash, position_size, position_type, position_cost_basis, commission_paid, action_taken, num_trades = \
                _execute_cross_short_only(
                    buy_signal, sell_signal, position_type, position_size,
                    cash, trade_price, config, num_trades
                )
        
        else:  # mixed
            cash, position_size, position_type, position_cost_basis, commission_paid, action_taken, num_trades = \
                _execute_cross_mixed(
                    buy_signal, sell_signal, position_type, position_size,
                    cash, trade_price, config, num_trades
                )
        
        # Recalculate position value after trades
        if position_type == 'long':
            position_value = position_size * trade_price
        elif position_type == 'short':
            position_value = abs(position_size) * trade_price
        else:
            position_value = 0.0
        
        # Recalculate portfolio value after trades
        portfolio_value = cash
        if position_type == 'long':
            portfolio_value += position_value
        elif position_type == 'short':
            portfolio_value -= position_value
        
        # Convert signal_generated to boolean signals for logging
        log_buy_signal = action_taken in ['BUY', 'COVER AND BUY', 'COVER']
        log_sell_signal = action_taken in ['SELL', 'SELL AND SHORT', 'SHORT']
        
        log_entry = {
            'Date': idx,
            'Price': trade_price,
            'Close': trade_price,
            'Cash': cash,
            'PositionSize': position_size,
            'PositionValue': position_value,
            'PositionType': position_type,
            'PortfolioValue': portfolio_value,
            'CommissionPaid': commission_paid,
            'ShortFee': short_fee,
            'LongFee': long_fee,
            'BuySignal': log_buy_signal,
            'SellSignal': log_sell_signal,
            'Action': action_taken,
            'PositionCostBasis': position_cost_basis
        }
        portfolio_log.append(log_entry)

    return portfolio_log, num_trades


def _execute_cross_long_only(
    buy_signal: bool,
    sell_signal: bool,
    position_type: str,
    position_size: int,
    cash: float,
    trade_price: float,
    config: BacktestConfig,
    num_trades: int
) -> tuple:
    """Execute long-only trading logic for cross trade."""
    action_taken = "HOLD"
    commission_paid = 0.0
    position_cost_basis = 0
    
    if position_type == 'none' and buy_signal:
        max_shares = int((cash * config.long_entry_pct_cash) / (trade_price * (1 + config.commission_long)))
        
        if max_shares > 0:
            position_size = max_shares
            position_type = 'long'
            commission_cost = position_size * trade_price * config.commission_long
            cash -= (position_size * trade_price + commission_cost)
            position_cost_basis = trade_price
            commission_paid = commission_cost
            num_trades += 1
            action_taken = "BUY"
        else:
            action_taken = "INSUFFICIENT_CASH"
            
    elif position_type == 'long' and sell_signal:
        commission_cost = position_size * trade_price * config.commission_long
        cash += (position_size * trade_price - commission_cost)
        commission_paid = commission_cost
        position_size = 0
        position_type = 'none'
        position_cost_basis = 0
        num_trades += 1
        action_taken = "SELL"

    return cash, position_size, position_type, position_cost_basis, commission_paid, action_taken, num_trades


def _execute_cross_short_only(
    buy_signal: bool,
    sell_signal: bool,
    position_type: str,
    position_size: int,
    cash: float,
    trade_price: float,
    config: BacktestConfig,
    num_trades: int
) -> tuple:
    """Execute short-only trading logic for cross trade."""
    action_taken = "HOLD"
    commission_paid = 0.0
    position_cost_basis = 0
    
    if position_type == 'none' and sell_signal:
        short_position_value = cash * config.short_entry_pct_cash
        max_shares = int(short_position_value / (trade_price * (1 + config.commission_short)))
        
        if max_shares > 0:
            position_size = -max_shares
            position_type = 'short'
            commission_cost = abs(position_size) * trade_price * config.commission_short
            cash += (abs(position_size) * trade_price - commission_cost)
            position_cost_basis = trade_price
            commission_paid = commission_cost
            num_trades += 1
            action_taken = "SHORT"
        else:
            action_taken = "INSUFFICIENT_CASH"
            
    elif position_type == 'short' and buy_signal:
        commission_cost = abs(position_size) * trade_price * config.commission_short
        cash -= (abs(position_size) * trade_price + commission_cost)
        commission_paid = commission_cost
        position_size = 0
        position_type = 'none'
        position_cost_basis = 0
        num_trades += 1
        action_taken = "COVER"

    return cash, position_size, position_type, position_cost_basis, commission_paid, action_taken, num_trades


def _execute_cross_mixed(
    buy_signal: bool,
    sell_signal: bool,
    position_type: str,
    position_size: int,
    cash: float,
    trade_price: float,
    config: BacktestConfig,
    num_trades: int
) -> tuple:
    """Execute mixed (long and short) trading logic for cross trade."""
    action_taken = "HOLD"
    commission_paid = 0.0
    position_cost_basis = 0
    
    if position_type == 'none':
        if buy_signal:
            max_shares = int((cash * config.long_entry_pct_cash) / (trade_price * (1 + config.commission_long)))
            
            if max_shares > 0:
                position_size = max_shares
                position_type = 'long'
                commission_cost = position_size * trade_price * config.commission_long
                cash -= (position_size * trade_price + commission_cost)
                position_cost_basis = trade_price
                commission_paid = commission_cost
                num_trades += 1
                action_taken = "BUY"
            else:
                action_taken = "INSUFFICIENT_CASH"
                
        elif sell_signal:
            short_position_value = cash * config.short_entry_pct_cash
            max_shares = int(short_position_value / (trade_price * (1 + config.commission_short)))
            
            if max_shares > 0:
                position_size = -max_shares
                position_type = 'short'
                commission_cost = abs(position_size) * trade_price * config.commission_short
                cash += (abs(position_size) * trade_price - commission_cost)
                position_cost_basis = trade_price
                commission_paid = commission_cost
                num_trades += 1
                action_taken = "SHORT"
            else:
                action_taken = "INSUFFICIENT_CASH"
    
    elif position_type == 'long':
        if sell_signal:
            if buy_signal:
                action_taken = "HOLD_CONFLICTING_SIGNAL"
            else:
                # Sell and Short
                commission_cost = position_size * trade_price * config.commission_long
                cash += (position_size * trade_price - commission_cost)
                commission_paid = commission_cost
                position_size = 0
                position_type = 'none'
                
                # Then enter short
                short_position_value = cash * config.short_entry_pct_cash
                max_shares = int(short_position_value / (trade_price * (1 + config.commission_short)))
                
                if max_shares > 0:
                    position_size = -max_shares
                    position_type = 'short'
                    commission_cost = abs(position_size) * trade_price * config.commission_short
                    cash += (abs(position_size) * trade_price - commission_cost)
                    commission_paid += commission_cost
                    position_cost_basis = trade_price
                    num_trades += 2
                    action_taken = "SELL AND SHORT"
                else:
                    action_taken = "SELL"
                    num_trades += 1
    
    elif position_type == 'short':
        if buy_signal:
            if sell_signal:
                action_taken = "HOLD_CONFLICTING_SIGNAL"
            else:
                # Cover and Buy
                commission_cost = abs(position_size) * trade_price * config.commission_short
                cash -= (abs(position_size) * trade_price + commission_cost)
                commission_paid = commission_cost
                position_size = 0
                position_type = 'none'
                
                # Then enter long
                max_shares = int((cash * config.long_entry_pct_cash) / (trade_price * (1 + config.commission_long)))
                
                if max_shares > 0:
                    position_size = max_shares
                    position_type = 'long'
                    commission_cost = position_size * trade_price * config.commission_long
                    cash -= (position_size * trade_price + commission_cost)
                    commission_paid += commission_cost
                    position_cost_basis = trade_price
                    num_trades += 2
                    action_taken = "COVER AND BUY"
                else:
                    action_taken = "COVER"
                    num_trades += 1

    return cash, position_size, position_type, position_cost_basis, commission_paid, action_taken, num_trades


def _build_cross_strategy_name(
    short_window_indicator: str,
    long_window_indicator: str,
    trading_type: str,
    day1_position: str
) -> str:
    """Build the strategy name string."""
    name = f"Cross Trade ({short_window_indicator}/{long_window_indicator})"
    if trading_type in ['short', 'mixed']:
        name += ' [Shorts Allowed]'
    if day1_position != 'none':
        name += f' [Day1 {day1_position.capitalize()}]'
    return name


def _get_empty_cross_results(
    config: BacktestConfig,
    short_window_indicator: str,
    long_window_indicator: str,
    trading_type: str,
    day1_position: str
) -> dict:
    """Returns default results for empty DataFrame scenarios."""
    return {
        "strategy": _build_cross_strategy_name(
            short_window_indicator, long_window_indicator, trading_type, day1_position
        ),
        "short_window_indicator": short_window_indicator,
        "long_window_indicator": long_window_indicator,
        "initial_cash": config.initial_cash,
        "final_value": config.initial_cash,
        "total_return_pct": 0.0,
        "num_trades": 0,
    }


