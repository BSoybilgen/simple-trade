"""
Band Trade backtesting functions.

This module provides function-based implementations for band trade strategies,
replacing the class-based BandTradeBacktester approach.
"""
import pandas as pd
from typing import Optional

from .config import BacktestConfig
from .metrics import compute_benchmark_return, calculate_performance_metrics


def run_band_trade(
    data: pd.DataFrame,
    indicator_col: str,
    upper_band_col: str,
    lower_band_col: str,
    config: Optional[BacktestConfig] = None,
    price_col: str = 'Close',
    trading_type: str = 'long',
    strategy_type: int = 1,
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
    Runs a backtest for a band trade strategy.

    Two strategy types are available via the `strategy_type` parameter:
    
    Strategy Type 1 (Mean Reversion - Default):
    - 'long': Buys when indicator crosses below lower band, sells when above upper band.
    - 'short': Shorts when indicator crosses above upper band, covers when below lower band.
    - 'mixed': Allows both long and short positions with seamless transitions.
        
    Strategy Type 2 (Breakout):
    - 'long': Buys when indicator crosses above upper band, sells when below lower band.
    - 'short': Shorts when indicator crosses below lower band, covers when above upper band.
    - 'mixed': Allows both long and short positions with seamless transitions.

    Args:
        data: DataFrame containing price data and indicator/band columns. Must have a DatetimeIndex.
        indicator_col: Column name of the indicator (e.g., 'RSI', 'Close').
        upper_band_col: Column name of the upper band (e.g., 'BB_Upper', 'RSI_Upper').
        lower_band_col: Column name of the lower band (e.g., 'BB_Lower', 'RSI_Lower').
        config: BacktestConfig object with all configuration parameters. If provided,
               individual parameters below are ignored.
        price_col: Column name to use for trade execution prices.
        trading_type: Defines the trading behavior ('long', 'short', 'mixed').
        strategy_type: Defines the band trade logic (1: mean_reversion, 2: breakout).
        day1_position: Specifies whether to take a position on day 1 ('none', 'long', 'short').
        
        # Legacy parameters (used if config is None):
        initial_cash: Starting cash balance for the backtest.
        commission_long: Commission rate for long trades.
        commission_short: Commission rate for short trades.
        short_borrow_fee_inc_rate: Time-based fee rate for holding short positions.
        long_borrow_fee_inc_rate: Time-based fee rate for holding long positions.
        long_entry_pct_cash: Pct of available cash to use for long entries (0.0 to 1.0).
        short_entry_pct_cash: Pct of available cash defining the value of short entries.
        risk_free_rate: Risk-free rate for Sharpe and Sortino ratios.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary with backtest summary results (final value, return, trades).
            - pd.DataFrame: DataFrame tracking daily portfolio evolution.
            
    Example:
        >>> config = BacktestConfig(initial_cash=50000, commission_long=0.002)
        >>> results, portfolio = run_band_trade(
        ...     data, 'RSI', 'RSI_Upper', 'RSI_Lower',
        ...     config=config, trading_type='long', strategy_type=1
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
    _validate_band_trade_inputs(
        data, indicator_col, upper_band_col, lower_band_col,
        price_col, trading_type, strategy_type, day1_position, config
    )

    # --- Signal Generation ---
    df = _generate_band_signals(data.copy(), indicator_col, upper_band_col, lower_band_col, strategy_type)
    df.dropna(inplace=True)

    # Check if DataFrame is empty after generating signals
    if df.empty:
        return _get_empty_band_results(
            config, indicator_col, upper_band_col, lower_band_col,
            strategy_type, trading_type, day1_position
        ), pd.DataFrame()

    # --- Run Backtest ---
    portfolio_log, end_state = _run_band_backtest(
        signal_df=df,
        config=config,
        price_col=price_col,
        trading_type=trading_type,
        day1_position=day1_position,
    )

    # --- Prepare and Return Results ---
    if not portfolio_log:
        final_df = pd.DataFrame({
            'PositionSize': [0],
            'PositionValue': [0.0],
            'Cash': [config.initial_cash],
            'PortfolioValue': [config.initial_cash],
            'Close': [df[price_col].iloc[-1]]
        }, index=df.index[[-1]])
    else:
        final_df = end_state

    results, portfolio_df = _prepare_band_results(
        portfolio_log=portfolio_log,
        final_df=final_df,
        data=df,
        config=config,
        indicator_col=indicator_col,
        upper_band_col=upper_band_col,
        lower_band_col=lower_band_col,
        strategy_type=strategy_type,
        trading_type=trading_type,
        day1_position=day1_position,
    )

    return results, portfolio_df


def _validate_band_trade_inputs(
    data: pd.DataFrame,
    indicator_col: str,
    upper_band_col: str,
    lower_band_col: str,
    price_col: str,
    trading_type: str,
    strategy_type: int,
    day1_position: str,
    config: BacktestConfig
) -> None:
    """Validates all inputs for band trade backtest."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    # Check for required columns
    required_cols = [price_col, indicator_col, upper_band_col, lower_band_col]
    for col in required_cols:
        if col not in data.columns:
            if col == indicator_col:
                raise ValueError(f"Indicator column '{col}' not found in DataFrame.")
            elif col == upper_band_col:
                raise ValueError(f"Upper band column '{col}' not found in DataFrame.")
            elif col == lower_band_col:
                raise ValueError(f"Lower band column '{col}' not found in DataFrame.")
            else:
                raise ValueError(f"Price column '{col}' not found in DataFrame.")

    valid_trading_types = ['long', 'short', 'mixed']
    if trading_type not in valid_trading_types:
        raise ValueError(f"Invalid trading_type '{trading_type}'. Must be one of {valid_trading_types}")
        
    if strategy_type not in [1, 2]:
        raise ValueError(f"Invalid strategy_type: {strategy_type}. Must be 1 (mean reversion) or 2 (breakout).")
        
    valid_day1_positions = ['none', 'long', 'short']
    if day1_position not in valid_day1_positions:
        raise ValueError(f"Invalid day1_position '{day1_position}'. Must be one of {valid_day1_positions}")
    
    # Check compatibility between day1_position and trading_type
    if day1_position == 'long' and trading_type == 'short':
        raise ValueError("Cannot use day1_position='long' with trading_type='short'")
    if day1_position == 'short' and trading_type == 'long':
        raise ValueError("Cannot use day1_position='short' with trading_type='long'")


def _generate_band_signals(
    df: pd.DataFrame,
    indicator_col: str,
    upper_band_col: str,
    lower_band_col: str,
    strategy_type: int
) -> pd.DataFrame:
    """Generates buy and sell signals based on indicator crossing bands."""
    prev_indicator = df[indicator_col].shift(1)
    prev_upper = df[upper_band_col].shift(1)
    prev_lower = df[lower_band_col].shift(1)
    prev_prev_indicator = df[indicator_col].shift(2)
    prev_prev_upper = df[upper_band_col].shift(2)
    prev_prev_lower = df[lower_band_col].shift(2)

    if strategy_type == 1:  # Mean Reversion Strategy
        df['buy_signal'] = (prev_indicator < prev_lower) & (prev_prev_indicator >= prev_prev_lower)
        df['sell_signal'] = (prev_indicator > prev_upper) & (prev_prev_indicator <= prev_prev_upper)
    else:  # strategy_type == 2: Breakout Strategy
        df['buy_signal'] = (prev_indicator > prev_upper) & (prev_prev_indicator <= prev_prev_upper)
        df['sell_signal'] = (prev_indicator < prev_lower) & (prev_prev_indicator >= prev_prev_lower)

    df['buy_signal'] = df['buy_signal'].astype(bool)
    df['sell_signal'] = df['sell_signal'].astype(bool)

    return df


def _run_band_backtest(
    signal_df: pd.DataFrame,
    config: BacktestConfig,
    price_col: str,
    trading_type: str,
    day1_position: str,
) -> tuple:
    """Runs the backtest simulation based on the generated signals."""
    portfolio_log = []
    cash = config.initial_cash
    position_size = 0
    position_value = 0.0
    position_type = 'none'
    commission_paid = 0.0
    
    # Handle day1_position if not 'none'
    if day1_position != 'none' and not signal_df.empty:
        first_price = signal_df[price_col].iloc[0]
        
        if day1_position == 'long':
            shares_to_buy = int((cash * config.long_entry_pct_cash) / first_price)
            if shares_to_buy > 0:
                commission = shares_to_buy * first_price * config.commission_long
                cash -= (shares_to_buy * first_price + commission)
                position_size = shares_to_buy
                position_value = shares_to_buy * first_price
                position_type = 'long'
                commission_paid += commission
        
        elif day1_position == 'short':
            shares_to_short = int((cash * config.short_entry_pct_cash) / first_price)
            if shares_to_short > 0:
                commission = shares_to_short * first_price * config.commission_short
                cash += (shares_to_short * first_price - commission)
                position_size = -shares_to_short
                position_value = abs(position_size) * first_price
                position_type = 'short'
                commission_paid += commission
    
    # Process each day's signals
    for i, (date, row) in enumerate(signal_df.iterrows()):
        current_price = row[price_col]
        buy_signal = row.get('buy_signal', False)
        sell_signal = row.get('sell_signal', False)
        
        start_of_day_position_type = position_type
        start_of_day_position_value = position_value
        
        # Apply borrow fees
        short_fee = 0.0
        long_fee = 0.0
        
        if start_of_day_position_type == 'short':
            short_fee = start_of_day_position_value * config.short_borrow_fee_inc_rate
            cash -= short_fee
        elif start_of_day_position_type == 'long':
            long_fee = start_of_day_position_value * config.long_borrow_fee_inc_rate
            cash -= long_fee
        
        # Update position value
        if position_type == 'long':
            position_value = position_size * current_price
        elif position_type == 'short':
            position_value = abs(position_size) * current_price
        else:
            position_value = 0.0
        
        action = 'HOLD'
        
        # Process signals based on trading_type
        if trading_type == 'long':
            cash, position_size, position_value, position_type, commission_paid, action = _execute_long_only(
                buy_signal, sell_signal, position_type, position_size, position_value,
                cash, current_price, config, commission_paid
            )
        
        elif trading_type == 'short':
            cash, position_size, position_value, position_type, commission_paid, action = _execute_short_only(
                buy_signal, sell_signal, position_type, position_size, position_value,
                cash, current_price, config, commission_paid
            )
        
        elif trading_type == 'mixed':
            cash, position_size, position_value, position_type, commission_paid, action = _execute_mixed(
                buy_signal, sell_signal, position_type, position_size, position_value,
                cash, current_price, config, commission_paid
            )
        
        # Calculate portfolio value
        portfolio_value = cash
        if position_type == 'long':
            portfolio_value += position_value
        elif position_type == 'short':
            portfolio_value -= position_value
        
        snapshot = {
            'Date': date,
            'Price': current_price,
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
        }
        
        portfolio_log.append(snapshot)
    
    # Create end state DataFrame
    if portfolio_log:
        end_state = pd.DataFrame(portfolio_log)
        end_state.set_index('Date', inplace=True)
    else:
        end_state = pd.DataFrame(columns=['Price', 'Close', 'Cash', 'PositionSize', 'PositionValue', 
                                         'PositionType', 'PortfolioValue', 'CommissionPaid', 'ShortFee', 'LongFee',
                                         'BuySignal', 'SellSignal', 'Action'])
    
    return portfolio_log, end_state


def _execute_long_only(
    buy_signal: bool,
    sell_signal: bool,
    position_type: str,
    position_size: int,
    position_value: float,
    cash: float,
    current_price: float,
    config: BacktestConfig,
    commission_paid: float
) -> tuple:
    """Execute long-only trading logic."""
    action = 'HOLD'
    
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
        if buy_signal:
            action = 'HOLD_CONFLICTING_SIGNAL'
        else:
            commission = position_value * config.commission_long
            cash += (position_value - commission)
            position_size = 0
            position_value = 0.0
            position_type = 'none'
            commission_paid += commission
            action = 'SELL'
    
    return cash, position_size, position_value, position_type, commission_paid, action


def _execute_short_only(
    buy_signal: bool,
    sell_signal: bool,
    position_type: str,
    position_size: int,
    position_value: float,
    cash: float,
    current_price: float,
    config: BacktestConfig,
    commission_paid: float
) -> tuple:
    """Execute short-only trading logic."""
    action = 'HOLD'
    
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
        if sell_signal:
            action = 'HOLD_CONFLICTING_SIGNAL'
        else:
            commission = position_value * config.commission_short
            cash -= (position_value + commission)
            position_size = 0
            position_value = 0.0
            position_type = 'none'
            commission_paid += commission
            action = 'COVER'
    
    return cash, position_size, position_value, position_type, commission_paid, action


def _execute_mixed(
    buy_signal: bool,
    sell_signal: bool,
    position_type: str,
    position_size: int,
    position_value: float,
    cash: float,
    current_price: float,
    config: BacktestConfig,
    commission_paid: float
) -> tuple:
    """Execute mixed (long and short) trading logic."""
    action = 'HOLD'
    
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
                
                if prev_position_type == 'short':
                    action = 'COVER AND BUY'
                else:
                    action = 'BUY'
            else:
                if prev_position_type == 'short':
                    action = 'COVER'
    
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
                
                if prev_position_type == 'long':
                    action = 'SELL AND SHORT'
                else:
                    action = 'SHORT'
            else:
                if prev_position_type == 'long':
                    action = 'SELL'
    
    return cash, position_size, position_value, position_type, commission_paid, action


def _prepare_band_results(
    portfolio_log: list,
    final_df: pd.DataFrame,
    data: pd.DataFrame,
    config: BacktestConfig,
    indicator_col: str,
    upper_band_col: str,
    lower_band_col: str,
    strategy_type: int,
    trading_type: str,
    day1_position: str,
) -> tuple:
    """Prepares the final results dictionary and portfolio DataFrame."""
    portfolio_df = pd.DataFrame(portfolio_log).set_index('Date')
    if 'Cash' in portfolio_df.columns:
        portfolio_df = portfolio_df.drop(columns=['Cash'])
    
    # Calculate benchmark and performance metrics
    data_traded = data.loc[portfolio_df.index]
    benchmark_results = compute_benchmark_return(data_traded, config.initial_cash, config.commission_long, price_col='Close')
    improved_results = calculate_performance_metrics(portfolio_df, config.risk_free_rate)

    # Calculate total fees
    total_short_fees = portfolio_df['ShortFee'].sum() if 'ShortFee' in portfolio_df.columns else 0
    total_long_fees = portfolio_df['LongFee'].sum() if 'LongFee' in portfolio_df.columns else 0
    total_fees = total_short_fees + total_long_fees

    # Build strategy name
    strategy_name = f"Band Trade ({indicator_col} vs {lower_band_col}/{upper_band_col} - {'Mean Reversion' if strategy_type == 1 else 'Breakout'})"
    if trading_type in ['short', 'mixed']:
        strategy_name += ' [Shorts Allowed]'
    if day1_position != 'none':
        strategy_name += f' [Day1 {day1_position.capitalize()}]'

    # Count trades
    num_trades = 0
    if 'Action' in portfolio_df.columns:
        action_counts = portfolio_df['Action'].value_counts()
        for action in ['BUY', 'SELL', 'SHORT', 'COVER']:
            num_trades += action_counts.get(action, 0)

    results = {
        "strategy": strategy_name,
        "indicator_col": indicator_col,
        "upper_band_col": upper_band_col,
        "lower_band_col": lower_band_col,
        "strategy_type": strategy_type,
        "initial_cash": config.initial_cash,
        "final_value": round(portfolio_df['PortfolioValue'].iloc[-1], 2),
        "total_return_pct": round(((portfolio_df['PortfolioValue'].iloc[-1] - config.initial_cash) / config.initial_cash) * 100, 2),
        "num_trades": num_trades,
        "total_short_fees": round(total_short_fees, 2),
        "total_long_fees": round(total_long_fees, 2),
        "total_borrow_fees": round(total_fees, 2),
    }
    results.update(benchmark_results)
    results.update(improved_results)

    return results, portfolio_df


def _get_empty_band_results(
    config: BacktestConfig,
    indicator_col: str,
    upper_band_col: str,
    lower_band_col: str,
    strategy_type: int,
    trading_type: str,
    day1_position: str
) -> dict:
    """Returns default results for empty DataFrame scenarios."""
    strategy_name = f"Band Trade ({indicator_col} vs {lower_band_col}/{upper_band_col} - {'Mean Reversion' if strategy_type == 1 else 'Breakout'})"
    if trading_type in ['short', 'mixed']:
        strategy_name += ' [Shorts Allowed]'
    if day1_position != 'none':
        strategy_name += f' [Day1 {day1_position.capitalize()}]'
    
    return {
        "error": "DataFrame became empty after signal generation/dropna, cannot run backtest.",
        "strategy": strategy_name,
        "indicator_col": indicator_col,
        "upper_band_col": upper_band_col,
        "lower_band_col": lower_band_col,
        "strategy_type": strategy_type,
        "start_date": None,
        "end_date": None,
        "duration_days": 0,
        "initial_cash": config.initial_cash,
        "final_value": config.initial_cash,
        "total_return_pct": 0.0,
        "num_trades": 0,
    }


