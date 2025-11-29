import pandas as pd
import numpy as np


def vro(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Rate of Change (VROC), a momentum indicator that measures
    the rate of change in volume over a specified period. It highlights significant
    volume increases or decreases.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for VROC calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VROC series and a list of column names.

    The Volume Rate of Change is calculated as follows:

    1. Identify Past Volume:
       Past Volume = Volume(n periods ago)

    2. Calculate VROC:
       VROC = ((Current Volume - Past Volume) / Past Volume) * 100

    Interpretation:
    - Positive VROC: Volume is increasing.
    - Negative VROC: Volume is decreasing.
    - High VROC: High trading activity/volatility.
    - Low VROC: Low trading activity/consolidation.

    Use Cases:
    - Breakout Validation: Breakouts should be accompanied by a surge in VROC.
    - Trend Strength: Rising VROC confirms trend participation.
    - Reversal Warning: Divergence between Price and VROC.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = parameters.get('period', 14)
    volume_col = columns.get('volume_col', 'Volume')
    
    volume = df[volume_col]
    
    # Calculate Volume n periods ago
    volume_n_periods_ago = volume.shift(period)
    
    # Calculate VROC (handle division by zero)
    vroc_values = ((volume - volume_n_periods_ago) / volume_n_periods_ago.replace(0, np.nan)) * 100
    
    vroc_values.name = f'VROC_{period}'
    columns_list = [vroc_values.name]
    return vroc_values, columns_list


def strategy_vro(
    data: pd.DataFrame,
    parameters: dict = None,
    config = None,
    trading_type: str = 'long',
    day1_position: str = 'none',
    risk_free_rate: float = 0.0,
    long_entry_pct_cash: float = 1.0,
    short_entry_pct_cash: float = 1.0
) -> tuple:
    """
    VRO (Volume Rate of Change) - Zero Line Cross Strategy
    
    LOGIC: Buy when VROC crosses above zero (volume increasing),
           sell when VROC crosses below zero (volume decreasing).
    WHY: VROC measures rate of change in volume. Positive VROC indicates
         increasing volume, negative indicates decreasing volume.
    BEST MARKETS: Stocks, ETFs. Good for breakout validation.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_cross_trade_strategies import run_cross_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    period = int(parameters.get('period', 14))
    price_col = 'Close'
    indicator_col = f'VROC_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='vro',
        parameters={"period": period},
        figure=False
    )
    
    # Create zero line for crossover
    data['zero'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'zero']
    
    return results, portfolio, indicator_cols_to_plot, data
