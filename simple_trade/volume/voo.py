import pandas as pd

def voo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Oscillator (voo), which displays the difference between
    two moving averages of volume. It measures volume trends using points (absolute difference)
    rather than percentage.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_period (int): The fast SMA period. Default is 5.
            - slow_period (int): The slow SMA period. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VO series and a list of column names.

    The Volume Oscillator is calculated as follows:

    1. Calculate Fast SMA:
       Fast SMA = SMA(Volume, fast_period)

    2. Calculate Slow SMA:
       Slow SMA = SMA(Volume, slow_period)

    3. Calculate VO:
       VOO = Fast SMA - Slow SMA

    Interpretation:
    - Positive VOO: Volume is increasing (Short-term volume > Long-term volume).
    - Negative VOO: Volume is decreasing (Short-term volume < Long-term volume).
    - Trend Strength: Rising VOO indicates increasing market participation.

    Use Cases:
    - Volume Trend Analysis: Confirming the strength of price moves.
    - Breakout Confirmation: VOO spikes often accompany valid breakouts.
    - Exhaustion: Extreme VOO readings may signal trend exhaustion.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    fast_window_param = parameters.get('fast_window')
    fast_period_param = parameters.get('fast_period')
    if fast_window_param is None and fast_period_param is not None:
        fast_window_param = fast_period_param
    elif fast_window_param is not None and fast_period_param is not None:
        if int(fast_window_param) != int(fast_period_param):
            raise ValueError("Provide either 'fast_window' or 'fast_period' (aliases) with the same value if both are set.")

    slow_window_param = parameters.get('slow_window')
    slow_period_param = parameters.get('slow_period')
    if slow_window_param is None and slow_period_param is not None:
        slow_window_param = slow_period_param
    elif slow_window_param is not None and slow_period_param is not None:
        if int(slow_window_param) != int(slow_period_param):
            raise ValueError("Provide either 'slow_window' or 'slow_period' (aliases) with the same value if both are set.")

    fast_period = int(fast_window_param if fast_window_param is not None else 5)
    slow_period = int(slow_window_param if slow_window_param is not None else 10)
    volume_col = columns.get('volume_col', 'Volume')
    
    volume = df[volume_col]
    
    # Calculate SMAs
    fast_sma = volume.rolling(window=fast_period).mean()
    slow_sma = volume.rolling(window=slow_period).mean()
    
    # Calculate VO
    vo_values = fast_sma - slow_sma
    
    vo_values.name = f'VOO_{fast_period}_{slow_period}'
    columns_list = [vo_values.name]
    return vo_values, columns_list


def strategy_voo(
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
    voo (Volume Oscillator) - Zero Line Cross Strategy
    
    LOGIC: Buy when voo crosses above zero (volume increasing),
           sell when voo crosses below zero (volume decreasing).
    WHY: voo shows difference between fast and slow volume MAs.
         Positive VO indicates increasing market participation.
    BEST MARKETS: Stocks, ETFs. Good for breakout confirmation.
    TIMEFRAME: Daily charts. 5/10 periods is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'fast_period' (default 5), 'slow_period' (default 10)
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
    
    fast_window_param = parameters.get('fast_window')
    fast_period_param = parameters.get('fast_period')
    if fast_window_param is None and fast_period_param is not None:
        fast_window_param = fast_period_param
    elif fast_window_param is not None and fast_period_param is not None:
        if int(fast_window_param) != int(fast_period_param):
            raise ValueError("Provide either 'fast_window' or 'fast_period' (aliases) with the same value if both are set.")

    slow_window_param = parameters.get('slow_window')
    slow_period_param = parameters.get('slow_period')
    if slow_window_param is None and slow_period_param is not None:
        slow_window_param = slow_period_param
    elif slow_window_param is not None and slow_period_param is not None:
        if int(slow_window_param) != int(slow_period_param):
            raise ValueError("Provide either 'slow_window' or 'slow_period' (aliases) with the same value if both are set.")

    fast_period = int(fast_window_param if fast_window_param is not None else 5)
    slow_period = int(slow_window_param if slow_window_param is not None else 10)
    price_col = 'Close'
    indicator_col = f'VOO_{fast_period}_{slow_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='voo',
        parameters={"fast_period": fast_period, "slow_period": slow_period},
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
