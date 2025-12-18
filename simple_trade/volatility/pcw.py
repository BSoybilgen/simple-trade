import pandas as pd


def pcw(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Price Channel Width (pcw), which measures the width of a price channel
    (Donchian-style) as a percentage of the closing price. It provides a simple measure
    of volatility based on the high-low range over a period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for channel calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the PCW series and a list of column names.

    Calculation Steps:
    1. Find Highest High and Lowest Low over the period.
    2. Calculate Channel Width as Percentage:
       PCW = ((Highest High - Lowest Low) / Close) * 100

    Interpretation:
    - Low PCW: Narrow price channel, low volatility, consolidation.
    - High PCW: Wide price channel, high volatility, trending.
    - Increasing PCW: Expanding volatility.
    - Decreasing PCW: Contracting volatility.

    Use Cases:
    - Simple volatility measurement.
    - Breakout identification (low PCW precedes breakouts).
    - Channel-based trading strategies.
    - Volatility comparison across assets.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 20))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate highest high and lowest low over period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Calculate channel width as percentage of close
    pcw_values = ((highest_high - lowest_low) / close) * 100
    
    pcw_values.name = f'PCW_{period}'
    columns_list = [pcw_values.name]
    return pcw_values, columns_list


def strategy_pcw(
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
    pcw (Price Channel Width) - Volatility Threshold Strategy
    
    LOGIC: Buy when pcw drops below lower threshold (narrow channel/consolidation),
           sell when rises above upper threshold (wide channel/trending).
    WHY: pcw measures Donchian-style channel width as percentage of close.
         Low PCW indicates consolidation, often preceding breakouts.
    BEST MARKETS: All markets. Good for breakout identification.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 20), 'upper' (default 15),
                    'lower' (default 5)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_band_trade_strategies import run_band_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 20))
    upper = float(parameters.get('upper', 15))
    lower = float(parameters.get('lower', 5))
    price_col = 'Close'
    indicator_col = f'PCW_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='pcw',
        parameters={"period": period},
        figure=False
    )
    
    data['upper'] = upper
    data['lower'] = lower
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
