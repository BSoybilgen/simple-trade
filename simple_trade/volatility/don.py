import pandas as pd


def don(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Donchian Channels (don), a volatility indicator that plots the highest high and lowest low
    over a specified period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Donchian Channels DataFrame and a list of column names.

    Calculation Steps:
    1. Upper Band:
       Highest High over the specified window.
    2. Lower Band:
       Lowest Low over the specified window.
    3. Middle Band:
       (Upper Band + Lower Band) / 2

    Interpretation:
    - Price breaking Upper Band: Bullish breakout.
    - Price breaking Lower Band: Bearish breakout.
    - Middle Band direction: Indicates overall trend.

    Use Cases:
    - Breakout trading: Basis of the "Turtle Trading" system.
    - Trend identification: Middle band slope.
    - Support and resistance: Dynamic S/R levels.
    - Volatility measurement: Width between bands.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    
    high = df[high_col]
    low = df[low_col]
    
    # Calculate the upper and lower bands
    upper_band = high.rolling(window=window).max()
    lower_band = low.rolling(window=window).min()
    
    # Calculate the middle band
    middle_band = (upper_band + lower_band) / 2
    
    # Prepare the result DataFrame
    result = pd.DataFrame({
        f'DON_Upper_{window}': upper_band,
        f'DON_Middle_{window}': middle_band,
        f'DON_Lower_{window}': lower_band
    }, index=high.index)
    
    columns_list = list(result.columns)
    return result, columns_list


def strategy_don(
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
    don (Donchian Channels) - Cross Trade Strategy with Middle Line
    
    LOGIC: Buy when Close crosses above the middle band (bullish),
           sell when Close crosses below the middle band (bearish).
    WHY: The middle band represents the equilibrium between highest high and lowest low.
         Price above middle = bullish momentum, price below = bearish momentum.
    BEST MARKETS: Trending markets. Commodities, forex, futures.
                  Good for trend-following strategies.
    TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20)
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
    
    window = int(parameters.get('window', 20))
    price_col = 'Close'
    middle_col = f'DON_Middle_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='don',
        parameters={"window": window},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=price_col,
        long_window_indicator=middle_col,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['Close', f'DON_Upper_{window}', 
                              f'DON_Middle_{window}', f'DON_Lower_{window}']
    
    return results, portfolio, indicator_cols_to_plot, data
