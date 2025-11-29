import pandas as pd


def acb(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Acceleration Bands, a volatility-based indicator that creates dynamic
    upper and lower bands around price using a percentage of the high and low prices.
    Unlike Bollinger Bands which use standard deviation, Acceleration Bands use a
    fixed percentage multiplier applied to highs and lows.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for the moving average. Default is 20.
            - factor (float): The percentage factor for band width. Default is 0.001 (0.1%).
                             Higher values create wider bands.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame with upper, middle, and lower bands,
               and a list of column names.

    The Acceleration Bands are calculated as follows:
    1. Calculate the middle band (Simple Moving Average):
       Middle Band = SMA(Close, period)
    2. Calculate the upper band:
       Upper Band = SMA(High * (1 + factor), period)
       This applies the factor to each high price before averaging, creating
       an upper band that "accelerates" away from price during uptrends.
    3. Calculate the lower band:
       Lower Band = SMA(Low * (1 - factor), period)
       This applies the factor to each low price before averaging, creating
       a lower band that "accelerates" away from price during downtrends.

    Interpretation:
    - Price above upper band: Strong uptrend, bullish momentum
    - Price below lower band: Strong downtrend, bearish momentum
    - Price between bands: Consolidation, indecision, or transition
    - Narrowing bands: Decreasing volatility, potential breakout setup
    - Widening bands: Increasing volatility, strong trending conditions

    Use Cases:
    - Trend identification: Price consistently above/below bands indicates strong trends.
    - Breakout trading: Price breaking above upper band or below lower band.
    - Support/resistance: The bands act as dynamic support and resistance levels.
    - Overbought/oversold: Extreme moves beyond the bands may indicate overextension.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 20))
    factor = float(parameters.get('factor', 0.001))  # 0.1% default
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate middle band (SMA of close)
    middle_band = close.rolling(window=period).mean()
    
    # Calculate upper band (SMA of high * (1 + factor))
    upper_band = (high * (1 + factor)).rolling(window=period).mean()
    
    # Calculate lower band (SMA of low * (1 - factor))
    lower_band = (low * (1 - factor)).rolling(window=period).mean()
    
    # Create result DataFrame
    result = pd.DataFrame(index=close.index)
    
    # Format factor for naming (convert to percentage)
    factor_pct = factor * 100
    
    upper_name = f'ACB_Upper_{period}_{factor_pct:.2f}'
    middle_name = f'ACB_Middle_{period}'
    lower_name = f'ACB_Lower_{period}_{factor_pct:.2f}'
    
    result[upper_name] = upper_band
    result[middle_name] = middle_band
    result[lower_name] = lower_band
    
    columns_list = [upper_name, middle_name, lower_name]
    return result, columns_list


def strategy_acb(
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
    ACB (Acceleration Bands) - Band Breakout Strategy
    
    LOGIC: Buy when price breaks above upper band (bullish breakout),
           sell when below lower band (bearish breakout).
    WHY: Acceleration Bands use percentage-based bands around price.
         Price breaking above upper band indicates strong uptrend momentum.
    BEST MARKETS: Trending markets. Stocks, forex, futures. Good for breakout
                  trading. Avoid in choppy, range-bound markets.
    TIMEFRAME: Daily or 4-hour charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 20), 'factor' (default 0.001)
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
    
    period = int(parameters.get('period', 20))
    factor = float(parameters.get('factor', 0.001))
    factor_pct = factor * 100
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='acb',
        parameters={"period": period, "factor": factor},
        figure=False
    )
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col='Close',
        upper_band_col=f'ACB_Upper_{period}_{factor_pct:.2f}',
        lower_band_col=f'ACB_Lower_{period}_{factor_pct:.2f}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['Close', f'ACB_Upper_{period}_{factor_pct:.2f}', 
                              f'ACB_Middle_{period}', f'ACB_Lower_{period}_{factor_pct:.2f}']
    
    return results, portfolio, indicator_cols_to_plot, data
