import pandas as pd

def ema(data: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Exponential Moving Average (EMA) of a series.
    The EMA is a type of moving average that gives more weight to recent
    prices, making it more responsive to new information than the SMA.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the EMA. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the EMA series and a list of column names.

    Calculation Steps:
    1. Calculate the Multiplier (k):
       k = 2 / (window + 1)

    2. Calculate EMA:
       EMA(today) = (Price(today) * k) + (EMA(yesterday) * (1 - k))

    Interpretation:
    - Rising EMA: Uptrend.
    - Falling EMA: Downtrend.
    - Price > EMA: Bullish.
    - Price < EMA: Bearish.

    Use Cases:
    - Identifying trends: The EMA can be used to identify the direction of a
      price trend.
    - Smoothing price data: The EMA can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The EMA can be used in crossover systems
      to generate buy and sell signals.
    - Reacting quickly to price changes: The EMA's responsiveness makes it
      suitable for identifying entry and exit points in fast-moving markets.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = data[close_col]
    series = series.ewm(span=window, adjust=False).mean()
    series.name = f'EMA_{window}'
    
    # Return as tuple with column names for consistency with other indicators
    columns_list = [series.name]
    return series, columns_list


def strategy_ema(
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
    EMA (Exponential Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast EMA crosses above slow EMA, sell when crosses below.
    WHY: EMA gives more weight to recent prices, making it more responsive than SMA.
         Classic trend-following strategy used across all markets.
    BEST MARKETS: Trending markets. Stocks, forex, crypto, commodities.
                  One of the most widely-used crossover strategies.
    TIMEFRAME: All timeframes. Common pairs: 9/21, 12/26, 50/200.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 12), 'long_window' (default 26)
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
    
    short_window = int(parameters.get('short_window', 12))
    long_window = int(parameters.get('long_window', 26))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'EMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='ema',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'EMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='ema',
        parameters={"window": long_window},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data