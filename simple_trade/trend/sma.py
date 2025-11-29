import pandas as pd

def sma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Simple Moving Average (SMA) of a series.
    The SMA is a moving average that is calculated by taking the arithmetic
    mean of a given set of values over a specified period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The window size for the SMA calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the SMA series and a list of column names.

    The Simple Moving Average is calculated as follows:

    1. Sum the values over the specified window.
    2. Divide the sum by the window size.
       SMA = Sum(Price, window) / window

    Interpretation:
    - Rising SMA: Uptrend.
    - Falling SMA: Downtrend.
    - Price > SMA: Bullish.
    - Price < SMA: Bearish.

    Use Cases:
    - Identifying trends: The SMA can be used to identify the direction of a
      price trend.
    - Smoothing price data: The SMA can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The SMA can be used in crossover systems
      to generate buy and sell signals (e.g. Price crosses SMA).
    - Support/Resistance: Often acts as dynamic support or resistance.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]
    series =series.rolling(window=window).mean()
    series.name = f'SMA_{window}'

    columns_list = [series.name]

    return series, columns_list


def strategy_sma(
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
    SMA (Simple Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast SMA crosses above slow SMA, sell when crosses below.
    WHY: SMA is the simplest trend indicator. Crossovers signal trend changes.
         Classic strategy used for decades across all markets.
    BEST MARKETS: Trending markets. Stocks, forex, commodities.
                  Golden Cross (50/200) is famous for long-term trends.
    TIMEFRAME: All timeframes. Common pairs: 10/20, 20/50, 50/200.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 20), 'long_window' (default 50)
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
    
    short_window = int(parameters.get('short_window', 20))
    long_window = int(parameters.get('long_window', 50))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'SMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='sma',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'SMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='sma',
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