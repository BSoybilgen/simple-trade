import pandas as pd


def soa(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Smoothed Moving Average (SmMA) of a series.
    The Smoothed Moving Average is similar to Wilder's moving average and applies
    heavier smoothing than a simple moving average. Each new value is a blend of
    the previous SmMA and the current price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the SmMA calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the SmMA series and a list of column names.

    The Smoothed Moving Average is calculated as follows:

    1. Calculate the first SmMA (usually SMA of the first window).

    2. Calculate subsequent SmMAs:
       SmMA(i) = (SmMA(i-1) * (n - 1) + Price(i)) / n
       where n is the window size.

    Interpretation:
    - The SmMA gives recent prices an equal weight to historic prices.
    - It does not respond to price changes as quickly as EMA.
    - Useful for identifying long-term trends.

    Use Cases:
    - Trend Identification: Filtering out short-term noise to see the long-term trend.
    - Support/Resistance: Often acts as a significant dynamic support or resistance level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]
    smoothed = series.ewm(alpha=1 / window, adjust=False).mean()
    smoothed.name = f'SOA_{window}'

    columns_list = [smoothed.name]

    return smoothed, columns_list


def strategy_soa(
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
    SOA (Smoothed MA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast SmMA crosses above slow SmMA, sell when crosses below.
    WHY: SmMA applies heavier smoothing than SMA, similar to Wilder's MA.
         Useful for identifying long-term trends with less noise.
    BEST MARKETS: Long-term trending markets. Stocks, indices.
                  Good for position trading.
    TIMEFRAME: Daily or weekly charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 10), 'long_window' (default 20)
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
    
    short_window = int(parameters.get('short_window', 10))
    long_window = int(parameters.get('long_window', 20))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'SOA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='soa',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'SOA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='soa',
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
