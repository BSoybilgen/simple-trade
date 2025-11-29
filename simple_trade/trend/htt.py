import pandas as pd


def htt(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Hilbert Transform Trendline (HTT).
    HTT approximates John Ehlers' Hilbert Transform smoothing by filtering the
    close series with the standard detrender kernel and subtracting it from the
    original price to derive a low-lag trendline.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The smoothing window. Default is 16.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the HTT series and a list of column names.

    The Hilbert Transform Trendline is calculated as follows:

    1. Calculate Detrended Price (Smooth Component extraction):
       Kernel = 0.0962*P + 0.5769*P[2] - 0.5769*P[4] - 0.0962*P[6]
       (This kernel is designed to remove the DC component and low frequencies)

    2. Smooth the Detrended Component:
       Apply EMA smoothing to the result.

    Interpretation:
    - HTT provides a trendline that reacts faster than traditional MAs.
    - It is effectively an Instantaneous Trendline derived from signal processing principles.

    Use Cases:
    - Trend Following: Using the slope of HTT to determine trend direction.
    - Crossovers: Price crossing HTT.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = max(7, int(parameters.get('window', 16)))  # need at least 7 samples for kernel

    close = df[close_col]
    
    # Apply Ehlers' Hilbert Transform smoothing
    # The detrender coefficients extract the smooth trend component
    smooth = (
        0.0962 * close
        + 0.5769 * close.shift(2)
        - 0.5769 * close.shift(4)
        - 0.0962 * close.shift(6)
    )
    
    # Apply EMA smoothing to the result
    alpha = 2.0 / (window + 1)
    trendline = smooth.ewm(alpha=alpha, adjust=False).mean()
    trendline.name = f'HTT_{window}'

    return trendline, [trendline.name]


def strategy_htt(
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
    HTT (Hilbert Transform Trendline) - Dual-HTT Crossover Strategy
    
    LOGIC: Generate signals when a faster HTT line crosses a slower HTT line.
    WHY: Using two HTT windows preserves the low-lag benefits of the indicator
         while adding confirmation through crossover behavior similar to MA pairs.
    BEST MARKETS: All markets. Stocks, forex, futures. Particularly useful when
                  you want smooth, low-lag crossovers without reverting to MAs.
    TIMEFRAME: Daily charts. Default short window 16, long window 24.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 16)
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
    
    long_window = int(parameters.get('window', 24))
    short_window = int(parameters.get('window', 16))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='htt',
        parameters={"window": long_window},
        figure=False
    )

    data, _, _ = compute_indicator(
        data=data,
        indicator='htt',
        parameters={"window": short_window},
        figure=False
    )
    
    short_window_indicator = f'HTT_{short_window}'
    long_window_indicator = f'HTT_{long_window}'
    
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
    
    # Include Close price in plot so users can see the crossover signals
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data
