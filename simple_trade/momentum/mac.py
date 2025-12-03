import pandas as pd

def mac(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Moving Average Convergence Divergence (mac), Signal Line, and Histogram.
    It is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window_slow (int): The window size for the slower EMA. Default is 26.
            - window_fast (int): The window size for the faster EMA. Default is 12.
            - window_signal (int): The window size for the signal line EMA. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the MAC DataFrame (MAC, Signal, Hist) and a list of column names.

    Calculation Steps:
    1. Calculate the Fast EMA:
       Fast EMA = EMA(Close, window_fast)

    2. Calculate the Slow EMA:
       Slow EMA = EMA(Close, window_slow)

    3. Calculate the MAC Line:
       MAC Line = Fast EMA - Slow EMA

    4. Calculate the Signal Line:
       Signal Line = EMA(MAC Line, window_signal)

    5. Calculate the MAC Histogram:
       Histogram = MAC Line - Signal Line

    Interpretation:
    - Crossovers: MAC crossing above Signal Line is bullish; crossing below is bearish.
    - Zero Line: MAC crossing above zero suggests uptrend (Fast EMA > Slow EMA); below zero suggests downtrend.
    - Divergence: Divergence between Price and MAC/Histogram suggests waning momentum and potential reversal.

    Use Cases:
    - Trend Identification: Confirming trend direction and strength.
    - Entry/Exit Signals: Signal line crossovers are common entry/exit points.
    - Momentum Measurement: The width of the histogram indicates the speed of price movement.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window_slow = int(parameters.get('window_slow', 26))
    window_fast = int(parameters.get('window_fast', 12))
    window_signal = int(parameters.get('window_signal', 9))
    close_col = columns.get('close_col', 'Close')
    
    series = df[close_col]
    ema_fast = series.ewm(span=window_fast, adjust=False).mean()
    ema_slow = series.ewm(span=window_slow, adjust=False).mean()
    mac_line = ema_fast - ema_slow
    signal_line = mac_line.ewm(span=window_signal, adjust=False).mean()
    histogram = mac_line - signal_line
    # Return DataFrame for multi-output indicators
    df_mac = pd.DataFrame({
        f'MAC_{window_fast}_{window_slow}': mac_line,
        f'Signal_{window_signal}': signal_line,
        f'Hist_{window_fast}_{window_slow}_{window_signal}': histogram
    })
    # Ensure index is passed explicitly, just in case
    df_mac.index = series.index
    columns_list = list(df_mac.columns)
    return df_mac, columns_list


def strategy_mac(
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
    mac - Signal Line Crossover Strategy
    
    LOGIC: Buy when mac line crosses above signal line, sell when crosses below.
    WHY: mac captures momentum shifts by comparing fast and slow EMAs. Signal line
         crossovers indicate changes in trend momentum. Classic trend-following indicator.
    BEST MARKETS: Trending markets across all asset classes. Stocks, forex, crypto,
                  commodities. One of the most versatile and widely-used indicators.
    TIMEFRAME: All timeframes. Daily/weekly for position trading, 4H/1H for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window_fast' (default 12), 'window_slow' (default 26),
                   'window_signal' (default 9)
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
    
    window_fast = int(parameters.get('window_fast', 12))
    window_slow = int(parameters.get('window_slow', 26))
    window_signal = int(parameters.get('window_signal', 9))
    
    indicator_params = {
        "window_fast": window_fast,
        "window_slow": window_slow,
        "window_signal": window_signal
    }
    short_window_indicator = f'MAC_{window_fast}_{window_slow}'
    long_window_indicator = f'Signal_{window_signal}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='mac',
        parameters=indicator_params,
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