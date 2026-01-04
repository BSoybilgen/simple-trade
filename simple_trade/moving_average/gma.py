import pandas as pd
from typing import Iterable, List


def gma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Guppy Multiple Moving Average (gma).
    gma applies two sets of exponential moving averages (EMAs):
    short-term averages that react quickly to price changes and long-term
    averages that capture the broader trend. The relationship between the two
    groups helps identify trend strength and potential transitions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_windows (iterable): List of periods for short-term EMAs. Default is (3, 5, 8, 10, 12, 15).
            - long_windows (iterable): List of periods for long-term EMAs. Default is (30, 35, 40, 45, 50, 60).
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the GMA DataFrame (multiple EMA columns) and a list of column names.

    The gma is calculated as follows:

    1. Calculate Short-Term EMAs:
       Calculate EMA for each period in short_windows.

    2. Calculate Long-Term EMAs:
       Calculate EMA for each period in long_windows.

    Interpretation:
    - Compression: When EMAs bunch together, it indicates agreement on price and potential for a breakout.
    - Expansion: When EMAs spread out, it indicates a strong trend.
    - Crossovers: Short-term group crossing above long-term group indicates a bullish trend reversal.
    - Separation: The space between the short-term and long-term groups indicates the strength of the trend.

    Use Cases:
    - Trend Identification: Determining the long-term trend direction.
    - Entry Signals: Short-term pullbacks into the long-term group during a strong trend.
    - Reversal Signals: Crossover of the two groups.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')

    if parameters.get('short_windows') is None and parameters.get('short_periods') is not None:
        parameters['short_windows'] = parameters.get('short_periods')
    if parameters.get('long_windows') is None and parameters.get('long_periods') is not None:
        parameters['long_windows'] = parameters.get('long_periods')

    short_windows_param = parameters.get('short_windows')
    short_window_param = parameters.get('short_window')
    short_period_param = parameters.get('short_period')
    if short_window_param is None and short_period_param is not None:
        short_window_param = short_period_param
    elif short_window_param is not None and short_period_param is not None:
        if int(short_window_param) != int(short_period_param):
            raise ValueError("Provide either 'short_window' or 'short_period' (aliases) with the same value if both are set.")

    long_windows_param = parameters.get('long_windows')
    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    short_windows_default = (3, 5, 8, 10, 12, 15)
    if short_windows_param is None:
        short_windows = _normalize_windows(short_window_param if short_window_param is not None else short_windows_default)
    else:
        short_windows = _normalize_windows(short_windows_param)
        if short_window_param is not None:
            short_window_value = int(short_window_param)
            if len(short_windows) != 1 or int(short_windows[0]) != short_window_value:
                raise ValueError("Provide either 'short_windows' or 'short_window'/'short_period' (aliases). If both are set, 'short_windows' must contain exactly that value.")

    long_windows_default = (30, 35, 40, 45, 50, 60)
    if long_windows_param is None:
        long_windows = _normalize_windows(long_window_param if long_window_param is not None else long_windows_default)
    else:
        long_windows = _normalize_windows(long_windows_param)
        if long_window_param is not None:
            long_window_value = int(long_window_param)
            if len(long_windows) != 1 or int(long_windows[0]) != long_window_value:
                raise ValueError("Provide either 'long_windows' or 'long_window'/'long_period' (aliases). If both are set, 'long_windows' must contain exactly that value.")

    close = df[close_col]

    ema_data = {}
    for window in short_windows:
        ema_series = close.ewm(span=window, adjust=False).mean()
        ema_series.name = f'GMA_short_{window}'
        ema_data[ema_series.name] = ema_series

    for window in long_windows:
        ema_series = close.ewm(span=window, adjust=False).mean()
        ema_series.name = f'GMA_long_{window}'
        ema_data[ema_series.name] = ema_series

    result_df = pd.DataFrame(ema_data, index=close.index)
    columns_list = list(result_df.columns)

    return result_df, columns_list


def _normalize_windows(value: Iterable) -> List[int]:
    try:
        windows = list(value)
    except TypeError:
        windows = [value]

    cleaned: List[int] = []
    for window in windows:
        try:
            window_int = int(window)
        except (TypeError, ValueError):
            continue
        if window_int > 0:
            cleaned.append(window_int)

    if not cleaned:
        cleaned = [1]

    return cleaned


def strategy_gma(
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
    gma (Guppy Multiple MA) - Short vs Long Group Crossover Strategy
    
    LOGIC: Buy when shortest short-term EMA crosses above longest long-term EMA,
           sell when crosses below.
    WHY: gma uses two groups of EMAs to show trend strength and transitions.
         Compression indicates consolidation, expansion indicates strong trend.
    BEST MARKETS: Trending markets. Stocks, forex, indices. Excellent for
                  identifying trend reversals and strength.
    TIMEFRAME: Daily charts. Standard short: 3,5,8,10,12,15; long: 30,35,40,45,50,60.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_windows', 'long_windows'
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
    
    price_col = 'Close'

    if parameters.get('short_windows') is None and parameters.get('short_periods') is not None:
        parameters['short_windows'] = parameters.get('short_periods')
    if parameters.get('long_windows') is None and parameters.get('long_periods') is not None:
        parameters['long_windows'] = parameters.get('long_periods')
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='gma',
        parameters=parameters,
        figure=False
    )
    
    # Use shortest short-term and longest long-term for crossover
    short_cols = [c for c in columns if 'short' in c]
    long_cols = [c for c in columns if 'long' in c]
    
    short_window_indicator = min(short_cols) if short_cols else 'GMA_short_3'
    long_window_indicator = max(long_cols) if long_cols else 'GMA_long_60'
    
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
