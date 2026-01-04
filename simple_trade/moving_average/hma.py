import pandas as pd
import numpy as np
from .wma import wma

def hma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Hull Moving Average (hma) of a series.
    The hma is a moving average that reduces lag and improves smoothing.
    It is calculated using weighted moving averages (WMAs) with specific
    window lengths to achieve this effect.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the hma. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the hma series and a list of column names.

    Calculation Steps:
    1. Calculate WMA of Half Length:
       WMA1 = WMA(Close, window / 2)

    2. Calculate WMA of Full Length:
       WMA2 = WMA(Close, window)

    3. Calculate Raw hma:
       Raw = 2 * WMA1 - WMA2

    4. Calculate Final hma:
       hma = WMA(Raw, sqrt(window))

    Interpretation:
    - hma hugs the price action much closer than SMA or EMA.
    - The turning points in hma are often sharper and more timely.

    Use Cases:
    - Identifying trends: The hma can be used to identify the direction of a
      price trend.
    - Smoothing price data: The hma can smooth out short-term price fluctuations
      to provide a clearer view of the underlying trend.
    - Generating buy and sell signals: The hma can be used in crossover systems
      to generate buy and sell signals.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    window = int(window_param if window_param is not None else 20)

    half_length = int(window / 2)
    sqrt_length = int(np.sqrt(window))
    # Create parameter dicts for wma function
    wma_params_half = {'window': half_length}
    wma_params_full = {'window': window}
    wma_cols = {'close_col': close_col}
    
    # wma now returns a tuple (series, columns)
    wma_half_series = wma(df, parameters=wma_params_half, columns=wma_cols)[0]
    wma_full_series = wma(df, parameters=wma_params_full, columns=wma_cols)[0]

    df_mid = pd.DataFrame(2 * wma_half_series - wma_full_series, columns=[close_col])
    
    wma_params_sqrt = {'window': sqrt_length}
    hma_series = wma(df_mid, parameters=wma_params_sqrt, columns=wma_cols)[0]
    hma_series.name = f'HMA_{window}'

    columns_list = [hma_series.name]
    return hma_series, columns_list


def strategy_hma(
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
    hma (Hull Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast hma crosses above slow hma, sell when crosses below.
    WHY: hma reduces lag significantly while maintaining smoothness. Uses WMA
         combination to achieve near-zero lag with good noise filtering.
    BEST MARKETS: Trending markets where quick response is needed. Stocks, forex.
                  Excellent for swing trading due to reduced lag.
    TIMEFRAME: All timeframes. Particularly effective on 4H and daily charts.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 9), 'long_window' (default 21)
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
    
    short_window_param = parameters.get('short_window')
    short_period_param = parameters.get('short_period')
    if short_window_param is None and short_period_param is not None:
        short_window_param = short_period_param
    elif short_window_param is not None and short_period_param is not None:
        if int(short_window_param) != int(short_period_param):
            raise ValueError("Provide either 'short_window' or 'short_period' (aliases) with the same value if both are set.")

    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    short_window = int(short_window_param if short_window_param is not None else 9)
    long_window = int(long_window_param if long_window_param is not None else 21)
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'HMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='hma',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'HMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='hma',
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