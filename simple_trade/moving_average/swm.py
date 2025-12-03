import math
import numpy as np
import pandas as pd


def swm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Sine Weighted Moving Average (swm).
    swm uses sine wave weighting to emphasize the middle portion of the
    moving window, providing smooth transitions with reduced lag compared
    to simple moving averages.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the SWM series and a list of column names.

    The swm is calculated as follows:

    1. Generate Sine Weights:
       Weight(i) = sin((i + 1) * PI / (window + 1))
       (Weights follow a sine curve, peaking in the middle of the window)

    2. Normalize Weights:
       Normalized Weight(i) = Weight(i) / Sum(Weights)

    3. Calculate swm:
       swm = Sum(Price(i) * Normalized Weight(i)) over the window

    Interpretation:
    - swm gives less weight to recent data compared to WMA or EMA, but more than the beginning of the window.
    - It aims to extract the cyclical component of the price action.

    Use Cases:
    - Cycle Analysis: Better suited for identifying cyclical turning points than trend following.
    - Smoothing: Provides a very smooth average.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]

    # Generate sine weights
    weights = []
    for i in range(window):
        # Sine wave from 0 to π
        weight = math.sin((i + 1) * math.pi / (window + 1))
        weights.append(weight)
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    def weighted_avg(x):
        return np.dot(x, weights)

    swma_series = series.rolling(window=window).apply(weighted_avg, raw=True)
    swma_series.name = f'SWM_{window}'

    columns_list = [swma_series.name]
    return swma_series, columns_list


def strategy_swm(
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
    swm (Sine Weighted MA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast swm crosses above slow swm, sell when crosses below.
    WHY: swm uses sine wave weighting emphasizing middle of window.
         Provides smooth transitions with reduced lag.
    BEST MARKETS: Cyclical markets. Stocks, commodities. Good for
                  identifying cyclical turning points.
    TIMEFRAME: Daily charts. 20-period is standard.
    
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
        short_window_indicator = f'SWM_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='swm',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'SWM_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='swm',
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
