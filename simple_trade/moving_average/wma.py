import pandas as pd
import numpy as np


def wma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Weighted Moving Average (wma) of a series.
    The wma is a moving average that assigns different weights to data points,
    typically giving more weight to recent data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the wma. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the wma series and a list of column names.

    Calculation Steps:
    1. Generate Weights:
       Create an array of weights from 1 to window.
       Weights = [1, 2, ..., window]
    2. Calculate Weighted Average:
       wma = Sum(Price * Weight) / Sum(Weights)

    Interpretation:
    - More responsive to recent price changes than SMA due to linear weighting.
    - Less lag than SMA but more than EMA (typically).

    Use Cases:
    - Trend Identification: Identifying direction with less lag than SMA.
    - Crossovers: Using wma in crossover strategies for faster signals.
    - Smoothing: General price smoothing.
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
    
    series = df[close_col]
    weights = np.arange(1, window + 1)
    series = series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    series.name = f'WMA_{window}'
    columns = [series.name]
    return series, columns


def strategy_wma(
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
    wma (Weighted Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast wma crosses above slow wma, sell when crosses below.
    WHY: wma gives linear weighting to recent prices, more responsive than SMA
         but less than EMA. Good balance of smoothness and responsiveness.
    BEST MARKETS: Trending markets. Stocks, forex, commodities.
                  Good for medium-term trend following.
    TIMEFRAME: All timeframes. Common pairs: 10/20, 20/50.
    
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

    short_window = int(short_window_param if short_window_param is not None else 10)
    long_window = int(long_window_param if long_window_param is not None else 20)
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'WMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='wma',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'WMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='wma',
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