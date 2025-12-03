import pandas as pd
import numpy as np


def pvo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Percentage Volume Oscillator (pvo), a momentum oscillator for volume
    that shows the relationship between two volume moving averages as a percentage.
    It is similar to MACD but applied to volume.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_period (int): The fast EMA period. Default is 12.
            - slow_period (int): The slow EMA period. Default is 26.
            - signal_period (int): The signal line EMA period. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing a DataFrame with PVO, Signal, and Histogram, and a list of column names.

    The Percentage Volume Oscillator is calculated as follows:

    1. Calculate Volume EMAs:
       Fast EMA = EMA(Volume, fast_period)
       Slow EMA = EMA(Volume, slow_period)

    2. Calculate PVO:
       PVO = ((Fast EMA - Slow EMA) / Slow EMA) * 100

    3. Calculate Signal Line:
       Signal = EMA(PVO, signal_period)

    4. Calculate Histogram:
       Histogram = PVO - Signal

    Interpretation:
    - Positive PVO: Volume is increasing (Fast EMA > Slow EMA).
    - Negative PVO: Volume is decreasing (Fast EMA < Slow EMA).
    - Histogram: Shows the momentum of volume changes.

    Use Cases:
    - Volume Trend: Identify if volume is expanding or contracting.
    - Breakout Confirmation: Rising PVO during breakouts confirms the move.
    - Divergence: Price highs not supported by PVO highs indicate weakness.
    - Entry/Exit: PVO crossing Signal Line or Zero Line.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    fast_period = parameters.get('fast_period', 12)
    slow_period = parameters.get('slow_period', 26)
    signal_period = parameters.get('signal_period', 9)
    volume_col = columns.get('volume_col', 'Volume')
    
    volume = df[volume_col]
    
    # Calculate Fast and Slow EMAs of volume
    fast_ema = volume.ewm(span=fast_period, adjust=False).mean()
    slow_ema = volume.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate PVO (handle division by zero)
    pvo_values = ((fast_ema - slow_ema) / slow_ema.replace(0, np.nan)) * 100
    pvo_values = pvo_values.fillna(0)
    
    # Calculate Signal Line
    signal = pvo_values.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Histogram
    histogram = pvo_values - signal
    
    # Create result DataFrame
    result = pd.DataFrame({
        f'PVO_{fast_period}_{slow_period}': pvo_values,
        f'PVO_SIGNAL_{signal_period}': signal,
        'PVO_HIST': histogram
    }, index=df.index)
    
    columns_list = list(result.columns)
    return result, columns_list


def strategy_pvo(
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
    pvo (Percentage Volume Oscillator) - Signal Line Crossover Strategy
    
    LOGIC: Buy when pvo crosses above signal line (volume expanding),
           sell when pvo crosses below signal line (volume contracting).
    WHY: pvo is like MACD for volume. Shows relationship between two volume MAs.
         Positive PVO indicates expanding volume, negative indicates contracting.
    BEST MARKETS: Stocks, ETFs. Good for breakout confirmation.
    TIMEFRAME: Daily charts. 12/26/9 periods is standard (like MACD).
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'fast_period' (default 12), 'slow_period' (default 26),
                    'signal_period' (default 9)
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
    
    fast_period = int(parameters.get('fast_period', 12))
    slow_period = int(parameters.get('slow_period', 26))
    signal_period = int(parameters.get('signal_period', 9))
    price_col = 'Close'
    pvo_col = f'PVO_{fast_period}_{slow_period}'
    signal_col = f'PVO_SIGNAL_{signal_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='pvo',
        parameters={"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=pvo_col,
        long_window_indicator=signal_col,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [pvo_col, signal_col]
    
    return results, portfolio, indicator_cols_to_plot, data
