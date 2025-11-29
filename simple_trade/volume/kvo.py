import pandas as pd
import numpy as np


def kvo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Klinger Volume Oscillator (KVO), a long-term money flow indicator
    that compares volume flowing through securities with price movements. It is designed
    to detect long-term money flow trends while remaining sensitive to short-term fluctuations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_period (int): The fast EMA period. Default is 34.
            - slow_period (int): The slow EMA period. Default is 55.
            - signal_period (int): The signal line EMA period. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing a DataFrame with KVO and Signal, and a list of column names.

    The Klinger Volume Oscillator is calculated as follows:

    1. Calculate Trend Direction:
       If Typical Price > Previous Typical Price, Trend = +1, else -1.

    2. Calculate Volume Force (VF):
       VF = Volume * Trend * Abs(2 * ((High - Low) / Cumulative(High - Low)) - 1) * 100

    3. Calculate KVO:
       KVO = EMA(VF, fast_period) - EMA(VF, slow_period)

    4. Calculate Signal Line:
       Signal = EMA(KVO, signal_period)

    Interpretation:
    - KVO > 0: Bullish money flow (Accumulation).
    - KVO < 0: Bearish money flow (Distribution).
    - Signal Crossover: KVO crossing Signal Line indicates potential entry/exit points.

    Use Cases:
    - Trend Identification: Positive KVO implies uptrend; negative implies downtrend.
    - Divergence: Price trends not supported by money flow signal reversals.
    - Trade Signals: Crossovers of the Signal Line or Zero Line.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    fast_period = parameters.get('fast_period', 34)
    slow_period = parameters.get('slow_period', 55)
    signal_period = parameters.get('signal_period', 13)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate typical price
    hlc = high + low + close
    
    # Determine trend
    trend = pd.Series(1, index=df.index)
    trend[hlc < hlc.shift(1)] = -1
    
    # Calculate daily measurement
    dm = high - low
    
    # Calculate cumulative measurement
    cm = dm.copy()
    for i in range(1, len(df)):
        if trend.iloc[i] == trend.iloc[i-1]:
            cm.iloc[i] = cm.iloc[i-1] + dm.iloc[i]
        else:
            cm.iloc[i] = dm.iloc[i-1] + dm.iloc[i]
    
    # Calculate volume force
    vf = volume * trend * abs(2 * ((dm / cm.replace(0, np.nan)) - 1)) * 100
    vf = vf.fillna(0)
    
    # Calculate KVO
    fast_ema = vf.ewm(span=fast_period, adjust=False).mean()
    slow_ema = vf.ewm(span=slow_period, adjust=False).mean()
    kvo_values = fast_ema - slow_ema
    
    # Calculate signal line
    signal = kvo_values.ewm(span=signal_period, adjust=False).mean()
    
    result = pd.DataFrame({
        f'KVO_{fast_period}_{slow_period}': kvo_values,
        f'KVO_SIGNAL_{signal_period}': signal
    }, index=df.index)
    
    columns_list = list(result.columns)
    return result, columns_list


def strategy_kvo(
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
    KVO (Klinger Volume Oscillator) - Signal Line Crossover Strategy
    
    LOGIC: Buy when KVO crosses above signal line (bullish money flow),
           sell when KVO crosses below signal line (bearish money flow).
    WHY: KVO compares volume flowing through securities with price movements.
         Designed for long-term money flow trends while sensitive to short-term.
    BEST MARKETS: Stocks, ETFs. Good for trend identification and divergence.
    TIMEFRAME: Daily charts. 34/55 fast/slow with 13 signal is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'fast_period' (default 34), 'slow_period' (default 55),
                    'signal_period' (default 13)
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
    
    fast_period = int(parameters.get('fast_period', 34))
    slow_period = int(parameters.get('slow_period', 55))
    signal_period = int(parameters.get('signal_period', 13))
    price_col = 'Close'
    kvo_col = f'KVO_{fast_period}_{slow_period}'
    signal_col = f'KVO_SIGNAL_{signal_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='kvo',
        parameters={"fast_period": fast_period, "slow_period": slow_period, "signal_period": signal_period},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=kvo_col,
        long_window_indicator=signal_col,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [kvo_col, signal_col]
    
    return results, portfolio, indicator_cols_to_plot, data
