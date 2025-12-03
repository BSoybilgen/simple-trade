import numpy as np
import pandas as pd


def jma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Jurík Moving Average (jma).
    jma is designed to remain smooth while maintaining low lag. This
    implementation approximates the JMA by dynamically adjusting a smoothing
    constant based on user-specified length and phase parameters.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - length (int): The lookback period. Default is 21.
            - phase (float): Phase parameter (-100 to 100) affecting overshoot. Default is 0.
            - power (float): Power parameter for smoothing curve. Default is 2.0.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the JMA series and a list of column names.

    The jma (Approximation) is calculated as follows:

    1. Calculate Base Smoothing Constant:
       SC = (2 / (length + 1))^power

    2. Calculate Phase Ratio:
       Ratio = phase / 100

    3. Calculate jma (Iterative):
       Base = Prev jma + SC * (Price - Prev jma)
       jma = Base + Ratio * (Price - Base)

    Interpretation:
    - jma is famous for its low lag and smooth curve.
    - Positive phase allows the MA to overshoot price changes slightly, reducing lag further.

    Use Cases:
    - Trend Following: Excellent for systems requiring fast reaction times.
    - Filtering: High noise reduction capability.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    length = max(1, int(parameters.get('length', 21)))
    phase = float(parameters.get('phase', 0.0))
    power = float(parameters.get('power', 2.0))

    phase = max(-100.0, min(100.0, phase))
    phase_ratio = phase / 100.0

    close = df[close_col].astype(float)
    values = close.to_numpy(dtype=float)
    jma_values = np.full_like(values, np.nan)

    smoothing_constant = (2.0 / (length + 1.0)) ** power
    smoothing_constant = np.clip(smoothing_constant, 0.0, 1.0)

    non_nan_idx = np.where(~np.isnan(values))[0]
    if non_nan_idx.size == 0:
        series = pd.Series(jma_values, index=close.index, name=f'JMA_{length}')
        return series, [series.name]

    start = non_nan_idx[0]
    jma_values[start] = values[start]

    for i in range(start + 1, len(values)):
        price = values[i]
        prev = jma_values[i - 1]
        if np.isnan(price):
            jma_values[i] = prev
            continue
        if np.isnan(prev):
            prev = price

        base = prev + smoothing_constant * (price - prev)
        jma_values[i] = base + phase_ratio * (price - base)

    jma_series = pd.Series(jma_values, index=close.index, name=f'JMA_{length}')
    return jma_series, [jma_series.name]


def strategy_jma(
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
    jma (Jurik Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast jma crosses above slow jma, sell when crosses below.
    WHY: jma is famous for low lag and smooth curve. Phase parameter allows
         overshoot control for even faster reaction.
    BEST MARKETS: Fast-moving markets. Stocks, forex, futures. Excellent for
                  systems requiring fast reaction times.
    TIMEFRAME: All timeframes. 21-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_length' (default 10), 'long_length' (default 21)
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
    
    short_length = int(parameters.get('short_length', 10))
    long_length = int(parameters.get('long_length', 21))
    price_col = 'Close'
    
    if short_length == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'JMA_{short_length}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='jma',
            parameters={"length": short_length},
            figure=False
        )
    
    long_window_indicator = f'JMA_{long_length}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='jma',
        parameters={"length": long_length},
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
