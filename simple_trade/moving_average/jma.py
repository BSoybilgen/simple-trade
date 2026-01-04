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
            - phase (float): Phase parameter (0 to 100) affecting overshoot. Default is 0.
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

    3. Calculate jma (Vectorized):
       This approximation mathematically simplifies to a single Exponential Moving Average (EMA)
       with an effective alpha derived from the length, power, and phase.
       
       Effective Alpha = 1 - (1 - Ratio) * (1 - SC)
       JMA = EMA(Price, alpha=Effective Alpha)

    Interpretation:
    - jma is famous for its low lag and smooth curve.
    - Positive phase allows the MA to overshoot price changes slightly, reducing lag further.
    - This implementation is fully vectorized for performance.

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

    phase = max(0.0, min(100.0, phase))
    phase_ratio = phase / 100.0

    close = df[close_col]
    
    # Calculate smoothing constant (alpha)
    # SC = (2 / (length + 1))^power
    alpha = (2.0 / (length + 1.0)) ** power
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Calculate Effective Alpha for the equivalent EMA
    # The iterative formula:
    # Base = Prev + alpha * (Price - Prev)
    # JMA = Base + beta * (Price - Base)
    #
    # Reduces to a single EMA with effective alpha:
    # alpha_eff = 1 - (1 - beta) * (1 - alpha)
    
    beta = phase_ratio
    effective_alpha = 1.0 - (1.0 - beta) * (1.0 - alpha)
    effective_alpha = np.clip(effective_alpha, 0.0, 1.0)
    
    # Calculate JMA using vectorized EMA
    # adjust=False matches the iterative recurrence relation
    jma_series = close.ewm(alpha=effective_alpha, adjust=False).mean()
    jma_series.name = f'JMA_{length}'
    
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

    short_length_param = parameters.get('short_length')
    long_length_param = parameters.get('long_length')

    if short_length_param is None and short_window_param is not None:
        short_length_param = short_window_param
    elif short_length_param is not None and short_window_param is not None:
        if int(short_length_param) != int(short_window_param):
            raise ValueError("Provide either 'short_length' or 'short_window'/'short_period' (aliases) with the same value if both are set.")

    if long_length_param is None and long_window_param is not None:
        long_length_param = long_window_param
    elif long_length_param is not None and long_window_param is not None:
        if int(long_length_param) != int(long_window_param):
            raise ValueError("Provide either 'long_length' or 'long_window'/'long_period' (aliases) with the same value if both are set.")

    short_length = int(short_length_param if short_length_param is not None else 10)
    long_length = int(long_length_param if long_length_param is not None else 21)
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
