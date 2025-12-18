import numpy as np
import pandas as pd


def cog(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Center of Gravity (cog) oscillator, developed by John Ehlers.
    It is designed to spot turning points in prices with zero lag by analogy to the physical center of gravity.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the cog series and a list of column names.

    The Center of Gravity is calculated using a weighted sum of prices:

    1. Assign weights to prices in the window:
       Weights range from 1 to window (most recent).

    2. Calculate Weighted Sums:
       Numerator = -Sum(Price[i] * (i + 1)) over the window (conceptually adjusted for lag)
       Denominator = Sum(Price[i]) over the window

    3. Calculate cog:
       cog = -Numerator / Denominator + (window + 1) / 2
       
       (Note: The specific implementation details may vary to align with Ehler's filter formula).

    Interpretation:
    - The cog acts as a leading indicator, helping to identify market turning points.
    - It helps in visualizing the "center of mass" of the price action over the window.

    Use Cases:
    - Turning Points: Identifying potential peaks and valleys in price action.
    - Cycle Analysis: Used in conjunction with other cycle-based indicators.
    - Trend Reversals: Sharp changes in cog can signal immediate trend reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 10)
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]

    def _cog(values: np.ndarray) -> float:
        if len(values) < window:
            return np.nan
        denom = values.sum()
        if denom == 0:
            return np.nan
        weights = np.arange(1, len(values) + 1, dtype=float)
        weighted_sum = (values[::-1] * weights).sum()
        return -weighted_sum / denom + (len(values) + 1) / 2

    cog_series = series.rolling(window=window, min_periods=window).apply(_cog, raw=True)
    cog_series.name = f'COG_{window}'

    columns_list = [cog_series.name]
    return cog_series, columns_list


def strategy_cog(
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
    cog (Center of Gravity) - Zero Line Crossover Strategy
    
    LOGIC: Buy when cog crosses above zero (bullish momentum), sell when crosses below.
    WHY: cog uses weighted linear regression to identify turning points with minimal lag.
         Developed by John Ehlers for cycle analysis. Zero crossings indicate momentum shifts.
    BEST MARKETS: Cyclical markets and assets with regular oscillations. Works well
                  on forex, indices, and stocks with predictable cycles.
    TIMEFRAME: Short-term trading (intraday to daily). Sensitive to noise on very short TFs.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 10), 'signal_window' (default 3)
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 10)

    signal_window_param = parameters.get('signal_window')
    signal_period_param = parameters.get('signal_period')
    if signal_window_param is None and signal_period_param is not None:
        signal_window_param = signal_period_param
    elif signal_window_param is not None and signal_period_param is not None:
        if int(signal_window_param) != int(signal_period_param):
            raise ValueError("Provide either 'signal_window' or 'signal_period' (aliases) with the same value if both are set.")
    signal_window = int(signal_window_param if signal_window_param is not None else 3)
    
    indicator_params = {"window": window}
    short_window_indicator = f'COG_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='cog',
        parameters=indicator_params,
        figure=False
    )
    
    # Create signal line as SMA of COG
    data['COG_Signal'] = data[short_window_indicator].rolling(window=signal_window).mean()
    long_window_indicator = 'COG_Signal'
    
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
