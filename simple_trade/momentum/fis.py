import numpy as np
import pandas as pd


def fis(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Fisher Transform (fis), a technical indicator created by John Ehlers 
    that converts prices into a Gaussian normal distribution.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for normalization. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Fisher Transform series and a list of column names.

    The Fisher Transform is calculated in several steps:

    1. Calculate Median Price:
       Median = (High + Low) / 2

    2. Normalize Median Price (over window):
       Value = 2 * ((Median - Min) / (Max - Min) - 0.5)
       (Value is clipped to stay within boundaries close to -1 and 1)

    3. Smooth the Normalized Value (optional/specific to implementation):
       Value = 0.33 * Value + 0.67 * Previous Value

    4. Calculate Fisher Transform:
       Fisher = 0.5 * ln((1 + Value) / (1 - Value)) + 0.5 * Previous Fisher

    Interpretation:
    - Sharp Turning Points: The Fisher Transform creates sharp peaks and troughs, making turning points clearer.
    - Extremes: Extreme positive values indicate overbought conditions; extreme negative values indicate oversold.
    - Crossings: Crossing the signal line (often the previous value or a moving average of Fisher) can generate signals.

    Use Cases:
    - Reversal Detection: Identifying precise reversal points in the market price.
    - Trend Analysis: Assessing the direction and strength of the trend.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 9))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    high = df[high_col]
    low = df[low_col]
    median_price = (high + low) / 2

    highest_high = median_price.rolling(window=window).max()
    lowest_low = median_price.rolling(window=window).min()
    price_range = (highest_high - lowest_low).replace(0, pd.NA)

    normalized = 2 * ((median_price - lowest_low) / price_range - 0.5)
    normalized = normalized.clip(-0.999, 0.999)

    fisher_values = pd.Series(index=df.index, dtype=float)
    fisher_values.name = f'FIS_{window}'

    prev_value = 0.0

    for idx in range(len(df)):
        value = normalized.iat[idx]
        if pd.isna(value):
            fisher_values.iat[idx] = np.nan
            continue

        value = 0.33 * value + 0.67 * prev_value
        value = max(min(value, 0.999), -0.999)

        fisher = 0.5 * np.log((1 + value) / (1 - value))
        fisher_values.iat[idx] = fisher
        prev_value = value

    columns_list = [fisher_values.name]
    return fisher_values, columns_list


def strategy_fis(
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
    fis (Fisher Transform) - Zero Line Crossover Strategy
    
    LOGIC: Buy when fis crosses above zero (bullish), sell when crosses below.
    WHY: fis converts prices to Gaussian distribution, creating sharp
         turning points. Zero crossings indicate momentum shifts with clear signals.
    BEST MARKETS: Trending markets with clear reversals. Forex, stocks, and futures.
                  Creates sharp peaks/troughs making reversals easier to identify.
    TIMEFRAME: Daily or 4-hour charts. 9-period is common. Good for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 9)
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
    
    window = int(parameters.get('window', 9))
    
    indicator_params = {"window": window}
    short_window_indicator = f'FIS_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='fis',
        parameters=indicator_params,
        figure=False
    )
    
    # Create zero line for crossover strategy
    data['zero_line'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, 'zero_line']
    
    return results, portfolio, indicator_cols_to_plot, data
