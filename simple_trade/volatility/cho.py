import pandas as pd
import numpy as np


def cho(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Choppiness Index (cho), a volatility indicator designed to determine
    whether the market is trending or trading sideways (choppy). It measures the market's
    trendiness on a scale from 0 to 100.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculations. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Choppiness Index series and a list of column names.

    Calculation Steps:
    1. Calculate True Range (TR) for each period.
    2. Sum True Range over the period: SumTR = Sum(TR, period)
    3. Calculate High-Low Range over the period: Range = Highest High - Lowest Low
    4. Calculate CHO: CHO = 100 * log10(SumTR / Range) / log10(period)

    Interpretation:
    - High cho (>61.8): Market is consolidating (choppy), avoid trend-following strategies.
    - Low cho (<38.2): Market is trending, favorable for trend-following strategies.
    - Rising cho: Trend is weakening, market entering consolidation.
    - Falling cho: Consolidation is ending, potential breakout approaching.

    Use Cases:
    - Trend vs. Range identification: Determine market regime.
    - Trade filtering: Filter out trend trades during high choppiness.
    - Breakout anticipation: Low volatility/high chop often precedes breakouts.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 14))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    sum_tr = tr.rolling(window=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    # Calculate high-low range
    hl_range = highest_high - lowest_low
    
    # Calculate Choppiness Index
    # CHO = 100 * log10(sum_tr / hl_range) / log10(period)
    # Handle division by zero
    cho_values = pd.Series(index=close.index, dtype=float)
    
    # Only calculate where hl_range > 0 to avoid division by zero
    valid_mask = hl_range > 0
    
    cho_values[valid_mask] = (
        100 * np.log10(sum_tr[valid_mask] / hl_range[valid_mask]) / np.log10(period)
    )
    
    cho_values.name = f'CHO_{period}'
    columns_list = [cho_values.name]
    return cho_values, columns_list


def strategy_cho(
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
    cho (Choppiness Index) - Trend vs Range Strategy
    
    LOGIC: Buy when cho drops below lower threshold (trending market),
           sell when rises above upper threshold (choppy/ranging market).
    WHY: cho measures market trendiness on 0-100 scale. High cho (>61.8)
         indicates consolidation, low cho (<38.2) indicates trending.
    BEST MARKETS: All markets. Use to filter trend-following strategies.
                  Avoid trend trades during high choppiness.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14), 'upper' (default 61.8),
                    'lower' (default 38.2)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_band_trade_strategies import run_band_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    period = int(parameters.get('period', 14))
    upper = float(parameters.get('upper', 61.8))
    lower = float(parameters.get('lower', 38.2))
    price_col = 'Close'
    indicator_col = f'CHO_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='cho',
        parameters={"period": period},
        figure=False
    )
    
    data['upper'] = upper
    data['lower'] = lower
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
