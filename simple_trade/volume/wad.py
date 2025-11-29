import pandas as pd
import numpy as np


def wad(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Williams Accumulation/Distribution (WAD), an indicator that uses
    price changes (True Range) to determine accumulation or distribution.
    It measures market pressure by comparing the close to the true range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the WAD series and a list of column names.

    The Williams Accumulation/Distribution is calculated as follows:

    1. Calculate True Range High (TRH) and True Range Low (TRL):
       TRH = Max(Current High, Previous Close)
       TRL = Min(Current Low, Previous Close)

    2. Calculate Price Move (PM):
       - If Close > Previous Close: PM = Close - TRL
       - If Close < Previous Close: PM = Close - TRH
       - If Close = Previous Close: PM = 0

    3. Calculate WAD (Cumulative):
       WAD = Previous WAD + PM

    Interpretation:
    - Rising WAD: Accumulation (Buying pressure).
    - Falling WAD: Distribution (Selling pressure).
    - Divergence: Price vs WAD divergence signals potential reversals.

    Use Cases:
    - Trend Confirmation: WAD should align with price direction.
    - Divergence Analysis: Powerful tool for spotting tops and bottoms.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    prev_close = close.shift(1)
    
    # 1. True Range High and Low
    # We use numpy fmax/fmin for element-wise comparison, filling NaN prev_close first
    # For the first element, prev_close is NaN. We can fill it with current close or high/low
    # Standard practice is usually to start accumulation from 0 or match price
    
    trh = np.maximum(high, prev_close.fillna(high))
    trl = np.minimum(low, prev_close.fillna(low))
    
    # 2. Calculate Price Move
    ad = pd.Series(0.0, index=df.index)
    
    # Close > Prev Close
    mask_up = close > prev_close
    ad[mask_up] = close[mask_up] - trl[mask_up]
    
    # Close < Prev Close
    mask_down = close < prev_close
    ad[mask_down] = close[mask_down] - trh[mask_down]
    
    # Close == Prev Close is already 0
    
    # 3. Cumulative Sum
    wad_values = ad.cumsum()
    
    wad_values.name = 'WAD'
    columns_list = [wad_values.name]
    return wad_values, columns_list


def strategy_wad(
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
    WAD (Williams Accumulation/Distribution) - SMA Crossover Strategy
    
    LOGIC: Buy when WAD crosses above its SMA (accumulation),
           sell when WAD crosses below its SMA (distribution).
    WHY: WAD uses price changes (True Range) to determine accumulation/distribution.
         Rising WAD indicates buying pressure, falling indicates selling pressure.
    BEST MARKETS: Stocks, ETFs. Good for trend confirmation and divergence.
    TIMEFRAME: Daily charts. Good for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'sma_period' (default 20)
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
    
    sma_period = int(parameters.get('sma_period', 20))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='wad',
        parameters={},
        figure=False
    )
    
    # Calculate SMA of WAD for crossover signals
    data[f'WAD_SMA_{sma_period}'] = data['WAD'].rolling(window=sma_period).mean()
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='WAD',
        long_window_indicator=f'WAD_SMA_{sma_period}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['WAD', f'WAD_SMA_{sma_period}']
    
    return results, portfolio, indicator_cols_to_plot, data
