import pandas as pd
import numpy as np


def adl(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Accumulation/Distribution Line (adl), a volume-based indicator
    that measures the cumulative flow of money into and out of a security. Unlike OBV
    which only considers price direction, the ADL considers the position of the
    close relative to the trading range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the AD Line series and a list of column names.

    The Accumulation/Distribution Line is calculated as follows:

    1. Calculate Money Flow Multiplier (MFM):
       MFM = ((Close - Low) - (High - Close)) / (High - Low)

    2. Calculate Money Flow Volume (MFV):
       MFV = MFM * Volume

    3. Calculate ADL (Cumulative Sum):
       ADL = Previous ADL + Current MFV

    Interpretation:
    - Rising ADL: Accumulation (buying pressure exceeds selling pressure).
    - Falling ADL: Distribution (selling pressure exceeds buying pressure).
    - Divergence: Price making new highs while ADL fails to do so suggests reversal.

    Use Cases:
    - Trend Confirmation: Confirm the strength and sustainability of a price trend.
    - Divergence Detection: Identify potential reversals through price/volume divergence.
    - Volume Analysis: Assess buying/selling pressure independent of price movement.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Handle division by zero - if high and low are the same,
    # money flow multiplier is zero (neutral)
    price_range = high - low
    price_range_nonzero = price_range.replace(0, np.nan)
    
    # Calculate Money Flow Multiplier (MFM)
    # MFM = ((Close - Low) - (High - Close)) / (High - Low)
    # This simplifies to MFM = (2*Close - High - Low) / (High - Low)
    mfm = ((2 * close - high - low) / price_range_nonzero).fillna(0)
    
    # Calculate Money Flow Volume (MFV)
    mfv = mfm * volume
    
    # Calculate ADL as cumulative sum of Money Flow Volume
    ad_line = mfv.cumsum()
    ad_line.name = 'ADL'
    columns_list = [ad_line.name]
    return ad_line, columns_list


def strategy_adl(
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
    adl (Accumulation/Distribution Line) - Trend Confirmation Strategy
    
    LOGIC: Buy when adl crosses above its SMA (accumulation),
           sell when adl crosses below its SMA (distribution).
    WHY: adl measures cumulative money flow. Rising adl indicates buying pressure,
         falling adl indicates selling pressure. Divergence with price signals reversals.
    BEST MARKETS: Stocks, ETFs. Good for confirming price trends with volume.
    TIMEFRAME: Daily charts. Good for swing trading and trend confirmation.
    
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
        indicator='adl',
        parameters={},
        figure=False
    )
    
    # Calculate SMA of ADL for crossover signals
    data[f'ADL_SMA_{sma_period}'] = data['ADL'].rolling(window=sma_period).mean()
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='ADL',
        long_window_indicator=f'ADL_SMA_{sma_period}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['ADL', f'ADL_SMA_{sma_period}']
    
    return results, portfolio, indicator_cols_to_plot, data
