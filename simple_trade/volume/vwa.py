import pandas as pd
import numpy as np


def vwa(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Weighted Average Price (VWAP), a trading benchmark
    that gives the average price a security has traded at throughout the day (or dataset),
    based on both volume and price.

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
        tuple: A tuple containing the VWAP series and a list of column names.

    The Volume Weighted Average Price is calculated as follows:

    1. Calculate Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate TPV (Typical Price * Volume):
       TPV = TP * Volume

    3. Calculate Cumulative TPV:
       CumTPV = CumulativeSum(TPV)

    4. Calculate Cumulative Volume:
       CumVol = CumulativeSum(Volume)

    5. Calculate VWAP:
       VWAP = CumTPV / CumVol

    Interpretation:
    - Price > VWAP: Bullish sentiment (Buyers in control).
    - Price < VWAP: Bearish sentiment (Sellers in control).
    - Benchmark: Acts as a measure of "fair value" for the period.

    Use Cases:
    - Intraday Trading: Assessing if price is expensive or cheap relative to the day's average.
    - Trade Execution: Benchmarking trade fills.
    - Support/Resistance: VWAP often acts as a magnet or dynamic level.
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
    
    # Calculate Typical Price
    typical_price = (high + low + close) / 3
    
    # Calculate cumulative TPV (Typical Price * Volume)
    cumulative_tpv = (typical_price * volume).cumsum()
    
    # Calculate cumulative volume
    cumulative_volume = volume.cumsum()
    
    # Calculate VWAP (handle division by zero)
    vwap_values = cumulative_tpv / cumulative_volume.replace(0, np.nan)
    vwap_values = vwap_values.fillna(method='ffill').fillna(0)
    
    vwap_values.name = 'VWAP'
    columns_list = [vwap_values.name]
    return vwap_values, columns_list


def strategy_vwa(
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
    VWA (Volume Weighted Average Price) - Price vs VWAP Crossover Strategy
    
    LOGIC: Buy when price crosses above VWAP (bullish sentiment),
           sell when price crosses below VWAP (bearish sentiment).
    WHY: VWAP is a benchmark for "fair value". Price above VWAP indicates
         buyers in control, below indicates sellers in control.
    BEST MARKETS: Stocks, ETFs. Good for intraday and swing trading.
    TIMEFRAME: Intraday or daily charts. Acts as dynamic support/resistance.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict (no parameters used)
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
    
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='vwa',
        parameters={},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='Close',
        long_window_indicator='VWAP',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['Close', 'VWAP']
    
    return results, portfolio, indicator_cols_to_plot, data
