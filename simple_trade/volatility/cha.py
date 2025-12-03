import pandas as pd
import numpy as np
from ..moving_average.ema import ema


def cha(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Chaikin Volatility (cha) indicator, which measures volatility by 
    calculating the rate of change of the high-low price range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - ema_window (int): The window for calculating the EMA of the high-low range. Default is 10.
            - roc_window (int): The window for calculating the rate of change. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Chaikin Volatility series and a list of column names.

    The Chaikin Volatility is calculated as follows:

    1. Calculate Daily Range:
       Range = High - Low

    2. Calculate EMA of Range:
       RangeEMA = EMA(Range, ema_window)

    3. Calculate Rate of Change (ROC):
       CHA = ((RangeEMA - RangeEMA(roc_window periods ago)) / RangeEMA(roc_window periods ago)) * 100

    Interpretation:
    - Higher values indicate increasing volatility (range expansion).
    - Lower values indicate decreasing volatility (range contraction).
    - Peaks in CHA often correlate with market tops or bottoms.

    Use Cases:
    - Volatility measurement: Identifies periods of increasing or decreasing volatility.
    - Market turning points: Rising volatility often precedes tops; falling volatility often precedes bottoms.
    - Breakout confirmation: Sharp increases in volatility can confirm breakouts.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    ema_window = int(parameters.get('ema_window', 10))
    roc_window = int(parameters.get('roc_window', 10))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    
    high = df[high_col]
    low = df[low_col]
    
    # Calculate the daily high-low range
    hl_range = high - low
    df = pd.DataFrame(hl_range, columns=['Close'])

    # Calculate the EMA of the high-low range
    # Create parameters for ema function
    ema_parameters = {'window': ema_window}
    ema_columns = {'close_col': 'Close'}
    range_ema_series, _ = ema(df, parameters=ema_parameters, columns=ema_columns)
    
    # Calculate the percentage rate of change over roc_window days
    # (Current EMA - EMA roc_window days ago) / (EMA roc_window days ago) * 100
    roc = ((range_ema_series - range_ema_series.shift(roc_window)) / range_ema_series.shift(roc_window)) * 100
    roc.name = f'CHA_{ema_window}_{roc_window}'
    columns_list = [roc.name]
    return roc, columns_list


def strategy_cha(
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
    cha (Chaikin Volatility) - Volatility Threshold Strategy
    
    LOGIC: Buy when cha drops below lower threshold (volatility contraction),
           sell when rises above upper threshold (volatility expansion).
    WHY: cha measures rate of change of high-low range.
         Peaks often correlate with market tops or bottoms.
    BEST MARKETS: All markets. Good for identifying volatility regimes.
                  Rising volatility often precedes tops.
    TIMEFRAME: Daily charts. 10-period EMA and ROC is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'ema_window' (default 10), 'roc_window' (default 10),
                    'upper' (default 50), 'lower' (default -50)
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
    
    ema_window = int(parameters.get('ema_window', 10))
    roc_window = int(parameters.get('roc_window', 10))
    upper = float(parameters.get('upper', 50))
    lower = float(parameters.get('lower', -50))
    price_col = 'Close'
    indicator_col = f'CHA_{ema_window}_{roc_window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='cha',
        parameters={"ema_window": ema_window, "roc_window": roc_window},
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
