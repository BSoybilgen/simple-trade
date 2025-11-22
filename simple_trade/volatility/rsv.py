import pandas as pd
import numpy as np


def rsv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Rogers-Satchell Volatility, which accounts for drift in price movements
    and handles trending markets better than Garman-Klass. Uses all OHLC components.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
            - trading_periods (int): Trading periods per year. Default is 252.
            - annualized (bool): Whether to annualize. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): Open prices column. Default is 'Open'.
            - high_col (str): High prices column. Default is 'High'.
            - low_col (str): Low prices column. Default is 'Low'.
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: Rogers-Satchell Volatility series and column names list.

    Calculation Steps:
    1. Calculate Log Ratios:
       log_hc = ln(High / Close)
       log_ho = ln(High / Open)
       log_lc = ln(Low / Close)
       log_lo = ln(Low / Open)

    2. Calculate RS Component:
       RS = log_hc * log_ho + log_lc * log_lo

    3. Average and Square Root:
       Vol = sqrt(Average(RS, period))

    4. Annualize (optional):
       RSV = Vol * sqrt(trading_periods) * 100

    Interpretation:
    - Lower values: Low volatility.
    - Higher values: High volatility.
    - Better for trending markets as it allows for non-zero drift.

    Use Cases:
    - Volatility estimation in trending markets.
    - Options pricing and risk management.
    - Advanced volatility analysis.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    trading_periods = int(parameters.get('trading_periods', 252))
    annualized = bool(parameters.get('annualized', True))
    open_col = columns.get('open_col', 'Open')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    open_price = df[open_col]
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Rogers-Satchell formula
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_price)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_price)
    
    rs_component = log_hc * log_ho + log_lc * log_lo
    
    # Rolling mean and square root
    rs_variance = rs_component.rolling(window=period).mean()
    rs_volatility = np.sqrt(rs_variance.abs())
    
    if annualized:
        rs_volatility = rs_volatility * np.sqrt(trading_periods)
    
    rsv_values = rs_volatility * 100
    
    rsv_values.name = f'RS_VOL_{period}_Ann' if annualized else f'RS_VOL_{period}'
    return rsv_values, [rsv_values.name]
