import pandas as pd
import numpy as np


def pav(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Parkinson Volatility, an efficient volatility estimator that uses only
    the high-low range. It's more efficient than close-to-close volatility when there
    are no overnight gaps.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
            - trading_periods (int): Trading periods per year. Default is 252.
            - annualized (bool): Whether to annualize. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): High prices column. Default is 'High'.
            - low_col (str): Low prices column. Default is 'Low'.

    Returns:
        tuple: Parkinson Volatility series and column names list.

    Calculation Steps:
    1. Calculate Log Ratios:
       log_hl = ln(High / Low)
    2. Calculate Parkinson Component:
       Comp = (1 / (4 * ln(2))) * (log_hl)^2
    3. Average and Square Root:
       Vol = sqrt(Average(Comp, period))
    4. Annualize (optional):
       Parkinson = Vol * sqrt(trading_periods) * 100

    Interpretation:
    - Lower values: Low volatility.
    - Higher values: High volatility.
    - Uses intraday range, so it captures volatility missed by close-to-close measures.

    Use Cases:
    - Efficient volatility estimation using high-low range.
    - Better than standard deviation when no overnight gaps.
    - Risk management and options pricing.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    trading_periods = int(parameters.get('trading_periods', 252))
    annualized = bool(parameters.get('annualized', True))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    
    high = df[high_col]
    low = df[low_col]
    
    # Parkinson formula
    log_hl = np.log(high / low)
    parkinson_component = (1 / (4 * np.log(2))) * (log_hl ** 2)
    
    # Rolling mean and square root
    pav_variance = parkinson_component.rolling(window=period).mean()
    pav_volatility = np.sqrt(pav_variance)
    
    if annualized:
        pav_volatility = pav_volatility * np.sqrt(trading_periods)
    
    pav_values = pav_volatility * 100
    
    pav_values.name = f'PARK_VOL_{period}_Ann' if annualized else f'PARK_VOL_{period}'
    return pav_values, [pav_values.name]
