import pandas as pd
import numpy as np


def hiv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Historical Volatility (HV), also known as realized volatility, which measures
    the actual volatility of an asset's returns over a historical period. It is the annualized
    standard deviation of logarithmic returns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculations. Default is 20.
            - trading_periods (int): Number of trading periods per year for annualization.
                                    Default is 252 (trading days per year).
                                    Use 365 for crypto, 52 for weekly data, 12 for monthly.
            - annualized (bool): Whether to annualize the volatility. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Historical Volatility series and a list of column names.

    Calculation Steps:
    1. Calculate Logarithmic Returns:
       Log Return = ln(Close[t] / Close[t-1])
    2. Calculate Standard Deviation:
       Volatility = std(Log Returns, period)
    3. Annualize (if annualized=True):
       Annualized HV = Volatility * sqrt(trading_periods)
    4. Convert to Percentage:
       HV (%) = Annualized HV * 100

    Interpretation:
    - Low HV (<10%): Low volatility, stable price movements.
    - Medium HV (10-20%): Normal volatility.
    - High HV (20-30%): Elevated volatility.
    - Very High HV (>30%): Extreme volatility.

    Use Cases:
    - Options pricing: Comparison with Implied Volatility (IV).
    - Risk measurement: Quantifying statistical risk.
    - Position sizing: Adjusting exposure based on realized volatility.
    - Volatility regime identification: Switching strategies based on HV levels.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 20))
    trading_periods = int(parameters.get('trading_periods', 252))
    annualized = bool(parameters.get('annualized', True))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate logarithmic returns
    # ln(Close[t] / Close[t-1]) = ln(Close[t]) - ln(Close[t-1])
    log_returns = np.log(close / close.shift(1))
    
    # Calculate rolling standard deviation of log returns
    volatility = log_returns.rolling(window=period).std()
    
    # Annualize the volatility if requested
    if annualized:
        volatility = volatility * np.sqrt(trading_periods)
    
    # Convert to percentage
    hv_values = volatility * 100
    
    # Create appropriate name based on whether it's annualized
    if annualized:
        hv_values.name = f'HV_{period}_Ann'
    else:
        hv_values.name = f'HV_{period}'
    
    columns_list = [hv_values.name]
    return hv_values, columns_list
