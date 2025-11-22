import pandas as pd
import numpy as np


def grv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Garman-Klass Volatility, a more efficient volatility estimator that uses
    OHLC data instead of just closing prices. It provides better volatility estimates
    with the same amount of data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 20.
            - trading_periods (int): Trading periods per year for annualization. Default is 252.
            - annualized (bool): Whether to annualize the volatility. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Garman-Klass Volatility series and a list of column names.

    Calculation Steps:
    1. Calculate Log Ratios:
       log_hl = ln(High / Low)
       log_co = ln(Close / Open)

    2. Calculate GK Component for each period:
       GK = 0.5 * (log_hl)^2 - (2 * ln(2) - 1) * (log_co)^2

    3. Average and Square Root:
       GK_Vol = sqrt(Average(GK, period))

    4. Annualize (optional):
       Annualized GK = GK_Vol * sqrt(trading_periods)

    Interpretation:
    - Lower values: Low volatility.
    - Higher values: High volatility.
    - More efficient than close-to-close volatility as it uses intrabar information.

    Use Cases:
    - Superior volatility estimation compared to standard deviation.
    - Options pricing and risk management.
    - More accurate with same data as Historical Volatility.
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
    
    # Calculate log ratios
    log_hl = np.log(high / low)
    log_co = np.log(close / open_price)
    
    # Garman-Klass formula
    # GK = sqrt(0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2)
    gk_component = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    # Take rolling mean and square root
    gk_variance = gk_component.rolling(window=period).mean()
    gk_volatility = np.sqrt(gk_variance)
    
    # Annualize if requested
    if annualized:
        gk_volatility = gk_volatility * np.sqrt(trading_periods)
    
    # Convert to percentage
    grv_values = gk_volatility * 100
    
    if annualized:
        grv_values.name = f'GK_VOL_{period}_Ann'
    else:
        grv_values.name = f'GK_VOL_{period}'
    
    columns_list = [grv_values.name]
    return grv_values, columns_list
