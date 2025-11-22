import pandas as pd


def vsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volatility Switch Index (VSI), a binary indicator that identifies
    volatility regime changes by comparing current volatility to historical levels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_period (int): Short volatility period. Default is 10.
            - long_period (int): Long volatility period. Default is 50.
            - threshold (float): Threshold for regime switch. Default is 1.2.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: VSI series and column names list.

    Calculation Steps:
    1. Calculate Short-Term Volatility:
       Short Vol = Standard Deviation(Close, short_period)
    2. Calculate Long-Term Volatility:
       Long Vol = Standard Deviation(Close, long_period)
    3. Calculate Ratio:
       Ratio = Short Vol / Long Vol
    4. Determine VSI:
       If Ratio > threshold, VSI = 1 (High Volatility Regime)
       Else, VSI = 0 (Low/Normal Volatility Regime)

    Interpretation:
    - VSI = 1: Market is experiencing elevated volatility compared to its history.
    - VSI = 0: Market is in a baseline volatility state.
    - Switches often accompany changes in market trend or condition.

    Use Cases:
    - Binary volatility regime identification.
    - Strategy switching (High Vol vs. Low Vol strategies).
    - Risk management (Reducing size when VSI is 1).
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    short_period = int(parameters.get('short_period', 10))
    long_period = int(parameters.get('long_period', 50))
    threshold = float(parameters.get('threshold', 1.2))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate short and long-term volatility
    short_vol = close.rolling(window=short_period).std()
    long_vol = close.rolling(window=long_period).std()
    
    # Calculate volatility ratio
    vol_ratio = short_vol / long_vol
    
    # Create binary switch
    vsi_values = (vol_ratio > threshold).astype(int)
    
    vsi_values.name = f'VSI_{short_period}_{long_period}_{threshold}'
    return vsi_values, [vsi_values.name]
