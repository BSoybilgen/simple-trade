import pandas as pd


def vhf(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Vertical Horizontal Filter (VHF), which determines whether prices
    are in a trending phase or a congestion phase by comparing the range of prices
    to the sum of price changes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 28.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the VHF series and a list of column names.

    Calculation Steps:
    1. Find Extremes:
       HCP = Highest Close over period
       LCP = Lowest Close over period

    2. Calculate Numerator:
       Numerator = HCP - LCP (Price Range)

    3. Calculate Denominator:
       Denominator = Sum of absolute price changes (|Close - Previous Close|) over period.

    4. Calculate VHF:
       VHF = Numerator / Denominator

    Interpretation:
    - High VHF (> 0.40): Strong trending phase (up or down).
    - Low VHF (< 0.25): Congestion or choppy phase.
    - Rising VHF: Developing trend.
    - Falling VHF: Entering congestion.

    Use Cases:
    - Trend Identification: Determine if the market is suitable for trend-following strategies.
    - Indicator Selection: Use moving averages when VHF is high; use oscillators when VHF is low.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 28))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate highest and lowest close over period
    highest_close = close.rolling(window=period).max()
    lowest_close = close.rolling(window=period).min()
    
    # Calculate numerator (range)
    numerator = highest_close - lowest_close
    
    # Calculate denominator (sum of absolute changes)
    price_changes = close.diff().abs()
    denominator = price_changes.rolling(window=period).sum()
    
    # Calculate VHF
    vhf_values = numerator / denominator
    
    vhf_values.name = f'VHF_{period}'
    columns_list = [vhf_values.name]
    return vhf_values, columns_list
