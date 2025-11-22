import pandas as pd
import numpy as np


def aro(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Aroon indicator, which measures the time it takes for a security
    to reach its highest and lowest points over a specified time period.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
    
    Returns:
        tuple: A tuple containing the Aroon DataFrame (Aroon Up, Aroon Down, Oscillator) and a list of column names.
    
    The Aroon indicator is calculated as follows:
    
    1. Calculate Aroon Up:
       Measures periods since the highest high within the lookback period.
       Aroon Up = ((period - periods since highest high) / period) * 100
    
    2. Calculate Aroon Down:
       Measures periods since the lowest low within the lookback period.
       Aroon Down = ((period - periods since lowest low) / period) * 100
       
    3. Calculate Aroon Oscillator:
       Aroon Oscillator = Aroon Up - Aroon Down
    
    Interpretation:
    - Aroon Up > 70: Strong uptrend.
    - Aroon Down > 70: Strong downtrend.
    - Aroon Up/Down < 30: Weak trend.
    - Crossovers: Aroon Up crossing above Aroon Down signals potential bullish trend.
    
    Use Cases:
    - Trend identification: Determine the direction and strength of the current trend.
    - Consolidation detection: When both Aroon Up and Down are low (< 50), it suggests price consolidation.
    - Breakout confirmation: A strong move in Aroon Up/Down can confirm a price breakout.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    period = int(parameters.get('period', 14))

    high = df[high_col]
    low = df[low_col]
    
    # Create aroon_up and aroon_down series
    aroon_up = pd.Series(index=high.index, dtype=float)
    aroon_down = pd.Series(index=low.index, dtype=float)
    
    # Calculate Aroon indicators for each rolling window
    for i in range(len(high) - period + 1):
        # Get the current window
        high_window = high.iloc[i:i+period]
        low_window = low.iloc[i:i+period]
        
        # Find the highest high and lowest low
        highest_high = high_window.max()
        lowest_low = low_window.min()
        
        # Find the periods since highest high and lowest low
        periods_since_highest = period - 1 - high_window.values.tolist().index(highest_high)
        periods_since_lowest = period - 1 - low_window.values.tolist().index(lowest_low)
        
        # Calculate Aroon Up and Aroon Down
        aroon_up.iloc[i+period-1] = ((period - periods_since_highest) / period) * 100
        aroon_down.iloc[i+period-1] = ((period - periods_since_lowest) / period) * 100
    
    # Calculate Aroon Oscillator
    aroon_oscillator = aroon_up - aroon_down

    df_aroon = pd.DataFrame({
        f'AROON_UP_{period}': aroon_down,
        f'AROON_DOWN_{period}': aroon_up,
        f'AROON_OSCILLATOR_{period}': aroon_oscillator
    })
    df_aroon.index = high.index

    columns = list(df_aroon.columns)

    return df_aroon, columns
