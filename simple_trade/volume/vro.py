import pandas as pd
import numpy as np


def vro(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Rate of Change (VROC), a momentum indicator that measures
    the rate of change in volume over a specified period. It highlights significant
    volume increases or decreases.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for VROC calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VROC series and a list of column names.

    The Volume Rate of Change is calculated as follows:

    1. Identify Past Volume:
       Past Volume = Volume(n periods ago)

    2. Calculate VROC:
       VROC = ((Current Volume - Past Volume) / Past Volume) * 100

    Interpretation:
    - Positive VROC: Volume is increasing.
    - Negative VROC: Volume is decreasing.
    - High VROC: High trading activity/volatility.
    - Low VROC: Low trading activity/consolidation.

    Use Cases:
    - Breakout Validation: Breakouts should be accompanied by a surge in VROC.
    - Trend Strength: Rising VROC confirms trend participation.
    - Reversal Warning: Divergence between Price and VROC.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = parameters.get('period', 14)
    volume_col = columns.get('volume_col', 'Volume')
    
    volume = df[volume_col]
    
    # Calculate Volume n periods ago
    volume_n_periods_ago = volume.shift(period)
    
    # Calculate VROC (handle division by zero)
    vroc_values = ((volume - volume_n_periods_ago) / volume_n_periods_ago.replace(0, np.nan)) * 100
    
    vroc_values.name = f'VROC_{period}'
    columns_list = [vroc_values.name]
    return vroc_values, columns_list
