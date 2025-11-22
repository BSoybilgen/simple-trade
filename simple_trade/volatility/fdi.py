import pandas as pd
import numpy as np


def fdi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Fractal Dimension Index (FDI), which measures market complexity
    and choppiness based on fractal geometry. Values near 1.5 indicate random walk.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: FDI series and column names list.

    The FDI is calculated as follows:

    1. Calculate Path Length:
       Sum of absolute differences between consecutive prices over the window.

    2. Calculate Direct Distance:
       Absolute difference between first and last price in the window.

    3. Calculate Fractal Dimension:
       FDI = 1 + (log(Path Length) - log(Direct Distance)) / log(window)

    Interpretation:
    - FDI near 1.0: Highly persistent, trending (Linear).
    - FDI near 1.5: Random walk, no clear pattern (Brownian motion).
    - FDI near 2.0: Anti-persistent, mean-reverting (Jagged).

    Use Cases:
    - Market complexity measurement.
    - Trend vs. random identification.
    - Strategy selection based on market structure.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    def calculate_fdi(window):
        if len(window) < 2:
            return 1.5
        
        n = len(window)
        prices = window.values
        
        # Calculate path length
        path_length = np.sum(np.abs(np.diff(prices)))
        
        # Calculate direct distance
        direct_distance = abs(prices[-1] - prices[0])
        
        if direct_distance == 0 or path_length == 0:
            return 1.5
        
        # Fractal dimension approximation
        # FD = log(path_length) / log(direct_distance)
        # Normalized to 1-2 range
        fd = 1 + (np.log(path_length) - np.log(direct_distance)) / np.log(n)
        
        # Clip to reasonable range
        fd = np.clip(fd, 1.0, 2.0)
        
        return fd
    
    fdi_values = close.rolling(window=period).apply(calculate_fdi, raw=False)
    
    fdi_values.name = f'FDI_{period}'
    return fdi_values, [fdi_values.name]
