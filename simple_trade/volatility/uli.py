import pandas as pd
import numpy as np


def uli(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ulcer Index (UI), a volatility indicator that measures downside risk
    by focusing on the depth and duration of price drawdowns from recent peaks, rather
    than treating upside and downside volatility equally.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculations. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Ulcer Index series and a list of column names.

    Calculation Steps:
    1. Calculate Percentage Drawdown:
       For each period, find the highest close over the lookback window.
       Drawdown = 100 * (Close - Highest Close) / Highest Close

    2. Calculate Squared Drawdowns:
       Squared Drawdown = (Drawdown)^2

    3. Calculate Mean Squared Drawdown:
       Mean = Sum(Squared Drawdowns) / period

    4. Calculate Ulcer Index:
       UI = sqrt(Mean)

    Interpretation:
    - Higher UI values indicate greater downside risk and deeper/longer drawdowns.
    - Lower UI values indicate stability or upward trends.
    - Unlike standard deviation, UI does not penalize upside volatility.

    Use Cases:
    - Downside risk measurement.
    - Portfolio optimization (e.g., Ulcer Performance Index).
    - Drawdown monitoring and risk management.
    - Comparison of strategies with similar returns but different risk profiles.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 14))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate the highest close over the rolling period
    highest_close = close.rolling(window=period).max()
    
    # Calculate percentage drawdown from the highest close
    # Drawdown % = 100 * (Close - Highest Close) / Highest Close
    drawdown_pct = 100 * (close - highest_close) / highest_close
    
    # Square each drawdown percentage
    squared_drawdown = drawdown_pct ** 2
    
    # Calculate the mean of squared drawdowns over the period
    mean_squared_drawdown = squared_drawdown.rolling(window=period).mean()
    
    # Take the square root to get the Ulcer Index
    ulcer_index = np.sqrt(mean_squared_drawdown)
    
    ulcer_index.name = f'UI_{period}'
    columns_list = [ulcer_index.name]
    return ulcer_index, columns_list
