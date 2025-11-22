import pandas as pd


def eac(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ehlers Adaptive CyberCycle (EAC) indicator.
    The Adaptive CyberCycle is based on John Ehlers' work on cycle analysis.
    It adapts to the dominant market cycle and provides a smooth, low-lag
    trend indicator that oscillates around the price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - alpha (float): The smoothing factor. Default is 0.07.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the EAC series and a list of column names.

    The Ehlers Adaptive CyberCycle is calculated as follows:

    1. Smooth the Price Data:
       Smooth = (Price + 2*Price[1] + 2*Price[2] + Price[3]) / 6

    2. Calculate Cycle:
       Cycle[i] = (1 - 0.5*alpha)^2 * (Smooth[i] - 2*Smooth[i-1] + Smooth[i-2]) 
                  + 2*(1 - alpha)*Cycle[i-1] 
                  - (1 - alpha)^2 * Cycle[i-2]

    3. Calculate Trend Line:
       Trend = Smooth - Cycle

    Interpretation:
    - The indicator separates the cycle component from the trend component.
    - It provides a very smooth trendline with minimal lag.

    Use Cases:
    - Trend Following: The resulting trendline tracks the underlying trend.
    - Cycle Analysis: The removed cycle component can be analyzed separately.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    alpha = float(parameters.get('alpha', 0.07))

    series = df[close_col].copy()
    
    # Smooth the price data
    smooth = pd.Series(index=series.index, dtype=float)
    smooth.iloc[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        if i < 3:
            smooth.iloc[i] = series.iloc[i]
        else:
            # Simple 4-bar average for smoothing
            smooth.iloc[i] = (series.iloc[i] + 2*series.iloc[i-1] + 
                             2*series.iloc[i-2] + series.iloc[i-3]) / 6.0
    
    # Initialize cycle
    cycle = pd.Series(index=series.index, dtype=float)
    cycle.iloc[:2] = 0.0
    
    # Calculate CyberCycle using recursive filter
    for i in range(2, len(series)):
        if pd.notna(smooth.iloc[i]) and pd.notna(smooth.iloc[i-1]) and pd.notna(smooth.iloc[i-2]):
            # Ehlers CyberCycle formula (simplified)
            cycle.iloc[i] = ((1 - 0.5*alpha)**2 * (smooth.iloc[i] - 2*smooth.iloc[i-1] + smooth.iloc[i-2]) + 
                            2*(1 - alpha)*cycle.iloc[i-1] - 
                            (1 - alpha)**2 * cycle.iloc[i-2])
        else:
            cycle.iloc[i] = 0.0
    
    # Create the trend line by subtracting cycle from smoothed price
    trend = smooth - cycle
    trend.name = f'EAC_{int(alpha*100)}'
    
    columns_list = [trend.name]
    return trend, columns_list
