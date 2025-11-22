import pandas as pd


def std(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Standard Deviation (STD), a statistical measure of volatility that quantifies
    the amount of variation or dispersion in a set of price values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
            - ddof (int): Delta Degrees of Freedom. Default is 0 (population std).
                         Use 1 for sample standard deviation.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Standard Deviation series and a list of column names.

    Calculation Steps:
    1. Calculate Mean:
       Calculate the simple moving average over the window.
    2. Calculate Variance:
       Sum of squared differences from the mean / N (or N-ddof).
    3. Calculate STD:
       Square root of Variance.

    Interpretation:
    - High STD: High volatility, prices are spread out.
    - Low STD: Low volatility, prices are close to the mean.
    - Rapid rise in STD often indicates market instability or breakout.

    Use Cases:
    - Volatility measurement.
    - Risk assessment (Sharpe Ratio).
    - Bollinger Bands component.
    - Position sizing adjustment.
    - Mean reversion strategies (z-scores).

    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    ddof = int(parameters.get('ddof', 0))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling standard deviation
    std_values = close.rolling(window=window).std(ddof=ddof)
    
    std_values.name = f'STD_{window}'
    columns_list = [std_values.name]
    return std_values, columns_list
