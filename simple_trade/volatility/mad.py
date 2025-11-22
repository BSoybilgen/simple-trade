import pandas as pd


def mad(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Median Absolute Deviation (MAD), a robust volatility measure that
    is less sensitive to outliers than standard deviation. It measures dispersion
    around the median rather than the mean.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
            - scale_factor (float): Scaling factor for normal distribution. Default is 1.4826.

        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: MAD series and column names list.

    Calculation Steps:
    1. Calculate Median:
       Find the median of returns over the rolling window.

    2. Calculate Absolute Deviations:
       AbsDev = |Return - Median|

    3. Calculate MAD:
       MAD = Median(AbsDev) * scale_factor

    Interpretation:
    - Low MAD: Low volatility, stable prices.
    - High MAD: High volatility, large price swings.
    - More robust to outliers than standard deviation.

    Use Cases:
    - Robust volatility measurement for non-normal distributions.
    - Outlier-resistant risk metrics.
    - Better for data with extreme values.

    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    scale_factor = float(parameters.get('scale_factor', 1.4826))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate returns
    returns = close.pct_change()
    
    def calculate_mad(window):
        if len(window) < 2:
            return 0
        median_return = window.median()
        abs_deviations = (window - median_return).abs()
        mad_value = abs_deviations.median() * scale_factor
        return mad_value
    
    mad_values = returns.rolling(window=period).apply(calculate_mad, raw=False)
    
    # Convert to percentage
    mad_values = mad_values * 100
    
    mad_values.name = f'MAD_{period}'
    return mad_values, [mad_values.name]
