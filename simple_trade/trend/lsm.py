import numpy as np
import pandas as pd


def lsm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Least Squares Moving Average (LSMA).
    LSMA is the end point of a linear regression line over a rolling window.
    It provides a statistically-based trend line that minimizes the sum of
    squared errors between the line and the actual prices.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the LSMA series and a list of column names.

    The Least Squares Moving Average is calculated as follows:

    1. Perform Linear Regression over the rolling window:
       y = mx + c
       where y is price and x is time (0 to window-1).

    2. Calculate the Forecast value at the end of the window:
       LSMA = m * (window - 1) + c

    Interpretation:
    - LSMA fits the data better than SMA or EMA but can overshoot in sudden reversals.
    - Ideally suited for identifying the direction of the primary trend.

    Use Cases:
    - Trend Confirmation: If price is above LSMA, trend is up.
    - Reversal Signal: Price crossing LSMA suggests a potential trend change.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]

    def linear_regression_forecast(x):
        """Calculate the forecast value from linear regression."""
        if len(x) < 2:
            return np.nan
        
        # Create time index
        t = np.arange(len(x))
        
        # Calculate linear regression coefficients
        # y = a + b*t
        n = len(x)
        sum_t = np.sum(t)
        sum_x = np.sum(x)
        sum_tx = np.sum(t * x)
        sum_t2 = np.sum(t * t)
        
        # Calculate slope (b) and intercept (a)
        b = (n * sum_tx - sum_t * sum_x) / (n * sum_t2 - sum_t * sum_t)
        a = (sum_x - b * sum_t) / n
        
        # Return forecast for the next point (end of window)
        return a + b * (n - 1)

    lsma_series = series.rolling(window=window).apply(linear_regression_forecast, raw=True)
    lsma_series.name = f'LSMA_{window}'

    columns_list = [lsma_series.name]
    return lsma_series, columns_list
