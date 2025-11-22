import numpy as np
import pandas as pd


def fis(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Fisher Transform, a technical indicator created by John Ehlers 
    that converts prices into a Gaussian normal distribution.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for normalization. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Fisher Transform series and a list of column names.

    The Fisher Transform is calculated in several steps:

    1. Calculate Median Price:
       Median = (High + Low) / 2

    2. Normalize Median Price (over window):
       Value = 2 * ((Median - Min) / (Max - Min) - 0.5)
       (Value is clipped to stay within boundaries close to -1 and 1)

    3. Smooth the Normalized Value (optional/specific to implementation):
       Value = 0.33 * Value + 0.67 * Previous Value

    4. Calculate Fisher Transform:
       Fisher = 0.5 * ln((1 + Value) / (1 - Value)) + 0.5 * Previous Fisher

    Interpretation:
    - Sharp Turning Points: The Fisher Transform creates sharp peaks and troughs, making turning points clearer.
    - Extremes: Extreme positive values indicate overbought conditions; extreme negative values indicate oversold.
    - Crossings: Crossing the signal line (often the previous value or a moving average of Fisher) can generate signals.

    Use Cases:
    - Reversal Detection: Identifying precise reversal points in the market price.
    - Trend Analysis: Assessing the direction and strength of the trend.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 9))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    high = df[high_col]
    low = df[low_col]
    median_price = (high + low) / 2

    highest_high = median_price.rolling(window=window).max()
    lowest_low = median_price.rolling(window=window).min()
    price_range = (highest_high - lowest_low).replace(0, pd.NA)

    normalized = 2 * ((median_price - lowest_low) / price_range - 0.5)
    normalized = normalized.clip(-0.999, 0.999)

    fisher_values = pd.Series(index=df.index, dtype=float)
    fisher_values.name = f'FISH_{window}'

    prev_value = 0.0

    for idx in range(len(df)):
        value = normalized.iat[idx]
        if pd.isna(value):
            fisher_values.iat[idx] = np.nan
            continue

        value = 0.33 * value + 0.67 * prev_value
        value = max(min(value, 0.999), -0.999)

        fisher = 0.5 * np.log((1 + value) / (1 - value))
        fisher_values.iat[idx] = fisher
        prev_value = value

    columns_list = [fisher_values.name]
    return fisher_values, columns_list
