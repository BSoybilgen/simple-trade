import numpy as np
import pandas as pd


def eit(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ehlers Instantaneous Trendline (EIT).
    EIT applies John Ehlers' technique of averaging a weighted price input and
    recursively smoothing it with a tunable alpha to obtain a low-lag trend estimate.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - alpha (float): The smoothing factor (0.01 to 1.0). Default is 0.07.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the EIT series and a list of column names.

    The Ehlers Instantaneous Trendline is calculated as follows:

    1. Calculate Weighted Price:
       Weighted = (Price + 2*Price[1] + Price[2]) / 4

    2. Calculate Instantaneous Trend (Recursive):
       IT[i] = (alpha * Weighted[i]) + ((1 - alpha) * IT[i-1])

    Interpretation:
    - Provides a very smooth trendline that tracks price closely.
    - Low lag compared to simple moving averages.

    Use Cases:
    - Trend Definition: Clear visualization of the current trend.
    - Crossovers: Price crossing EIT or EIT crossing another MA.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    alpha = float(parameters.get('alpha', 0.07))
    alpha = min(max(alpha, 0.01), 1.0)

    close = df[close_col].astype(float)
    weighted_price = (close + 2 * close.shift(1) + close.shift(2)) / 4

    values = weighted_price.to_numpy(dtype=float)
    itrend = np.full_like(values, np.nan)

    valid_idx = np.where(~np.isnan(values))[0]
    if valid_idx.size == 0:
        series = pd.Series(itrend, index=close.index, name=f'EIT_{alpha}')
        return series, [series.name]

    start = valid_idx[0]
    itrend[start] = values[start]

    for i in range(start + 1, len(values)):
        price_term = values[i]
        prev = itrend[i - 1]

        if np.isnan(price_term):
            itrend[i] = prev
            continue

        if np.isnan(prev):
            prev = price_term

        itrend[i] = alpha * price_term + (1 - alpha) * prev

    series = pd.Series(itrend, index=close.index, name=f'EIT_{alpha}')
    return series, [series.name]
