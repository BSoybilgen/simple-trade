import numpy as np
import pandas as pd


def fis(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Fisher Transform momentum indicator.

    Args:
        df (pd.DataFrame): Input price data containing high and low columns.
        parameters (dict, optional): Calculation parameters.
            - window (int): Lookback period for normalization. Default is 9.
        columns (dict, optional): Column overrides.
            - high_col (str): Column name for high prices. Default is 'High'.
            - low_col (str): Column name for low prices. Default is 'Low'.

    Returns:
        tuple: (fisher_series, [column_name])
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
