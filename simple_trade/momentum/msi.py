import numpy as np
import pandas as pd


def msi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Momentum Strength Index (MSI).

    MSI aims to quantify the strength of price momentum by comparing recent gains
    and losses similar to RSI, but with optional sensitivity tuning via a power
    parameter that emphasizes stronger moves.

    Args:
        df (pd.DataFrame): Input DataFrame containing price data.
        parameters (dict, optional): Calculation parameters.
            - window (int): Rolling window for averaging gains/losses. Default 14.
            - power (float): Exponent applied to gains/losses to accentuate strong moves. Default 1.0.
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices. Default 'Close'.

    Returns:
        tuple: (msi_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    power = float(parameters.get('power', 1.0))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    delta = series.diff()

    gains = delta.clip(lower=0).pow(power)
    losses = (-delta.clip(upper=0)).pow(power)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    strength_ratio = avg_gain / avg_loss.replace({0: np.nan})
    msi_values = 100 * strength_ratio / (1 + strength_ratio)
    msi_values = msi_values.astype(float)
    msi_values.name = f'MSI_{window}_{power}'

    columns_list = [msi_values.name]
    return msi_values, columns_list
