import numpy as np
import pandas as pd


def msi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Momentum Strength Index (MSI), a custom momentum indicator.
    MSI aims to quantify the strength of price momentum by comparing recent gains 
    and losses, with an optional power parameter to emphasize stronger moves.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): Rolling window for averaging gains/losses. Default is 14.
            - power (float): Exponent applied to gains/losses to accentuate strong moves. Default is 1.0.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the MSI series and a list of column names.

    The Momentum Strength Index is calculated as follows:

    1. Calculate Price Differences:
       Diff = Close - Close(prev)

    2. Separate and Power-Scale Gains and Losses:
       Gain = (Diff if Diff > 0 else 0) ^ power
       Loss = (Abs(Diff) if Diff < 0 else 0) ^ power

    3. Calculate Average Gain and Loss:
       Avg Gain = SMA(Gain, window)
       Avg Loss = SMA(Loss, window)

    4. Calculate MSI:
       Strength Ratio = Avg Gain / Avg Loss
       MSI = 100 * Strength Ratio / (1 + Strength Ratio)

    Interpretation:
    - Range: 0 to 100.
    - Power Parameter: If power > 1, large moves have a disproportionately larger effect on the index, making it more sensitive to volatility spikes.
    - High Values: Strong upside momentum.
    - Low Values: Strong downside momentum.

    Use Cases:
    - Volatility-Adjusted Momentum: Using power > 1 allows traders to filter out low-volatility noise and focus on high-momentum moves.
    - Overbought/Oversold: Similar to RSI, can identify extremes.
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
