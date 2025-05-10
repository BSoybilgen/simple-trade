import pandas as pd
import numpy as np


def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2, close_col: str = 'Close') -> pd.DataFrame:
    """
    Calculates Bollinger Bands of a series.

    Bollinger Bands are a type of statistical chart illustrating the relative high and low prices of a security in relation to its average price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window (int): The window size for calculating the moving average and standard deviation.
        num_std (int): The number of standard deviations to use for the upper and lower bands.
        close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        pd.DataFrame: A DataFrame containing the middle band (SMA), upper band, and lower band.

    Bollinger Bands consist of:

    1. A middle band, which is a simple moving average (SMA) of the price.
    2. An upper band, which is the SMA plus a certain number of standard deviations (typically 2).
    3. A lower band, which is the SMA minus the same number of standard deviations.

    Use Cases:

    - Identifying overbought and oversold conditions: Prices near the upper band may indicate overbought conditions, while prices near the lower band may indicate oversold conditions.
    - Identifying volatility: The width of the Bollinger Bands can be used to gauge volatility. Wide bands indicate high volatility, while narrow bands indicate low volatility.
    - Generating buy and sell signals: Some traders use Bollinger Bands to generate buy and sell signals based on price breakouts or reversals near the bands.
    """
    series = df[close_col]

    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    # Return DataFrame for multi-output indicators
    df_bb = pd.DataFrame({
        f'BB_Middle_{window}': sma,
        f'BB_Upper_{window}_{num_std}': upper_band,
        f'BB_Lower_{window}_{num_std}': lower_band
    })
    # Ensure index is passed explicitly, just in case
    df_bb.index = series.index
    return df_bb