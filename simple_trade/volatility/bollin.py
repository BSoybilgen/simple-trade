import pandas as pd
import numpy as np


def bollinger_bands(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Bollinger Bands of a series.

    Bollinger Bands are a type of statistical chart illustrating the relative high and low prices of a security in relation to its average price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The window size for calculating the moving average and standard deviation. Default is 20.
            - num_std (int): The number of standard deviations to use for the upper and lower bands. Default is 2.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Bollinger Bands DataFrame and a list of column names.

    Bollinger Bands consist of:

    1. A middle band, which is a simple moving average (SMA) of the price.
    2. An upper band, which is the SMA plus a certain number of standard deviations (typically 2).
    3. A lower band, which is the SMA minus the same number of standard deviations.

    Use Cases:

    - Identifying overbought and oversold conditions: Prices near the upper band may indicate overbought conditions, while prices near the lower band may indicate oversold conditions.
    - Identifying volatility: The width of the Bollinger Bands can be used to gauge volatility. Wide bands indicate high volatility, while narrow bands indicate low volatility.
    - Generating buy and sell signals: Some traders use Bollinger Bands to generate buy and sell signals based on price breakouts or reversals near the bands.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = parameters.get('window', 20)
    num_std = parameters.get('num_std', 2)
    close_col = columns.get('close_col', 'Close')
    
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
    columns_list = list(df_bb.columns)
    return df_bb, columns_list