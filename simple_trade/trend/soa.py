import pandas as pd


def soa(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Smoothed Moving Average (SmMA) of a series.

    The Smoothed Moving Average is similar to Wilder's moving average and applies
    heavier smoothing than a simple moving average. Each new value is a blend of
    the previous SmMA and the current price, helping highlight longer-term
    trends while filtering short-term volatility.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have a close column.
        parameters (dict): Dictionary with the window size for the SmMA calculation.
        columns (dict): Dictionary specifying the close column name.

    Returns:
        tuple: A tuple containing the SmMA series and a list with its column name.

    The SmMA is computed iteratively using the formula:

        SmMA_t = (SmMA_{t-1} * (n - 1) + price_t) / n

    This can be efficiently implemented with an exponential weighted mean using
    alpha = 1 / n and adjust=False, which matches the Wilder smoothing method.

    Use cases mirror those of other moving averages but emphasize smoother trend
    detection with reduced lag compared to a simple moving average of the same
    length.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]
    smoothed = series.ewm(alpha=1 / window, adjust=False).mean()
    smoothed.name = f'SOA_{window}'

    columns_list = [smoothed.name]

    return smoothed, columns_list
