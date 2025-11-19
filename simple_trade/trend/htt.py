import pandas as pd


def htt(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Hilbert Transform Trendline (HTT).

    HTT approximates John Ehlers' Hilbert Transform smoothing by filtering the
    close series with the standard detrender kernel and subtracting it from the
    original price to derive a low-lag trendline.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = max(7, int(parameters.get('window', 16)))  # need at least 7 samples for kernel

    close = df[close_col]
    smoothing = 0.075 * window + 0.54

    detrender = (
        0.0962 * close
        + 0.5769 * close.shift(2)
        - 0.5769 * close.shift(4)
        - 0.0962 * close.shift(6)
    ) * smoothing

    trendline = close - detrender
    trendline.name = f'HTT_{window}'

    return trendline, [trendline.name]
