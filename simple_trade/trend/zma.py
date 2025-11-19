import pandas as pd


def zma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Zero-Lag Moving Average (ZLEMA/ZMA) of a series.

    ZMA reduces lag by pre-adjusting the input series with its own momentum
    before applying an EMA. This keeps responsiveness similar to shorter EMAs
    while preserving smoothness.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = max(1, int(parameters.get('window', 20)))
    lag = max(1, window // 2)

    price = df[close_col]
    adjusted_price = price + (price - price.shift(lag))
    zlema_series = adjusted_price.ewm(span=window, adjust=False).mean()
    zlema_series.name = f'ZMA_{window}'

    return zlema_series, [zlema_series.name]
