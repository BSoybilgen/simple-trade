import pandas as pd


def zma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Zero-Lag Moving Average (ZLEMA).
    ZLEMA reduces lag by pre-adjusting the input series with its own momentum
    before applying an EMA. This keeps responsiveness similar to shorter EMAs
    while preserving smoothness.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ZLEMA series and a list of column names.

    The Zero-Lag Moving Average is calculated as follows:

    1. Calculate Lag:
       Lag = Window / 2

    2. Adjust Data for Lag (De-lagged Data):
       Adjusted Price = Price + (Price - Price(Lag periods ago))

    3. Calculate EMA of Adjusted Data:
       ZLEMA = EMA(Adjusted Price, window)

    Interpretation:
    - ZLEMA removes much of the lag associated with trend-following indicators.
    - It tracks prices very closely, making it sensitive to reversals.

    Use Cases:
    - Trend Following: Catching trends earlier than SMA/EMA.
    - Reversal Trading: Sharp turns in ZLEMA can signal reversals.
    - Crossovers: ZLEMA crossing price or another MA.
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
