import pandas as pd


def cmo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Chande Momentum Oscillator (CMO).

    The CMO is a momentum indicator oscillating between -100 and +100 that compares
    the sum of recent gains to the sum of recent losses over a specified lookback window.
    Values above 50 typically indicate strong upward momentum, while values below -50
    indicate strong downward momentum.

    Args:
        df (pd.DataFrame): Input OHLC data.
        parameters (dict, optional): Calculation parameters.
            - window (int): Lookback period for rolling sums. Default is 14.
        columns (dict, optional): Column overrides.
            - close_col (str): Close price column. Default is 'Close'.

    Returns:
        tuple: (cmo_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    delta = series.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = (-delta.where(delta < 0, 0.0))

    sum_gains = gains.rolling(window=window, min_periods=window).sum()
    sum_losses = losses.rolling(window=window, min_periods=window).sum()
    denominator = sum_gains + sum_losses

    cmo_values = 100 * (sum_gains - sum_losses) / denominator.where(denominator != 0)
    cmo_values.name = f'CMO_{window}'

    columns_list = [cmo_values.name]
    return cmo_values, columns_list
