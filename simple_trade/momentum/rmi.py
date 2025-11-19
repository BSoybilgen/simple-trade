import numpy as np
import pandas as pd


def rmi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Relative Momentum Index (RMI).

    The RMI extends RSI by measuring momentum over a fixed lookback (momentum period)
    and then smoothing gains/losses over a second window. Values oscillate between
    0 and 100, similar to RSI, aiding overbought/oversold detection.

    Args:
        df (pd.DataFrame): Input DataFrame containing prices.
        parameters (dict, optional): Calculation parameters.
            - window (int): Smoothing window for average gains/losses (default 20).
            - momentum_period (int): Lookback for momentum difference (default 5).
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices (default 'Close').

    Returns:
        tuple: (rmi_series, [column_name])
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 20))
    momentum_period = int(parameters.get('momentum_period', 5))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    momentum = series - series.shift(momentum_period)

    gains = momentum.where(momentum > 0, 0.0)
    losses = (-momentum.where(momentum < 0, 0.0))

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace({0: np.nan})
    rmi_values = 100 - (100 / (1 + rs))
    rmi_values = rmi_values.astype(float)
    rmi_values.name = f'RMI_{window}_{momentum_period}'

    columns_list = [rmi_values.name]
    return rmi_values, columns_list
