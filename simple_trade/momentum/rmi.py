import numpy as np
import pandas as pd


def rmi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Relative Momentum Index (RMI), a variation of the Relative Strength Index (RSI).
    Instead of using day-to-day price changes, RMI uses price changes over a specified momentum period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The smoothing window for average gains/losses. Default is 20.
            - momentum_period (int): The lookback period for momentum difference. Default is 5.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the RMI series and a list of column names.

    The Relative Momentum Index is calculated as follows:

    1. Calculate Momentum:
       Momentum = Close - Close(shifted by momentum_period)

    2. Separate Gains and Losses:
       Gain = Momentum if Momentum > 0 else 0
       Loss = Abs(Momentum) if Momentum < 0 else 0

    3. Calculate Average Gain and Loss:
       Avg Gain = SMA(Gain, window)
       Avg Loss = SMA(Loss, window)

    4. Calculate RMI:
       RS = Avg Gain / Avg Loss
       RMI = 100 - (100 / (1 + RS))

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 indicate potential overbought conditions.
    - Oversold: Values below 30 indicate potential oversold conditions.
    - Divergence: Divergence between price and RMI can signal reversals.

    Use Cases:
    - Cycle Analysis: RMI is often smoother than RSI and can better highlight cyclical turns.
    - Trend Confirmation: Confirming the direction of the trend.
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
