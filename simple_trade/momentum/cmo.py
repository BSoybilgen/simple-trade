import pandas as pd


def cmo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Chande Momentum Oscillator (CMO), a technical momentum indicator developed by Tushar Chande.
    It compares the sum of recent gains to the sum of recent losses to determine momentum.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the CMO series and a list of column names.

    The Chande Momentum Oscillator is calculated as follows:

    1. Calculate Price Differences:
       Diff = Close - Close(prev)

    2. Separate Gains and Losses:
       Gain = Diff if Diff > 0 else 0
       Loss = Abs(Diff) if Diff < 0 else 0

    3. Sum Gains and Losses over the Window:
       Sum Gains = Sum(Gain, window)
       Sum Losses = Sum(Loss, window)

    4. Calculate CMO:
       CMO = 100 * (Sum Gains - Sum Losses) / (Sum Gains + Sum Losses)

    Interpretation:
    - Range: The oscillator fluctuates between -100 and +100.
    - Overbought: Values above +50 typically indicate overbought conditions.
    - Oversold: Values below -50 typically indicate oversold conditions.
    - Trend Strength: High absolute values indicate strong trends.

    Use Cases:
    - Overbought/Oversold Levels: Identifying potential reversal points when CMO reaches extremes.
    - Trend Strength: Measuring the strength of the trend (higher absolute value = stronger trend).
    - Crosses: Crossing the zero line can be used as a signal (Bullish > 0, Bearish < 0).
    - Divergence: Divergence between price and CMO can signal potential reversals.
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
