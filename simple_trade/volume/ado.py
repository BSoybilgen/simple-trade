import numpy as np


def ado(df, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Accumulation/Distribution Oscillator (ADO), which measures the momentum
    of the Accumulation/Distribution Line. It helps identify the strength of accumulation
    or distribution by comparing the current A/D value to a past value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for rate of change. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the A/D Oscillator series and a list of column names.

    The Accumulation/Distribution Oscillator is calculated as follows:

    1. Calculate A/D Line:
       See Accumulation/Distribution Line (ADL) calculation.

    2. Calculate Oscillator (Rate of Change):
       ADO = A/D Line - A/D Line(n periods ago)

    Interpretation:
    - Positive ADO: Accumulation pressure is increasing.
    - Negative ADO: Distribution pressure is increasing.
    - Crossing Zero: Signal of change in pressure direction (Buying <-> Selling).

    Use Cases:
    - Momentum Analysis: Gauge the acceleration of money flow.
    - Reversal Detection: Look for divergences between Price and ADO.
    - Trend Strength: Increasing ADO confirms uptrend strength.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = parameters.get('period', 14)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate A/D Line
    price_range = high - low
    price_range_nonzero = price_range.replace(0, np.nan)
    mfm = ((2 * close - high - low) / price_range_nonzero).fillna(0)
    mfv = mfm * volume
    ad_line = mfv.cumsum()
    
    # Calculate A/D Oscillator as rate of change
    ado_values = ad_line - ad_line.shift(period)
    
    ado_values.name = f'ADO_{period}'
    columns_list = [ado_values.name]
    return ado_values, columns_list
