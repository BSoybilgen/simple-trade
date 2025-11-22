import pandas as pd


def vor(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Vortex Indicator (VI), a technical indicator developed by Etienne Botes and Douglas Siepman.
    It consists of two lines (VI+ and VI-) that capture positive and negative trend movements.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the indicator. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Vortex Indicator DataFrame (VI_Plus, VI_Minus) and a list of column names.

    The Vortex Indicator is calculated as follows:

    1. Calculate True Range (TR):
       TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))

    2. Calculate Positive and Negative Trend Movements (VM):
       VM+ = Abs(Current High - Previous Low)
       VM- = Abs(Current Low - Previous High)

    3. Sum TR, VM+, and VM- over the window:
       SumTR = Sum(TR, window)
       SumVM+ = Sum(VM+, window)
       SumVM- = Sum(VM-, window)

    4. Calculate VI lines:
       VI+ = SumVM+ / SumTR
       VI- = SumVM- / SumTR

    Interpretation:
    - VI+ > VI-: Bulls are in control (Uptrend).
    - VI- > VI+: Bears are in control (Downtrend).
    - Crossovers: VI+ crossing above VI- is a buy signal; VI- crossing above VI+ is a sell signal.

    Use Cases:
    - Trend Identification: Determining the current trend direction.
    - Reversal Signals: Crossovers of the two lines signal potential trend reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    tr_sum = true_range.rolling(window=window, min_periods=window).sum().replace(0, pd.NA)

    vm_plus = (high - prev_low).abs()
    vm_minus = (low - prev_high).abs()

    vm_plus_sum = vm_plus.rolling(window=window, min_periods=window).sum()
    vm_minus_sum = vm_minus.rolling(window=window, min_periods=window).sum()

    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum

    columns_list = [f'VI_Plus_{window}', f'VI_Minus_{window}']
    result = pd.DataFrame({
        columns_list[0]: vi_plus,
        columns_list[1]: vi_minus,
    }, index=df.index)

    return result, columns_list
