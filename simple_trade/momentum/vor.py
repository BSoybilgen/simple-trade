import pandas as pd


def vor(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the Vortex Indicator (VI+ and VI-).

    Args:
        df (pd.DataFrame): Input price data containing high, low, and close columns.
        parameters (dict, optional): Calculation parameters.
            - window (int): Lookback period for the indicator. Default is 14.
        columns (dict, optional): Column overrides.
            - high_col (str): Column name for high prices. Default is 'High'.
            - low_col (str): Column name for low prices. Default is 'Low'.
            - close_col (str): Column name for closing prices. Default is 'Close'.

    Returns:
        tuple: (DataFrame with VI+ and VI-, [column names])
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
