import pandas as pd


def rvg(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Relative Vigor Index (RVG/RVI), a technical indicator that measures the conviction 
    of a recent price action and the likelihood that it will continue.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for calculation. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the RVG DataFrame (RVG, Signal) and a list of column names.

    The Relative Vigor Index is calculated as follows:

    1. Calculate Numerator (Close - Open, Smoothed):
       Num = (Close-Open) + 2*(C1-O1) + 2*(C2-O2) + (C3-O3) / 6

    2. Calculate Denominator (High - Low, Smoothed):
       Denom = (High-Low) + 2*(H1-L1) + 2*(H2-L2) + (H3-L3) / 6

    3. Calculate SMA of Numerator and Denominator:
       SMA_Num = SMA(Num, window)
       SMA_Denom = SMA(Denom, window)

    4. Calculate RVG:
       RVG = SMA_Num / SMA_Denom

    5. Calculate Signal Line:
       Signal = (RVG + 2*RVG1 + 2*RVG2 + RVG3) / 6

    Interpretation:
    - Concept: Prices tend to close higher than they open in uptrends and lower in downtrends.
    - Crossovers: RVG crossing above the Signal Line is a buy signal; below is a sell signal.

    Use Cases:
    - Trend Confirmation: Validating the strength of the current trend.
    - Signal Generation: Crossovers and Divergences.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 10))
    open_col = columns.get('open_col', 'Open')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

    o = df[open_col]
    h = df[high_col]
    low_vals = df[low_col]
    c = df[close_col]

    # RVG Numerator: (Close - Open) + 2*(C1-O1) + 2*(C2-O2) + (C3-O3) / 6
    co = c - o
    num = (co + 2 * co.shift(1) + 2 * co.shift(2) + co.shift(3)) / 6

    # RVG Denominator: (High - Low) + ...
    hl = h - low_vals
    denom = (hl + 2 * hl.shift(1) + 2 * hl.shift(2) + hl.shift(3)) / 6

    # SMA of Num and Denom
    sma_num = num.rolling(window=window).mean()
    sma_denom = denom.rolling(window=window).mean()

    rvg_val = sma_num / sma_denom

    # Signal Line: (RVG + 2*RVG1 + 2*RVG2 + RVG3) / 6
    signal = (rvg_val + 2 * rvg_val.shift(1) + 2 * rvg_val.shift(2) + rvg_val.shift(3)) / 6

    rvg_col = f'RVG_{window}'
    sig_col = 'RVG_SIG'

    result = pd.DataFrame({
        rvg_col: rvg_val,
        sig_col: signal
    })

    return result, list(result.columns)
