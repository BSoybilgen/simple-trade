import pandas as pd


def efr(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Efficiency Ratio (ER), also known as Kaufman Efficiency, which
    measures the efficiency of price movement by comparing net price change to the
    sum of absolute price changes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Efficiency Ratio series and a list of column names.

    The Efficiency Ratio is calculated as follows:

    1. Calculate Net Price Change:
       Change = abs(Close - Close[period ago])

    2. Calculate Volatility (Sum of absolute changes):
       Volatility = Sum(abs(Close - Close[previous])) over period

    3. Calculate Ratio:
       ER = Change / Volatility

    Interpretation:
    - ER near 1.0: Highly efficient, strong trending market.
    - ER near 0.0: Inefficient, choppy/sideways market.
    - ER > 0.7: Strong trend.
    - ER < 0.3: Choppy/Range.

    Use Cases:
    - Trend vs. noise identification: Distinguish trending from ranging markets.
    - Strategy selection: Select trend-following vs mean-reversion systems.
    - Adaptive indicators: Used in KAMA and other adaptive averages.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 10))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate net price change over period
    net_change = (close - close.shift(period)).abs()
    
    # Calculate sum of absolute price changes
    price_changes = close.diff().abs()
    sum_changes = price_changes.rolling(window=period).sum()
    
    # Calculate Efficiency Ratio
    er_values = net_change / sum_changes
    
    # Ensure values are between 0 and 1
    er_values = er_values.clip(0, 1)
    
    er_values.name = f'ER_{period}'
    columns_list = [er_values.name]
    return er_values, columns_list
