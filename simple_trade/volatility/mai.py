import pandas as pd


def mai(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Mass Index (MI), a volatility indicator designed to identify trend reversals
    by measuring the narrowing and widening of the range between high and low prices.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - ema_period (int): The period for the first EMA calculation. Default is 9.
            - sum_period (int): The period for summing the EMA ratio. Default is 25.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Mass Index series and a list of column names.

    Calculation Steps:
    1. Calculate High-Low Range:
       Range = High - Low
    2. Calculate Single EMA:
       EMA1 = EMA(Range, ema_period)
    3. Calculate Double EMA:
       EMA2 = EMA(EMA1, ema_period)
    4. Calculate EMA Ratio:
       Ratio = EMA1 / EMA2
    5. Calculate Mass Index:
       MI = Sum(Ratio, sum_period)

    Interpretation:
    - MI typically ranges between 18 and 30.
    - Reversal Bulge: MI rises above 27 then drops below 26.5.
    - Suggests potential trend reversal (doesn't indicate direction).

    Use Cases:
    - Reversal detection: Identifying "reversal bulge" patterns.
    - Volatility expansion/contraction: Rising MI = Expansion; Falling MI = Contraction.
    - Trend exhaustion: Extreme values suggest trend is losing momentum.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    ema_period = int(parameters.get('ema_period', 9))
    sum_period = int(parameters.get('sum_period', 25))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    
    high = df[high_col]
    low = df[low_col]
    
    # Calculate the high-low range
    hl_range = high - low
    
    # Calculate single EMA of the range
    single_ema = hl_range.ewm(span=ema_period, adjust=False).mean()
    
    # Calculate double EMA (EMA of the single EMA)
    double_ema = single_ema.ewm(span=ema_period, adjust=False).mean()
    
    # Calculate the EMA ratio
    ema_ratio = single_ema / double_ema
    
    # Sum the EMA ratio over the sum_period
    mass_index = ema_ratio.rolling(window=sum_period).sum()
    
    mass_index.name = f'MI_{ema_period}_{sum_period}'
    columns_list = [mass_index.name]
    return mass_index, columns_list
