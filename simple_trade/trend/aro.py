import pandas as pd


def aro(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Aroon indicator, which measures the time it takes for a security
    to reach its highest and lowest points over a specified time period.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
    
    Returns:
        tuple: A tuple containing the Aroon DataFrame (Aroon Up, Aroon Down, Oscillator) and a list of column names.
    
    The Aroon indicator is calculated as follows:
    
    1. Calculate Aroon Up:
       Measures periods since the highest high within the lookback period.
       Aroon Up = ((period - periods since highest high) / period) * 100
    
    2. Calculate Aroon Down:
       Measures periods since the lowest low within the lookback period.
       Aroon Down = ((period - periods since lowest low) / period) * 100
       
    3. Calculate Aroon Oscillator:
       Aroon Oscillator = Aroon Up - Aroon Down
    
    Interpretation:
    - Aroon Up > 70: Strong uptrend.
    - Aroon Down > 70: Strong downtrend.
    - Aroon Up/Down < 30: Weak trend.
    - Crossovers: Aroon Up crossing above Aroon Down signals potential bullish trend.
    
    Use Cases:
    - Trend identification: Determine the direction and strength of the current trend.
    - Consolidation detection: When both Aroon Up and Down are low (< 50), it suggests price consolidation.
    - Breakout confirmation: A strong move in Aroon Up/Down can confirm a price breakout.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    period = int(parameters.get('period', 14))

    high = df[high_col]
    low = df[low_col]
    
    # Create aroon_up and aroon_down series
    aroon_up = pd.Series(index=high.index, dtype=float)
    aroon_down = pd.Series(index=low.index, dtype=float)
    
    # Calculate Aroon indicators for each rolling window
    for i in range(len(high) - period + 1):
        # Get the current window
        high_window = high.iloc[i:i+period]
        low_window = low.iloc[i:i+period]
        
        # Find the highest high and lowest low
        highest_high = high_window.max()
        lowest_low = low_window.min()
        
        # Find the periods since highest high and lowest low
        periods_since_highest = period - 1 - high_window.values.tolist().index(highest_high)
        periods_since_lowest = period - 1 - low_window.values.tolist().index(lowest_low)
        
        # Calculate Aroon Up and Aroon Down
        aroon_up.iloc[i+period-1] = ((period - periods_since_highest) / period) * 100
        aroon_down.iloc[i+period-1] = ((period - periods_since_lowest) / period) * 100
    
    # Calculate Aroon Oscillator
    aroon_oscillator = aroon_up - aroon_down

    df_aroon = pd.DataFrame({
        f'AROON_UP_{period}': aroon_down,
        f'AROON_DOWN_{period}': aroon_up,
        f'AROON_OSCILLATOR_{period}': aroon_oscillator
    })
    df_aroon.index = high.index

    columns = list(df_aroon.columns)

    return df_aroon, columns


def strategy_aro(
    data: pd.DataFrame,
    parameters: dict = None,
    config = None,
    trading_type: str = 'long',
    day1_position: str = 'none',
    risk_free_rate: float = 0.0,
    long_entry_pct_cash: float = 1.0,
    short_entry_pct_cash: float = 1.0
) -> tuple:
    """
    ARO (Aroon) - Aroon Up/Down Crossover Strategy
    
    LOGIC: Buy when Aroon Up crosses above Aroon Down, sell when crosses below.
    WHY: Aroon measures time since highest high and lowest low. Aroon Up > Down
         indicates uptrend, Down > Up indicates downtrend.
    BEST MARKETS: Trending markets. Stocks, forex, commodities. Good for
                  identifying trend direction and strength.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_cross_trade_strategies import run_cross_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    period = int(parameters.get('period', 14))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='aro',
        parameters={"period": period},
        figure=False
    )
    
    short_window_indicator = f'AROON_UP_{period}'
    long_window_indicator = f'AROON_DOWN_{period}'
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data
