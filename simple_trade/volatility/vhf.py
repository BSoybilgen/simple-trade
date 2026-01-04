import pandas as pd


def vhf(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Vertical Horizontal Filter (vhf), which determines whether prices
    are in a trending phase or a congestion phase by comparing the range of prices
    to the sum of price changes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 28.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the VHF series and a list of column names.

    Calculation Steps:
    1. Find Extremes:
       HCP = Highest Close over period
       LCP = Lowest Close over period

    2. Calculate Numerator:
       Numerator = HCP - LCP (Price Range)

    3. Calculate Denominator:
       Denominator = Sum of absolute price changes (|Close - Previous Close|) over period.

    4. Calculate VHF:
       VHF = Numerator / Denominator

    Interpretation:
    - High VHF (> 0.40): Strong trending phase (up or down).
    - Low VHF (< 0.25): Congestion or choppy phase.
    - Rising VHF: Developing trend.
    - Falling VHF: Entering congestion.

    Use Cases:
    - Trend Identification: Determine if the market is suitable for trend-following strategies.
    - Indicator Selection: Use moving averages when VHF is high; use oscillators when VHF is low.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 28))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate highest and lowest close over period
    highest_close = close.rolling(window=period).max()
    lowest_close = close.rolling(window=period).min()
    
    # Calculate numerator (range)
    numerator = highest_close - lowest_close
    
    # Calculate denominator (sum of absolute changes)
    price_changes = close.diff().abs()
    denominator = price_changes.rolling(window=period).sum()
    
    # Calculate VHF
    vhf_values = numerator / denominator
    
    vhf_values.name = f'VHF_{period}'
    columns_list = [vhf_values.name]
    return vhf_values, columns_list


def strategy_vhf(
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
    vhf (Vertical Horizontal Filter) - Trend vs Congestion Strategy
    
    LOGIC: Buy when vhf rises above upper threshold (trending phase),
           sell when drops below lower threshold (congestion phase).
    WHY: vhf compares price range to sum of changes. High vhf = trending,
         low vhf = choppy/congestion.
    BEST MARKETS: All markets. Use to select trend vs mean-reversion strategies.
    TIMEFRAME: Daily charts. 28-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 28), 'upper' (default 0.40),
                    'lower' (default 0.25)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_band_trade_strategies import run_band_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 28))
    upper = float(parameters.get('upper', 0.40))
    lower = float(parameters.get('lower', 0.25))
    price_col = 'Close'
    indicator_col = f'VHF_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='vhf',
        parameters={"period": period},
        figure=False
    )
    
    data['upper'] = upper
    data['lower'] = lower
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
