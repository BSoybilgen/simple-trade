import pandas as pd


def vor(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Vortex Indicator (vor), a technical indicator developed by Etienne Botes and Douglas Siepman.
    It consists of two lines (VOR+ and VOR-) that capture positive and negative trend movements.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the indicator. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Vortex Indicator DataFrame (VOR_Plus, VOR_Minus) and a list of column names.

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

    4. Calculate VOR lines:
       VOR+ = SumVM+ / SumTR
       VOR- = SumVM- / SumTR

    Interpretation:
    - VOR+ > VOR-: Bulls are in control (Uptrend).
    - VOR- > VOR+: Bears are in control (Downtrend).
    - Crossovers: VOR+ crossing above VOR- is a buy signal; VOR- crossing above VOR+ is a sell signal.

    Use Cases:
    - Trend Identification: Determining the current trend direction.
    - Reversal Signals: Crossovers of the two lines signal potential trend reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 14)
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

    vor_plus = vm_plus_sum / tr_sum
    vor_minus = vm_minus_sum / tr_sum

    columns_list = [f'VOR_Plus_{window}', f'VOR_Minus_{window}']
    result = pd.DataFrame({
        columns_list[0]: vor_plus,
        columns_list[1]: vor_minus,
    }, index=df.index)

    return result, columns_list


def strategy_vor(
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
    vor (Vortex Indicator) - VOR+/VOR- Crossover Strategy
    
    LOGIC: Buy when VOR+ crosses above VOR- (bulls in control), sell when VOR- crosses above VOR+.
    WHY: vor captures positive and negative trend movements. VOR+ > VOR- = uptrend,
         VOR- > VOR+ = downtrend. Crossovers signal trend reversals.
    BEST MARKETS: Trending markets. Stocks, forex, futures. Good for identifying
                  trend direction and potential reversals.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14)
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 14)
    
    indicator_params = {"window": window}
    short_window_indicator = f'VOR_Plus_{window}'
    long_window_indicator = f'VOR_Minus_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='vor',
        parameters=indicator_params,
        figure=False
    )
    
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
