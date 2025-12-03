import pandas as pd


def kst(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Know Sure Thing (kst), a momentum oscillator developed by Martin Pring 
    that combines multiple Rate of Change (ROC) timeframes into a single indicator.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - roc_periods (iterable): Lookback periods for the ROC calculations. Default is (10, 15, 20, 30).
            - ma_periods (iterable): Smoothing window for each ROC series. Default is (10, 10, 10, 15).
            - weights (iterable): Weights applied to each smoothed ROC. Default is (1, 2, 3, 4).
            - signal (int): Window length for the signal line. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame (with KST and Signal columns) and a list of column names.

    The KST is calculated in the following steps:

    1. Calculate Rate of Change (ROC) for four different periods:
       ROC1, ROC2, ROC3, ROC4

    2. Smooth each ROC with a Simple Moving Average (SMA):
       RCMA1 = SMA(ROC1, ma_period1)
       RCMA2 = SMA(ROC2, ma_period2)
       RCMA3 = SMA(ROC3, ma_period3)
       RCMA4 = SMA(ROC4, ma_period4)

    3. Calculate KST Line (Weighted Sum):
       KST = (RCMA1 * W1) + (RCMA2 * W2) + (RCMA3 * W3) + (RCMA4 * W4)

    4. Calculate Signal Line:
       Signal = SMA(KST, signal_period)

    Interpretation:
    - KST crossing above Signal Line: Bullish signal.
    - KST crossing below Signal Line: Bearish signal.
    - Zero Line Crossover: KST crossing zero confirms trend direction (Positive = Uptrend, Negative = Downtrend).

    Use Cases:
    - Trend Confirmation: Validating the strength and direction of a trend using multiple timeframes.
    - Signal Generation: Crossovers of KST and its signal line.
    - Divergence: Divergence between price and KST can signal reversals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    roc_periods = parameters.get('roc_periods', (10, 15, 20, 30))
    ma_periods = parameters.get('ma_periods', (10, 10, 10, 15))
    weights = parameters.get('weights', (1, 2, 3, 4))
    signal_period = int(parameters.get('signal', 9))
    close_col = columns.get('close_col', 'Close')

    roc_periods = [int(p) for p in roc_periods]
    ma_periods = [int(p) for p in ma_periods]
    weights = [float(w) for w in weights]

    if not (len(roc_periods) == len(ma_periods) == len(weights)):
        raise ValueError('roc_periods, ma_periods, and weights must have the same length')

    close = df[close_col]
    smoothed_rocs = []

    for roc_period, ma_period in zip(roc_periods, ma_periods):
        roc_series = close.pct_change(periods=roc_period) * 100
        smoothed = roc_series.rolling(window=ma_period).mean()
        smoothed_rocs.append(smoothed)

    kst_series = sum(weight * series for weight, series in zip(weights, smoothed_rocs))
    kst_series.name = 'KST'

    signal_series = kst_series.rolling(window=signal_period).mean()
    signal_series.name = f'KST_Signal_{signal_period}'

    result = pd.DataFrame({kst_series.name: kst_series, signal_series.name: signal_series})
    result.index = df.index

    columns_list = list(result.columns)
    return result, columns_list


def strategy_kst(
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
    kst (Know Sure Thing) - Signal Line Crossover Strategy
    
    LOGIC: Buy when kst crosses above its signal line, sell when crosses below.
    WHY: kst combines 4 ROC timeframes with smoothing, capturing momentum across
         multiple cycles. Signal crossovers indicate momentum shifts confirmed by
         multiple timeframes.
    BEST MARKETS: Trending markets across all asset classes. Stocks, forex, commodities.
                  Excellent for confirming trend changes with multiple timeframe confirmation.
    TIMEFRAME: Daily or weekly charts. Good for position trading and trend confirmation.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'signal' (default 9)
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
    
    signal_period = int(parameters.get('signal', 9))
    
    indicator_params = {"signal": signal_period}
    short_window_indicator = 'KST'
    long_window_indicator = f'KST_Signal_{signal_period}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='kst',
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
