import pandas as pd


def evw(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Elastic Volume Weighted Moving Average (evw).
    evw is a volume-weighted moving average that treats time as volume blocks,
    effectively stretching or compressing the time axis based on volume activity.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The base lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the EVW series and a list of column names.

    The evw is calculated as follows:

    1. Calculate Cumulative Volume:
       CumVol = Cumulative Sum of Volume

    2. Calculate Volume Weighted Position:
       The idea is that each bar "occupies" space proportional to its volume.

    3. Apply Elastic Smoothing:
       EVW uses an adaptive alpha based on the volume ratio.
       alpha = Volume / SumVolume(window)
       EVW = alpha * Price + (1 - alpha) * PrevEVW

    Interpretation:
    - High volume bars have more influence on the EVW.
    - In low volume periods, EVW changes slowly (more smoothing).
    - In high volume periods, EVW reacts quickly.

    Use Cases:
    - Volume Confirmation: Trend moves with high volume are more significant.
    - Dynamic Support/Resistance: EVW often acts as S/R level.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    window = int(parameters.get('window', 20))

    price = df[close_col].copy()
    volume = df[volume_col].copy()

    # Calculate rolling sum of volume
    vol_sum = volume.rolling(window=window, min_periods=1).sum()

    # Initialize EVWMA
    evwma = pd.Series(index=price.index, dtype=float)
    evwma.iloc[0] = price.iloc[0]

    for i in range(1, len(price)):
        if pd.notna(price.iloc[i]) and pd.notna(volume.iloc[i]) and vol_sum.iloc[i] > 0:
            # Calculate adaptive alpha based on volume ratio
            alpha = volume.iloc[i] / vol_sum.iloc[i]
            alpha = min(max(alpha, 0.01), 1.0)  # Bound alpha

            prev_evwma = evwma.iloc[i-1] if pd.notna(evwma.iloc[i-1]) else price.iloc[i]
            evwma.iloc[i] = alpha * price.iloc[i] + (1 - alpha) * prev_evwma
        else:
            evwma.iloc[i] = evwma.iloc[i-1] if pd.notna(evwma.iloc[i-1]) else price.iloc[i]

    evwma.name = f'EVW_{window}'

    columns_list = [evwma.name]
    return evwma, columns_list


def strategy_evw(
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
    evw (Elastic Volume Weighted MA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast evw crosses above slow evw, sell when crosses below.
    WHY: evw adapts its smoothing based on volume. High volume periods have
         more influence. Effectively stretches/compresses time based on activity.
    BEST MARKETS: Volume-driven markets. Stocks, ETFs. Good for confirming
                  trend moves with volume support.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 10), 'long_window' (default 20)
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
    
    short_window = int(parameters.get('short_window', 10))
    long_window = int(parameters.get('long_window', 20))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'EVW_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='evw',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'EVW_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='evw',
        parameters={"window": long_window},
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
