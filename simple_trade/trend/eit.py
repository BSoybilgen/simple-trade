import numpy as np
import pandas as pd


def eit(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ehlers Instantaneous Trendline (eit).
    eit applies John Ehlers' technique of averaging a weighted price input and
    recursively smoothing it with a tunable alpha to obtain a low-lag trend estimate.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - alpha (float): The smoothing factor (0.01 to 1.0). Default is 0.07.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the EIT series and a list of column names.

    The Ehlers Instantaneous Trendline is calculated as follows:

    1. Calculate Weighted Price:
       Weighted = (Price + 2*Price[1] + Price[2]) / 4

    2. Calculate Instantaneous Trend (Recursive):
       IT[i] = (alpha * Weighted[i]) + ((1 - alpha) * IT[i-1])

    Interpretation:
    - Provides a very smooth trendline that tracks price closely.
    - Low lag compared to simple moving averages.

    Use Cases:
    - Trend Definition: Clear visualization of the current trend.
    - Crossovers: Price crossing EIT or EIT crossing another MA.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    alpha = float(parameters.get('alpha', 0.07))
    alpha = min(max(alpha, 0.01), 1.0)

    close = df[close_col].astype(float)
    weighted_price = (close + 2 * close.shift(1) + close.shift(2)) / 4

    values = weighted_price.to_numpy(dtype=float)
    itrend = np.full_like(values, np.nan)

    valid_idx = np.where(~np.isnan(values))[0]
    if valid_idx.size == 0:
        series = pd.Series(itrend, index=close.index, name=f'EIT_{alpha}')
        return series, [series.name]

    start = valid_idx[0]
    itrend[start] = values[start]

    for i in range(start + 1, len(values)):
        price_term = values[i]
        prev = itrend[i - 1]

        if np.isnan(price_term):
            itrend[i] = prev
            continue

        if np.isnan(prev):
            prev = price_term

        itrend[i] = alpha * price_term + (1 - alpha) * prev

    series = pd.Series(itrend, index=close.index, name=f'EIT_{alpha}')
    return series, [series.name]


def strategy_eit(
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
    eit (Ehlers Instantaneous Trendline) - Price vs Trend Crossover Strategy
    
    LOGIC: Buy when price crosses above eit, sell when crosses below.
    WHY: eit provides smooth, low-lag trendline using Ehlers' technique.
         Tracks price closely with minimal lag compared to standard MAs.
    BEST MARKETS: All markets. Stocks, forex, futures. Good for
                  trend definition and crossover strategies.
    TIMEFRAME: Daily charts. Alpha of 0.07 is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'alpha' (default 0.07)
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
    
    alpha = float(parameters.get('alpha', 0.07))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='eit',
        parameters={"alpha": alpha},
        figure=False
    )
    
    short_window_indicator = 'Close'
    long_window_indicator = f'EIT_{alpha}'
    
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
    
    # Include Close price in plot so users can see the crossover signals
    indicator_cols_to_plot = [long_window_indicator, 'Close']
    
    return results, portfolio, indicator_cols_to_plot, data
