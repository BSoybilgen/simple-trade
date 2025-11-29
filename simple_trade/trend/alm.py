import math
import numpy as np
import pandas as pd


def alm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Arnaud Legoux Moving Average (ALMA).
    ALMA uses a Gaussian distribution to apply weights to the moving window,
    allowing for smoothness and responsiveness with minimal lag.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 9.
            - sigma (float): The standard deviation of the Gaussian distribution. Default is 6.
            - offset (float): The offset of the Gaussian distribution (0 to 1). Default is 0.85.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ALMA series and a list of column names.

    The ALMA is calculated as follows:

    1. Determine Weights using Gaussian Distribution:
       Weight(i) = exp( - (i - m)^2 / (2 * s^2) )
       Where:
       m = offset * (window - 1)
       s = window / sigma

    2. Normalize Weights:
       Normalized Weight(i) = Weight(i) / Sum(Weights)

    3. Calculate ALMA:
       ALMA = Sum(Price(i) * Normalized Weight(i)) over the window

    Interpretation:
    - ALMA is designed to be smoother than SMA/EMA but with less lag.
    - Offset > 0.5 makes it more responsive to recent prices (less lag, more overshoot).
    - Sigma controls the width of the filter (smoothness).

    Use Cases:
    - Trend Identification: Similar to other moving averages but with better fidelity.
    - Support/Resistance: Acts as dynamic support and resistance.
    - Crossovers: Fast ALMA crossing slow ALMA.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 9))
    sigma = float(parameters.get('sigma', 6))
    offset = float(parameters.get('offset', 0.85))

    series = df[close_col]

    m = offset * (window - 1)
    s = window / sigma

    w = []
    for i in range(window):
        weight = math.exp(-((i - m) ** 2) / (2 * (s ** 2)))
        w.append(weight)

    weights = np.array(w)
    weights = weights / weights.sum()  # Normalize

    def weighted_avg(x):
        return np.dot(x, weights)

    alma_series = series.rolling(window=window).apply(weighted_avg, raw=True)
    alma_series.name = f'ALMA_{window}'

    columns_list = [alma_series.name]
    return alma_series, columns_list


def strategy_alm(
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
    ALM (ALMA - Arnaud Legoux Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast ALMA crosses above slow ALMA, sell when crosses below.
    WHY: ALMA uses Gaussian distribution for weighting, reducing lag while maintaining
         smoothness. Offset parameter controls responsiveness vs smoothness tradeoff.
    BEST MARKETS: Trending markets. Particularly good for stocks and forex where
                  reduced lag is valuable. Less effective in ranging markets.
    TIMEFRAME: Daily charts. Can be used on lower TFs with adjusted parameters.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 9), 'long_window' (default 27)
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
    
    short_window = int(parameters.get('short_window', 9))
    long_window = int(parameters.get('long_window', 27))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'ALMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='alm',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'ALMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='alm',
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
