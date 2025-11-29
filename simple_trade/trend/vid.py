import numpy as np
import pandas as pd


def vid(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Variable Index Dynamic Average (VIDYA).
    VIDYA is an adaptive moving average developed by Tushar Chande. It adjusts its
    smoothing constant based on market volatility, measured by the Chande Momentum Oscillator (CMO).

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The base lookback period. Default is 21.
            - cmo_window (int): The lookback period for CMO. Default is 9.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the VIDYA series and a list of column names.

    The VIDYA is calculated as follows:

    1. Calculate Chande Momentum Oscillator (CMO):
       CMO = 100 * (Sum(Up) - Sum(Down)) / (Sum(Up) + Sum(Down))
       over cmo_window.

    2. Calculate Smoothing Constant (Alpha):
       Base Alpha = 2 / (window + 1)
       Alpha = Base Alpha * Abs(CMO) / 100

    3. Calculate VIDYA:
       VIDYA = Alpha * Price + (1 - Alpha) * Previous VIDYA

    Interpretation:
    - In trending markets (high volatility/CMO), VIDYA tracks price closely (high Alpha).
    - In sideways markets (low volatility/CMO), VIDYA flattens out (low Alpha) to act as support/resistance.

    Use Cases:
    - Trend Following: Adapts to market speed.
    - Dynamic Support/Resistance: Provides reliable S/R levels in varying volatility conditions.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 21))
    cmo_window = int(parameters.get('cmo_window', 9))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]

    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)

    sum_up = up.rolling(window=cmo_window).sum()
    sum_down = down.rolling(window=cmo_window).sum()
    denominator = sum_up + sum_down
    denominator = denominator.replace(0, np.nan)

    cmo = ((sum_up - sum_down) / denominator) * 100
    cmo = cmo.fillna(0.0)

    base_alpha = 2 / (window + 1)

    vidya = pd.Series(np.nan, index=series.index)
    prev_vidya = None

    for idx, price in enumerate(series):
        cmo_value = cmo.iat[idx]
        if np.isnan(price) or np.isnan(cmo_value):
            continue

        alpha = base_alpha * abs(cmo_value) / 100
        if prev_vidya is None:
            prev_vidya = price

        vid_value = alpha * price + (1 - alpha) * prev_vidya
        vidya.iat[idx] = vid_value
        prev_vidya = vid_value

    name = f'VID_{window}_{cmo_window}'
    vidya.name = name

    columns_list = [name]
    return vidya, columns_list


def strategy_vid(
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
    VID (VIDYA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast VIDYA crosses above slow VIDYA, sell when crosses below.
    WHY: VIDYA adapts smoothing based on CMO volatility. Fast in trends,
         slow in sideways markets. Developed by Tushar Chande.
    BEST MARKETS: All market conditions due to adaptive nature. Stocks, forex.
                  Good for varying volatility environments.
    TIMEFRAME: Daily charts. 21-period with 9-period CMO is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 10), 'long_window' (default 21)
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
    long_window = int(parameters.get('long_window', 21))
    cmo_window = int(parameters.get('cmo_window', 9))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'VID_{short_window}_{cmo_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='vid',
            parameters={"window": short_window, "cmo_window": cmo_window},
            figure=False
        )
    
    long_window_indicator = f'VID_{long_window}_{cmo_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='vid',
        parameters={"window": long_window, "cmo_window": cmo_window},
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
