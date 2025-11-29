import numpy as np
import pandas as pd


def lsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Laguerre RSI (LRSI), a technical indicator developed by John Ehlers 
    that uses a Laguerre filter to create an RSI with less lag and less noise.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - gamma (float): The damping factor (0 to 1). Higher gamma means more damping (smoother). Default is 0.5.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Laguerre RSI series and a list of column names.

    The Laguerre RSI is calculated using a four-element Laguerre filter:

    1. Calculate Filtered Prices (L0, L1, L2, L3):
       Apply the Laguerre transform to the price series.

    2. Calculate Up/Down Moves (Gains and Losses):
       CU = (L0 - L1) + (L1 - L2) + (L2 - L3) (only positive differences)
       CD = (L0 - L1) + (L1 - L2) + (L2 - L3) (only negative differences abs value)

    3. Calculate LRSI:
       LRSI = 100 * CU / (CU + CD)

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 80 (or 0.8 depending on scale, here 100) typically indicate overbought.
    - Oversold: Values below 20 typically indicate oversold.
    - Responsiveness: Reacts faster to price changes than standard RSI due to the Laguerre filter.

    Use Cases:
    - Scalping/Short-term Trading: Due to its low lag, it is popular for short-term entries.
    - Reversal Detection: Quick identification of turning points.
    - Trend Filters: Gamma can be adjusted to tune the indicator to the market cycle.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    gamma = float(parameters.get('gamma', 0.5))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    prices = series.to_numpy(dtype=float)
    length = len(series)

    l0 = np.full(length, np.nan)
    l1 = np.full(length, np.nan)
    l2 = np.full(length, np.nan)
    l3 = np.full(length, np.nan)
    lrsi = np.full(length, np.nan)

    prev_l0 = prev_l1 = prev_l2 = prev_l3 = None

    for idx, price in enumerate(prices):
        if np.isnan(price):
            continue

        if prev_l0 is None:
            prev_l0 = prev_l1 = prev_l2 = prev_l3 = price
            l0[idx] = prev_l0
            l1[idx] = prev_l1
            l2[idx] = prev_l2
            l3[idx] = prev_l3
            continue

        new_l0 = (1 - gamma) * price + gamma * prev_l0
        new_l1 = -gamma * new_l0 + prev_l0 + gamma * prev_l1
        new_l2 = -gamma * new_l1 + prev_l1 + gamma * prev_l2
        new_l3 = -gamma * new_l2 + prev_l2 + gamma * prev_l3

        l0[idx] = new_l0
        l1[idx] = new_l1
        l2[idx] = new_l2
        l3[idx] = new_l3

        diff0 = new_l0 - new_l1
        diff1 = new_l1 - new_l2
        diff2 = new_l2 - new_l3

        cu = max(diff0, 0.0) + max(diff1, 0.0) + max(diff2, 0.0)
        cd = max(-diff0, 0.0) + max(-diff1, 0.0) + max(-diff2, 0.0)

        denom = cu + cd
        lrsi[idx] = np.nan if denom == 0 else 100 * cu / denom

        prev_l0, prev_l1, prev_l2, prev_l3 = new_l0, new_l1, new_l2, new_l3

    gamma_str = f"{gamma:g}"
    lrsi_series = pd.Series(lrsi, index=series.index, name=f'LRSI_{gamma_str}')

    columns_list = [lrsi_series.name]
    return lrsi_series, columns_list


def strategy_lsi(
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
    LSI (Laguerre RSI) - Mean Reversion Strategy
    
    LOGIC: Buy when LRSI drops below lower threshold (oversold), sell when above upper.
    WHY: Laguerre filter creates an RSI with less lag and noise. Reacts faster to price
         changes than standard RSI. Gamma controls smoothing (higher = smoother).
    BEST MARKETS: Short-term trading and scalping. Forex, futures, and liquid stocks.
                  Popular for quick reversal detection due to low lag.
    TIMEFRAME: Intraday to daily. Gamma 0.5 is common. Lower gamma = faster response.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'gamma' (default 0.5), 'upper' (default 80), 'lower' (default 20)
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
    
    gamma = float(parameters.get('gamma', 0.5))
    upper = int(parameters.get('upper', 80))
    lower = int(parameters.get('lower', 20))
    
    indicator_params = {"gamma": gamma}
    gamma_str = f"{gamma:g}"
    indicator_col = f'LRSI_{gamma_str}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='lsi',
        parameters=indicator_params,
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
