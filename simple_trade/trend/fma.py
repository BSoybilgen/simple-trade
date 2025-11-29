import math
import numpy as np
import pandas as pd


def fma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Fractal Adaptive Moving Average (FRAMA).
    FRAMA adapts its smoothing factor based on the fractal dimension of price
    movements, allowing it to react quickly during strong trends while
    remaining smooth in choppy markets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 16.
            - alpha_floor (float): Minimum smoothing factor. Default is 0.01.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the FRAMA series and a list of column names.

    The FRAMA is calculated as follows:

    1. Calculate Fractal Dimension (D):
       D = (Log(N1 + N2) - Log(N3)) / Log(2)
       Where N1, N2, N3 are price ranges over different sub-windows.

    2. Calculate Alpha:
       Alpha = Exp(-4.6 * (D - 1))
       (Clamped between alpha_floor and 1.0)

    3. Calculate FRAMA:
       FRAMA = Alpha * Price + (1 - Alpha) * Previous FRAMA

    Interpretation:
    - When fractal dimension is high (choppy market), Alpha is low (more smoothing).
    - When fractal dimension is low (trending market), Alpha is high (less smoothing, faster reaction).

    Use Cases:
    - Trend Following: Adapts to changing market conditions effectively.
    - Filtering: Reduces noise in consolidation zones.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = max(2, int(parameters.get('window', 16)))
    alpha_floor = float(parameters.get('alpha_floor', 0.01))

    close = df[close_col]
    values = close.to_numpy(dtype=float)
    n = len(values)
    frama_values = np.full(n, np.nan)

    non_nan_idx = np.where(~np.isnan(values))[0]
    if non_nan_idx.size == 0:
        series = pd.Series(frama_values, index=close.index, name=f'FMA_{window}')
        return series, [series.name]

    start_idx = non_nan_idx[0]
    frama_values[start_idx] = values[start_idx]

    half_window = max(1, window // 2)

    for i in range(start_idx + 1, n):
        price = values[i]
        prev = frama_values[i - 1]

        if np.isnan(price):
            frama_values[i] = prev
            continue

        if i < start_idx + window:
            # Use simple moving average during warmup period
            lookback = min(i - start_idx + 1, window)
            warmup_slice = values[max(0, i - lookback + 1):i + 1]
            warmup_slice = warmup_slice[~np.isnan(warmup_slice)]
            if warmup_slice.size > 0:
                frama_values[i] = warmup_slice.mean()
            else:
                frama_values[i] = price if np.isnan(prev) else prev
            continue

        dimension = _fractal_dimension(values, i, window, half_window)
        alpha = math.exp(-4.6 * (dimension - 1))
        alpha = min(max(alpha, alpha_floor), 1.0)

        if np.isnan(prev):
            frama_values[i] = price
        else:
            frama_values[i] = alpha * price + (1 - alpha) * prev

    frama_series = pd.Series(frama_values, index=close.index, name=f'FMA_{window}')
    return frama_series, [frama_series.name]


def _fractal_dimension(values: np.ndarray, idx: int, window: int, half_window: int) -> float:
    start = idx - window + 1
    window_slice = values[start: idx + 1]
    first_slice = window_slice[:half_window]
    second_slice = window_slice[half_window: half_window * 2]

    def _range(data: np.ndarray) -> float:
        data = data[~np.isnan(data)]
        if data.size == 0:
            return 0.0
        return data.max() - data.min()

    n1 = _range(first_slice) / half_window if half_window else 0.0
    n2 = _range(second_slice) / half_window if half_window else 0.0
    n3 = _range(window_slice) / window if window else 0.0

    if n3 <= 0 or (n1 + n2) <= 0:
        return 1.0

    dimension = (math.log(n1 + n2) - math.log(n3)) / math.log(2)
    return min(max(dimension, 1.0), 2.0)


def strategy_fma(
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
    FMA (FRAMA - Fractal Adaptive MA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast FRAMA crosses above slow FRAMA, sell when crosses below.
    WHY: FRAMA adapts smoothing based on fractal dimension. Fast in trends,
         smooth in choppy markets. Excellent adaptive behavior.
    BEST MARKETS: All market conditions. Stocks, forex, futures.
                  Particularly good for varying volatility environments.
    TIMEFRAME: Daily charts. 16-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 8), 'long_window' (default 16)
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
    
    short_window = int(parameters.get('short_window', 8))
    long_window = int(parameters.get('long_window', 16))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'FMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='fma',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'FMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='fma',
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
