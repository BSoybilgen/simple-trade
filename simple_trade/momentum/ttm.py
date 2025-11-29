import numpy as np
import pandas as pd


def ttm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the TTM Squeeze (Trade The Markets Squeeze), a volatility and momentum indicator developed by John Carter.
    It identifies periods of consolidation (squeeze) followed by breakouts.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - length (int): Lookback window for Bollinger/Keltner calculations. Default is 20.
            - std_dev (float): Standard deviation multiplier for Bollinger Bands. Default is 2.0.
            - atr_length (int): Lookback window for ATR in Keltner Channels. Default is 20.
            - atr_multiplier (float): Multiplier applied to ATR for Keltner Channels. Default is 1.5.
            - smooth (int): EMA smoothing span for the momentum line. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the TTM DataFrame (Momentum, Squeeze_On, Squeeze_Off) and a list of column names.

    The TTM Squeeze is calculated using Bollinger Bands and Keltner Channels:

    1. Calculate Bollinger Bands:
       Basis = SMA(Close, length)
       Upper BB = Basis + (std_dev * StdDev)
       Lower BB = Basis - (std_dev * StdDev)

    2. Calculate Keltner Channels:
       Basis = EMA(Close, length) (Implementation typically uses SMA of typical price or similar)
       ATR = Average True Range
       Upper KC = Basis + (atr_multiplier * ATR)
       Lower KC = Basis - (atr_multiplier * ATR)

    3. Identify Squeeze:
       Squeeze On = Bollinger Bands are INSIDE Keltner Channels.
       Squeeze Off = Bollinger Bands are OUTSIDE Keltner Channels.

    4. Calculate Momentum:
       Calculated using linear regression of price relative to a mean (or similar momentum proxy).
       (This implementation uses a specific linear regression of the typical price minus an average).

    Interpretation:
    - Squeeze On (Red dots): Volatility is low, market is consolidating. Preparing for a move.
    - Squeeze Off (Green dots): Volatility is expanding, breakout has occurred.
    - Momentum Histogram:
      - Cyan/Green: Bullish momentum (Price rising).
      - Red/Yellow: Bearish momentum (Price falling).

    Use Cases:
    - Breakout Trading: Enter when the squeeze fires (Squeeze On -> Squeeze Off).
    - Trend Direction: Follow the direction of the momentum histogram after a squeeze.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    length = int(parameters.get('length', 20))
    std_dev = float(parameters.get('std_dev', 2.0))
    atr_length = int(parameters.get('atr_length', 20))
    atr_multiplier = float(parameters.get('atr_multiplier', 1.5))
    smooth = int(parameters.get('smooth', 3))

    close_col = columns.get('close_col', 'Close')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    typical_price = (high + low + close) / 3

    # Bollinger Bands for squeeze detection
    sma = typical_price.rolling(window=length).mean()
    std = typical_price.rolling(window=length).std(ddof=0)
    upper_bb = sma + std_dev * std
    lower_bb = sma - std_dev * std

    # Keltner Channels for squeeze detection
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.rolling(window=atr_length).mean()

    ema_typical = typical_price.ewm(span=length, adjust=False).mean()
    upper_kc = ema_typical + atr_multiplier * atr
    lower_kc = ema_typical - atr_multiplier * atr

    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)

    def _momentum(arr: np.ndarray) -> float:
        x = np.arange(len(arr), dtype=float)
        if len(arr) < 2:
            return np.nan
        x_mean = x.mean()
        y_mean = arr.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return np.nan
        slope = ((x - x_mean) * (arr - y_mean)).sum() / denom
        intercept = y_mean - slope * x_mean
        fitted_last = slope * x[-1] + intercept
        return arr[-1] - fitted_last

    momentum = typical_price.rolling(window=length).apply(_momentum, raw=True)
    if smooth > 1:
        momentum = momentum.ewm(span=smooth, adjust=False).mean()

    momentum_col = f'TTM_MOM_{length}'
    squeeze_on_col = f'Squeeze_On_{length}'
    squeeze_off_col = f'Squeeze_Off_{length}'

    result = pd.DataFrame(
        {
            momentum_col: momentum,
            squeeze_on_col: squeeze_on.astype(bool),
            squeeze_off_col: squeeze_off.astype(bool),
        }
    )
    result.index = df.index

    columns_list = [momentum_col, squeeze_on_col, squeeze_off_col]
    return result, columns_list


def strategy_ttm(
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
    TTM (TTM Squeeze) - Momentum Zero Line Crossover Strategy
    
    LOGIC: Buy when TTM momentum crosses above zero, sell when crosses below.
    WHY: TTM Squeeze identifies consolidation (squeeze) followed by breakouts.
         Momentum histogram direction after squeeze indicates breakout direction.
    BEST MARKETS: All markets. Excellent for breakout trading after consolidation.
                  Stocks, forex, futures. Best when combined with squeeze signals.
    TIMEFRAME: Daily or 4-hour charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'length' (default 20), 'std_dev' (default 2.0),
                   'atr_length' (default 20), 'atr_multiplier' (default 1.5)
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
    
    length = int(parameters.get('length', 20))
    
    indicator_params = {
        "length": length,
        "std_dev": float(parameters.get('std_dev', 2.0)),
        "atr_length": int(parameters.get('atr_length', 20)),
        "atr_multiplier": float(parameters.get('atr_multiplier', 1.5)),
        "smooth": int(parameters.get('smooth', 3))
    }
    short_window_indicator = f'TTM_MOM_{length}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='ttm',
        parameters=indicator_params,
        figure=False
    )
    
    # Create zero line for crossover strategy
    data['zero_line'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, 'zero_line']
    
    return results, portfolio, indicator_cols_to_plot, data
