import numpy as np
import pandas as pd


def ttm(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """Calculate the TTM Squeeze momentum indicator.

    Args:
        df (pd.DataFrame): Input price data containing high, low, and close columns.
        parameters (dict, optional): Calculation parameters.
            - length (int): Lookback window for Bollinger/Keltner calculations. Default is 20.
            - std_dev (float): Standard deviation multiplier for Bollinger Bands. Default is 2.0.
            - atr_length (int): Lookback window for ATR in Keltner Channels. Default is 20.
            - atr_multiplier (float): Multiplier applied to ATR for Keltner Channels. Default is 1.5.
            - smooth (int): EMA smoothing span for the momentum line. Default is 3.
        columns (dict, optional): Column overrides.
            - close_col (str): Column name for closing prices. Default is 'Close'.
            - high_col (str): Column name for high prices. Default is 'High'.
            - low_col (str): Column name for low prices. Default is 'Low'.

    Returns:
        tuple: (DataFrame with momentum and squeeze state, [column names])
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
