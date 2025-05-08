import pandas as pd
import numpy as np


def ichimoku(df: pd.DataFrame,
             tenkan_period: int = 9, kijun_period: int = 26,
             senkou_b_period: int = 52, displacement: int = 26,
             high_col: str = 'High', low_col: str = 'Low', close_col: str = 'Close') -> dict:
    """
    Calculates the Ichimoku Cloud indicators (Ichimoku Kinko Hyo).

    Ichimoku Kinko Hyo, or the Ichimoku Cloud, is a versatile indicator that defines
    support and resistance, identifies trend direction, gauges momentum, and provides
    trading signals.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high, low, and close columns.
        tenkan_period (int): Period for Tenkan-sen (Conversion Line). Default is 9.
        kijun_period (int): Period for Kijun-sen (Base Line). Default is 26.
        senkou_b_period (int): Period for Senkou Span B. Default is 52.
        displacement (int): Displacement period for Senkou Span A and B. Default is 26.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').
        close_col (str): The name of the close price column (default: 'Close').

    Returns:
        dict: A dictionary containing all Ichimoku components as pandas Series:
            - tenkan_sen: Conversion Line
            - kijun_sen: Base Line
            - senkou_span_a: Leading Span A
            - senkou_span_b: Leading Span B
            - chikou_span: Lagging Span

    The Ichimoku Cloud consists of five components:

    1. Tenkan-sen (Conversion Line):
       (highest high + lowest low) / 2 for the specified period (default: 9)

    2. Kijun-sen (Base Line):
       (highest high + lowest low) / 2 for the specified period (default: 26)

    3. Senkou Span A (Leading Span A):
       (Tenkan-sen + Kijun-sen) / 2, plotted ahead by the displacement period

    4. Senkou Span B (Leading Span B):
       (highest high + lowest low) / 2 for the specified period (default: 52),
       plotted ahead by the displacement period

    5. Chikou Span (Lagging Span):
       Close price plotted back by the displacement period

    Use Cases:

    - Trend identification: When price is above the cloud, the trend is up.
      When price is below the cloud, the trend is down.

    - Support and resistance: The cloud serves as dynamic support and resistance areas.

    - Signal generation:
      - Bullish signal: When Tenkan-sen crosses above Kijun-sen
      - Bearish signal: When Tenkan-sen crosses below Kijun-sen

    - Strength confirmation: The thicker the cloud, the stronger the support/resistance.
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # Calculate Tenkan-sen (Conversion Line)
    tenkan_sen = _donchian_channel_middle(df, tenkan_period, high_col=high_col, low_col=low_col)

    # Calculate Kijun-sen (Base Line)
    kijun_sen = _donchian_channel_middle(df, kijun_period, high_col=high_col, low_col=low_col)

    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Calculate Senkou Span B (Leading Span B)
    senkou_span_b = _donchian_channel_middle(df, senkou_b_period, high_col=high_col, low_col=low_col).shift(displacement)

    # Calculate Chikou Span (Lagging Span)
    chikou_span = close.shift(-displacement)

    df = pd.DataFrame({
        f'tenkan_sen_{tenkan_period}': tenkan_sen,
        f'kijun_sen_{kijun_period}': kijun_sen,
        f'senkou_span_a_{tenkan_period}_{kijun_period}': senkou_span_a,
        f'senkou_span_b_{senkou_b_period}': senkou_span_b,
        f'chikou_span_{displacement}': chikou_span
    })
    df.index = close.index
    return df


def _donchian_channel_middle(df: pd.DataFrame, period: int, high_col: str = 'High', low_col: str = 'Low') -> pd.Series:
    """
    Calculate the middle line of the Donchian Channel.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high and low columns.
        period (int): The period for the calculation.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').

    Returns:
        pd.Series: The middle line of the Donchian Channel.
    """
    high = df[high_col]
    low = df[low_col]
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return (highest_high + lowest_low) / 2


def tenkan_sen(df: pd.DataFrame, period: int = 9, high_col: str = 'High', low_col: str = 'Low') -> pd.Series:
    """
    Calculates Tenkan-sen (Conversion Line) component of Ichimoku Cloud.

    This is the midpoint of the highest high and lowest low over the specified period.
    It represents a shorter-term trend indicator.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high and low columns.
        period (int): The period for calculation. Default is 9.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').

    Returns:
        pd.Series: The Tenkan-sen (Conversion Line) values.
    """
    high = df[high_col]
    low = df[low_col]
    return _donchian_channel_middle(high, low, period)


def kijun_sen(df: pd.DataFrame, period: int = 26, high_col: str = 'High', low_col: str = 'Low') -> pd.Series:
    """
    Calculates Kijun-sen (Base Line) component of Ichimoku Cloud.

    This is the midpoint of the highest high and lowest low over the specified period.
    It represents a longer-term trend indicator and can act as a dynamic support/resistance level.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high and low columns.
        period (int): The period for calculation. Default is 26.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').

    Returns:
        pd.Series: The Kijun-sen (Base Line) values.
    """
    high = df[high_col]
    low = df[low_col]
    return _donchian_channel_middle(high, low, period)


def senkou_span_a(df: pd.DataFrame, 
                 tenkan_period: int = 9, kijun_period: int = 26, 
                 displacement: int = 26,
                 high_col: str = 'High', low_col: str = 'Low') -> pd.Series:
    """
    Calculates Senkou Span A (Leading Span A) component of Ichimoku Cloud.

    This is the midpoint of Tenkan-sen and Kijun-sen, shifted forward by the displacement period.
    It forms one of the boundaries of the Ichimoku Cloud.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high and low columns.
        tenkan_period (int): Period for Tenkan-sen calculation. Default is 9.
        kijun_period (int): Period for Kijun-sen calculation. Default is 26.
        displacement (int): Number of periods to shift forward. Default is 26.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').

    Returns:
        pd.Series: The Senkou Span A (Leading Span A) values.
    """
    tenkan = tenkan_sen(high, low, tenkan_period)
    kijun = kijun_sen(high, low, kijun_period)
    return ((tenkan + kijun) / 2).shift(displacement)


def senkou_span_b(high: pd.Series, low: pd.Series, 
                 period: int = 52, displacement: int = 26) -> pd.Series:
    """
    Calculates Senkou Span B (Leading Span B) component of Ichimoku Cloud.

    This is the midpoint of the highest high and lowest low over a longer period,
    shifted forward by the displacement period. It forms the other boundary of the Ichimoku Cloud.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high and low columns.
        period (int): The period for calculation. Default is 52.
        displacement (int): Number of periods to shift forward. Default is 26.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').

    Returns:
        pd.Series: The Senkou Span B (Leading Span B) values.
    """
    high = df[high_col]
    low = df[low_col]
    return _donchian_channel_middle(high, low, period).shift(displacement)


def chikou_span(df: pd.DataFrame, displacement: int = 26, close_col: str = 'Close') -> pd.Series:
    """
    Calculates Chikou Span (Lagging Span) component of Ichimoku Cloud.

    This is the closing price shifted backward by the displacement period.
    It is used to confirm trends and potential reversal points.

    Args:
        close (pd.Series): The close prices.
        displacement (int): Number of periods to shift backward. Default is 26.

    Returns:
        pd.Series: The Chikou Span (Lagging Span) values.
    """
    close = df[close_col]
    return close.shift(-displacement)