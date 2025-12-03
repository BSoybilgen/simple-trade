import pandas as pd


def ich(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the ich indicators (ich).
    ich is a versatile indicator that defines support and resistance, 
    identifies trend direction, gauges momentum, and provides trading signals.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - tenkan_period (int): Period for Tenkan-sen. Default is 9.
            - kijun_period (int): Period for Kijun-sen. Default is 26.
            - senkou_b_period (int): Period for Senkou Span B. Default is 52.
            - displacement (int): Displacement for spans. Default is 26.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame with ich components and a list of column names.

    Calculation Steps:
    1. Tenkan-sen (Conversion Line):
       (Highest High + Lowest Low) / 2 over tenkan_period
    2. Kijun-sen (Base Line):
       (Highest High + Lowest Low) / 2 over kijun_period
    3. Senkou Span A (Leading Span A):
       (Tenkan-sen + Kijun-sen) / 2, plotted ahead by displacement
    4. Senkou Span B (Leading Span B):
       (Highest High + Lowest Low) / 2 over senkou_b_period, plotted ahead by displacement
    5. Chikou Span (Lagging Span):
       Close price plotted back by displacement

    Interpretation:
    - Trend: Price > Cloud = Uptrend; Price < Cloud = Downtrend.
    - Signals: Tenkan-sen crossing Kijun-sen (Golden/Death Cross).
    - Support/Resistance: The Cloud (Kumo) acts as dynamic S/R.
    - Cloud Thickness: Thicker cloud means stronger S/R.

    Use Cases:
    - Trend identification: Determine overall market direction.
    - Entry/Exit: Crossovers and cloud breakouts.
    - Stop Loss: Placing stops on the other side of the Kijun-sen or Cloud.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    tenkan_period = int(parameters.get('tenkan_period', 9))
    kijun_period = int(parameters.get('kijun_period', 26))
    senkou_b_period = int(parameters.get('senkou_b_period', 52))
    displacement = int(parameters.get('displacement', 26))
    
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

    df_out = pd.DataFrame({
        f'tenkan_sen_{tenkan_period}': tenkan_sen,
        f'kijun_sen_{kijun_period}': kijun_sen,
        f'senkou_span_a_{tenkan_period}_{kijun_period}': senkou_span_a,
        f'senkou_span_b_{senkou_b_period}': senkou_span_b,
        f'chikou_span_{displacement}': chikou_span
    })
    df_out.index = close.index
    columns_list = list(df_out.columns)
    return df_out, columns_list


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
    Calculates Tenkan-sen (Conversion Line) component of ich.

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
    return _donchian_channel_middle(df, period, high_col=high_col, low_col=low_col)


def kijun_sen(df: pd.DataFrame, period: int = 26, high_col: str = 'High', low_col: str = 'Low') -> pd.Series:
    """
    Calculates Kijun-sen (Base Line) component of ich.

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
    return _donchian_channel_middle(df, period, high_col=high_col, low_col=low_col)


def senkou_span_a(df: pd.DataFrame,
                 tenkan_period: int = 9, kijun_period: int = 26,
                 displacement: int = 26,
                 high_col: str = 'High', low_col: str = 'Low') -> pd.Series:
    """
    Calculates Senkou Span A (Leading Span A) component of ich.

    This is the midpoint of Tenkan-sen and Kijun-sen, shifted forward by the displacement period.
    It forms one of the boundaries of the ich Cloud.

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
    tenkan = tenkan_sen(df, period=tenkan_period, high_col=high_col, low_col=low_col)
    kijun = kijun_sen(df, period=kijun_period, high_col=high_col, low_col=low_col)
    return ((tenkan + kijun) / 2).shift(displacement)


def senkou_span_b(df: pd.DataFrame,
                 period: int = 52, displacement: int = 26,
                 high_col: str = 'High', low_col: str = 'Low') -> pd.Series:
    """
    Calculates Senkou Span B (Leading Span B) component of ich.

    This is the midpoint of the highest high and lowest low over a longer period,
    shifted forward by the displacement period. It forms the other boundary of the ich Cloud.

    Args:
        df (pd.DataFrame): The dataframe containing price data. Must have high and low columns.
        period (int): The period for calculation. Default is 52.
        displacement (int): Number of periods to shift forward. Default is 26.
        high_col (str): The name of the high price column (default: 'High').
        low_col (str): The name of the low price column (default: 'Low').

    Returns:
        pd.Series: The Senkou Span B (Leading Span B) values.
    """
    return _donchian_channel_middle(df, period, high_col=high_col, low_col=low_col).shift(displacement)


def chikou_span(df: pd.DataFrame, displacement: int = 26, close_col: str = 'Close') -> pd.Series:
    """
    Calculates Chikou Span (Lagging Span) component of ich.

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


def strategy_ich(
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
    ich (Ichimoku Cloud) - Tenkan/Kijun Crossover Strategy
    
    LOGIC: Buy when Tenkan-sen crosses above Kijun-sen, sell when crosses below.
    WHY: ich is a comprehensive indicator showing support, resistance, trend,
         and momentum. Tenkan/Kijun crossovers are classic entry signals.
    BEST MARKETS: Trending markets. Stocks, forex, crypto. One of the most
                  complete technical analysis systems available.
    TIMEFRAME: Daily charts. Standard settings: 9/26/52.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'tenkan_period' (default 9), 'kijun_period' (default 26)
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
    
    tenkan_period = int(parameters.get('tenkan_period', 9))
    kijun_period = int(parameters.get('kijun_period', 26))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='ich',
        parameters={
            "tenkan_period": tenkan_period,
            "kijun_period": kijun_period
        },
        figure=False
    )
    
    short_window_indicator = f'tenkan_sen_{tenkan_period}'
    long_window_indicator = f'kijun_sen_{kijun_period}'
    
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