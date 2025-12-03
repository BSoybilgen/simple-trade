import pandas as pd


def bbw(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Bollinger Band Width (bbw), a volatility indicator that measures the
    width between the upper and lower Bollinger Bands, normalized by the middle band.
    It quantifies the expansion and contraction of volatility.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for SMA and standard deviation. Default is 20.
            - num_std (float): The number of standard deviations for the bands. Default is 2.0.
            - normalize (bool): Whether to normalize by middle band. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Bollinger Band Width series and a list of column names.

    The Bollinger Band Width is calculated in several steps:

    1. Calculate the middle band (Simple Moving Average):
       Middle Band = SMA(Close, window)

    2. Calculate the standard deviation:
       Std Dev = std(Close, window)

    3. Calculate the upper and lower bands:
       Upper Band = Middle Band + (num_std * Std Dev)
       Lower Band = Middle Band - (num_std * Std Dev)

    4. Calculate the band width:
       If normalize=True:
           BBW = ((Upper Band - Lower Band) / Middle Band) * 100
       If normalize=False:
           BBW = Upper Band - Lower Band

    The normalized BBW is expressed as a percentage of the middle band, making it
    comparable across different price levels and assets.

    Interpretation:
    - Low bbw: Low volatility, tight bands, consolidation phase
    - High bbw: High volatility, wide bands, trending or volatile phase
    - Decreasing bbw: Volatility contracting, "The Squeeze" forming
    - Increasing bbw: Volatility expanding, potential breakout occurring

    The Squeeze:
    - When BBW reaches extremely low levels, it indicates "The Squeeze"
    - The Squeeze is a period of very low volatility that often precedes
      a significant price movement (breakout)
    - Traders watch for BBW to start expanding after a squeeze as a signal
      that a new trend may be beginning

    Use Cases:

    - Volatility measurement: BBW provides a simple, normalized measure of
      current volatility levels.
    - The Squeeze identification: Identify periods of extremely low BBW as
      potential pre-breakout setups.
    - Breakout confirmation: Rising BBW confirms that a breakout is occurring
      with expanding volatility.
    - Trend strength: Wide BBW during trends indicates strong momentum, while
      narrowing BBW suggests weakening trends.
    - Mean reversion setups: Extremely high BBW values may indicate overextension
      and potential mean reversion opportunities.
    - Comparative analysis: Compare BBW across different assets to identify
      which are in consolidation vs. trending phases.
    - Entry timing: Enter positions when BBW starts expanding after a squeeze,
      exit when BBW reaches extreme highs.
    - Risk management: Adjust stop-losses based on BBW - wider stops during
      high BBW periods, tighter stops during low BBW.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    num_std = float(parameters.get('num_std', 2.0))
    normalize = bool(parameters.get('normalize', True))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate the middle band (SMA)
    middle_band = close.rolling(window=window).mean()
    
    # Calculate standard deviation
    std_dev = close.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    
    # Calculate Bollinger Band Width
    if normalize:
        # Normalized BBW as percentage of middle band
        bbw_values = ((upper_band - lower_band) / middle_band) * 100
        bbw_values.name = f'BBW_{window}_{num_std}'
    else:
        # Absolute BBW (not normalized)
        bbw_values = upper_band - lower_band
        bbw_values.name = f'BBW_{window}_{num_std}_Abs'
    
    columns_list = [bbw_values.name]
    return bbw_values, columns_list


def strategy_bbw(
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
    bbw (Bollinger Band Width) - Volatility Squeeze Strategy
    
    LOGIC: Buy when bbw drops below lower threshold (squeeze/consolidation),
           sell when rises above upper threshold (volatility expansion).
    WHY: bbw measures width of Bollinger Bands. Low bbw indicates "The Squeeze" -
         a period of low volatility that often precedes significant breakouts.
    BEST MARKETS: All markets. Excellent for identifying pre-breakout setups.
                  Combine with price action for breakout direction.
    TIMEFRAME: Daily charts. 20-period with 2 std dev is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'num_std' (default 2.0),
                    'upper' (default 10.0), 'lower' (default 4.0)
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
    
    window = int(parameters.get('window', 20))
    num_std = float(parameters.get('num_std', 2.0))
    upper = float(parameters.get('upper', 10.0))
    lower = float(parameters.get('lower', 4.0))
    price_col = 'Close'
    indicator_col = f'BBW_{window}_{num_std}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='bbw',
        parameters={"window": window, "num_std": num_std},
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
