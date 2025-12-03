import pandas as pd


def vsi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volatility Switch Index (vsi), a binary indicator that identifies
    volatility regime changes by comparing current volatility to historical levels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_period (int): Short volatility period. Default is 10.
            - long_period (int): Long volatility period. Default is 50.
            - threshold (float): Threshold for regime switch. Default is 1.2.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): Close prices column. Default is 'Close'.

    Returns:
        tuple: VSI series and column names list.

    Calculation Steps:
    1. Calculate Short-Term Volatility:
       Short Vol = Standard Deviation(Close, short_period)
    2. Calculate Long-Term Volatility:
       Long Vol = Standard Deviation(Close, long_period)
    3. Calculate Ratio:
       Ratio = Short Vol / Long Vol
    4. Determine VSI:
       If Ratio > threshold, VSI = 1 (High Volatility Regime)
       Else, VSI = 0 (Low/Normal Volatility Regime)

    Interpretation:
    - VSI = 1: Market is experiencing elevated volatility compared to its history.
    - VSI = 0: Market is in a baseline volatility state.
    - Switches often accompany changes in market trend or condition.

    Use Cases:
    - Binary volatility regime identification.
    - Strategy switching (High Vol vs. Low Vol strategies).
    - Risk management (Reducing size when VSI is 1).
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    short_period = int(parameters.get('short_period', 10))
    long_period = int(parameters.get('long_period', 50))
    threshold = float(parameters.get('threshold', 1.2))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate short and long-term volatility
    short_vol = close.rolling(window=short_period).std()
    long_vol = close.rolling(window=long_period).std()
    
    # Calculate volatility ratio
    vol_ratio = short_vol / long_vol
    
    # Create binary switch
    vsi_values = (vol_ratio > threshold).astype(int)
    
    vsi_values.name = f'VSI_{short_period}_{long_period}_{threshold}'
    return vsi_values, [vsi_values.name]


def strategy_vsi(
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
    vsi (Volatility Switch Index) - Binary Regime Strategy
    
    LOGIC: Buy when vsi drops below lower threshold (low volatility regime),
           sell when rises above upper threshold (high volatility regime).
    WHY: vsi is a binary indicator that identifies volatility regime changes.
         vsi = 1 means high volatility, vsi = 0 means low/normal volatility.
    BEST MARKETS: All markets. Good for binary volatility regime identification.
    TIMEFRAME: Daily charts. 10/50 periods with 1.2 threshold is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_period', 'long_period', 'threshold',
                    'upper' (default 1), 'lower' (default 0)
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
    
    short_period = int(parameters.get('short_period', 10))
    long_period = int(parameters.get('long_period', 50))
    threshold = float(parameters.get('threshold', 1.2))
    upper = float(parameters.get('upper', 1))
    lower = float(parameters.get('lower', 0))
    price_col = 'Close'
    indicator_col = f'VSI_{short_period}_{long_period}_{threshold}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='vsi',
        parameters={"short_period": short_period, "long_period": long_period, "threshold": threshold},
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
