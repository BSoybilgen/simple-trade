import pandas as pd


def kur(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Kurtosis (kur), a statistical measure that describes the shape of a distribution's
    tails in relation to its overall shape. It measures the "tailedness" of the price distribution.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Kurtosis series and a list of column names.

    Calculation Steps:
    1. Calculate rolling kurtosis over the specified window.
    2. Kurtosis measures the fourth standardized moment.

    Interpretation:
    - Kurtosis > 3 (Leptokurtic): Heavy tails, more outliers, higher risk of extreme moves.
    - Kurtosis = 3 (Mesokurtic): Normal distribution-like tails.
    - Kurtosis < 3 (Platykurtic): Light tails, fewer outliers, lower risk of extreme moves.
    - Excess kurtosis = Kurtosis - 3 (measures deviation from normal).

    Use Cases:
    - Risk assessment (fat tail detection).
    - Volatility regime identification.
    - Options pricing adjustments.
    - Portfolio risk management.
    - Identifying potential market stress.

    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling kurtosis
    kur_values = close.rolling(window=window).kurt()
    
    kur_values.name = f'KUR_{window}'
    columns_list = [kur_values.name]
    return kur_values, columns_list


def strategy_kur(
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
    kur (Kurtosis) - Fat Tail Detection Strategy
    
    LOGIC: Buy when kur drops below lower threshold (normal distribution),
           sell when rises above upper threshold (fat tails, high risk).
    WHY: High kurtosis indicates increased probability of extreme price moves.
         Low kurtosis suggests more predictable, normal price behavior.
    BEST MARKETS: All markets. Useful for risk-adjusted position sizing.
    TIMEFRAME: Daily charts. 20-period is standard.
    NOTE: Uses rolling percentile bands since kurtosis values vary by asset.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'upper_pct' (default 80),
                    'lower_pct' (default 20), 'lookback' (default 100)
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
    upper_pct = float(parameters.get('upper_pct', 80))
    lower_pct = float(parameters.get('lower_pct', 20))
    lookback = int(parameters.get('lookback', 100))
    price_col = 'Close'
    indicator_col = f'KUR_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='kur',
        parameters={"window": window},
        figure=False
    )
    
    # Calculate rolling percentile bands for kurtosis
    data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(upper_pct / 100)
    data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(lower_pct / 100)
    
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
