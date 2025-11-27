import pandas as pd

from .indicator_handler import compute_indicator
from .cross_trade import CrossTradeBacktester
from .band_trade import BandTradeBacktester
from .plot_test import BacktestPlotter


# Strategy catalog with descriptions organized by category
_STRATEGY_CATALOG = {
    'momentum': {
        'awo': 'AWO (Awesome Oscillator) - Zero Line Crossover Strategy',
        'bop': 'BOP (Balance of Power) - Zero Line Crossover Strategy',
        'cci': 'CCI (Commodity Channel Index) - Mean Reversion Strategy',
        'cmo': 'CMO (Chande Momentum Oscillator) - Mean Reversion Strategy',
        'cog': 'COG (Center of Gravity) - Signal Line Crossover Strategy',
        'crs': 'CRS (Connors RSI) - Mean Reversion Strategy',
        'dpo': 'DPO (Detrended Price Oscillator) - Zero Line Crossover Strategy',
        'eri': 'ERI (Elder-Ray Index) - Bear Power Zero Line Crossover Strategy',
        'fis': 'FIS (Fisher Transform) - Zero Line Crossover Strategy',
        'imi': 'IMI (Intraday Momentum Index) - Mean Reversion Strategy',
        'kst': 'KST (Know Sure Thing) - Signal Line Crossover Strategy',
        'lsi': 'LSI (Laguerre RSI) - Mean Reversion Strategy',
        'mac': 'MAC (MACD) - Signal Line Crossover Strategy',
        'msi': 'MSI (Momentum Strength Index) - Mean Reversion Strategy',
        'pgo': 'PGO (Pretty Good Oscillator) - Mean Reversion Strategy',
        'ppo': 'PPO (Percentage Price Oscillator) - Signal Line Crossover Strategy',
        'psy': 'PSY (Psychological Line) - Mean Reversion Strategy',
        'qst': 'QST (Qstick) - Zero Line Crossover Strategy',
        'rmi': 'RMI (Relative Momentum Index) - Mean Reversion Strategy',
        'roc': 'ROC (Rate of Change) - Zero Line Crossover Strategy',
        'rsi': 'RSI (Relative Strength Index) - Mean Reversion Strategy',
        'rvg': 'RVG (Relative Vigor Index) - Signal Line Crossover Strategy',
        'sri': 'SRI (Stochastic RSI) - Mean Reversion Strategy',
        'stc': 'STC (Schaff Trend Cycle) - Mean Reversion Strategy',
        'sto': 'STO (Stochastic Oscillator) - Mean Reversion Strategy',
        'tsi': 'TSI (True Strength Index) - Signal Line Crossover Strategy',
        'ttm': 'TTM (TTM Squeeze) - Momentum Histogram Strategy',
        'ult': 'ULT (Ultimate Oscillator) - Mean Reversion Strategy',
        'vor': 'VOR (Vortex Indicator) - VI+/VI- Crossover Strategy',
        'wil': 'WIL (Williams %R) - Mean Reversion Strategy',
    },
    'trend': {
        'ads': 'ADS (Adaptive Deviation-Scaled MA) - Dual MA Crossover Strategy',
        'adx': 'ADX (Average Directional Index) - Trend Strength + MA Strategy',
        'alm': 'ALM (Arnaud Legoux MA) - Dual MA Crossover Strategy',
        'ama': 'AMA (Adaptive Moving Average) - Dual MA Crossover Strategy',
        'aro': 'ARO (Aroon) - Aroon Up/Down Crossover Strategy',
        'dem': 'DEM (Double EMA) - Dual MA Crossover Strategy',
        'eac': 'EAC (Ehlers Adaptive CyberCycle) - Dual Cycle Crossover Strategy',
        'eit': 'EIT (Ehlers Instantaneous Trendline) - Dual Trendline Crossover Strategy',
        'ema': 'EMA (Exponential Moving Average) - Dual MA Crossover Strategy',
        'fma': 'FMA (Fractal Adaptive MA) - Price Cross Strategy',
        'gma': 'GMA (Guppy Multiple MA) - Short/Long Group Crossover Strategy',
        'hma': 'HMA (Hull Moving Average) - Dual MA Crossover Strategy',
        'htt': 'HTT (Hilbert Transform Trendline) - Dual Trendline Crossover Strategy',
        'ich': 'ICH (Ichimoku Cloud) - Tenkan/Kijun Crossover Strategy',
        'jma': 'JMA (Jurik Moving Average) - Dual MA Crossover Strategy',
        'kma': 'KMA (Kaufman Adaptive MA) - Dual MA Crossover Strategy',
        'lsm': 'LSM (Least Squares MA) - Dual MA Crossover Strategy',
        'mgd': 'MGD (McGinley Dynamic) - Dual MA Crossover Strategy',
        'psa': 'PSA (Parabolic SAR) - Price Cross Strategy',
        'sma': 'SMA (Simple Moving Average) - Dual MA Crossover Strategy',
        'soa': 'SOA (Smoothed Moving Average) - Dual MA Crossover Strategy',
        'str': 'STR (SuperTrend) - Price Cross Strategy',
        'swm': 'SWM (Sine Weighted MA) - Dual MA Crossover Strategy',
        'tem': 'TEM (Triple EMA) - Dual MA Crossover Strategy',
        'tma': 'TMA (Triangular MA) - Dual MA Crossover Strategy',
        'tri': 'TRI (TRIX) - Zero Line Crossover Strategy',
        'vid': 'VID (Variable Index Dynamic Average) - Dual MA Crossover Strategy',
        'wma': 'WMA (Weighted Moving Average) - Dual MA Crossover Strategy',
        'zma': 'ZMA (Zero-Lag MA) - Dual MA Crossover Strategy',
    },
    'volatility': {
        'acb': 'ACB (Acceleration Bands) - Band Breakout Strategy',
        'atp': 'ATP (Average True Price) - Band Mean Reversion Strategy',
        'atr': 'ATR (Average True Range) - Percentile Band Strategy',
        'bbw': 'BBW (Bollinger Band Width) - Volatility Squeeze Strategy',
        'bol': 'BOL (Bollinger Bands) - Band Mean Reversion Strategy',
        'cha': 'CHA (Chaikin Volatility) - Volatility Expansion Strategy',
        'cho': 'CHO (Choppiness Index) - Trend/Range Filter Strategy',
        'don': 'DON (Donchian Channels) - Band Breakout Strategy',
        'dvi': 'DVI (DV Intermediate Oscillator) - Mean Reversion Strategy',
        'efr': 'EFR (Elder Force Index) - Zero Line Crossover Strategy',
        'fdi': 'FDI (Fractal Dimension Index) - Trend/Range Filter Strategy',
        'grv': 'GRV (Garman-Klass Volatility) - Percentile Band Strategy',
        'hav': 'HAV (Historical Average Volatility) - Percentile Band Strategy',
        'hiv': 'HIV (Historical Intraday Volatility) - Percentile Band Strategy',
        'kel': 'KEL (Keltner Channels) - Band Mean Reversion Strategy',
        'mad': 'MAD (Median Absolute Deviation) - Percentile Band Strategy',
        'mai': 'MAI (Mass Index) - Reversal Bulge Strategy',
        'nat': 'NAT (Normalized ATR) - Percentile Band Strategy',
        'pav': 'PAV (Parkinson Volatility) - Percentile Band Strategy',
        'pcw': 'PCW (Price Channel Width) - Volatility Squeeze Strategy',
        'pro': 'PRO (Projection Oscillator) - Mean Reversion Strategy',
        'rsv': 'RSV (Rogers-Satchell Volatility) - Percentile Band Strategy',
        'rvi': 'RVI (Relative Volatility Index) - Mean Reversion Strategy',
        'std': 'STD (Standard Deviation) - Percentile Band Strategy',
        'svi': 'SVI (Stochastic Volatility Index) - Mean Reversion Strategy',
        'tsv': 'TSV (True Strength Volatility) - Mean Reversion Strategy',
        'uli': 'ULI (Ulcer Index) - Risk-Based Strategy',
        'vhf': 'VHF (Vertical Horizontal Filter) - Trend/Range Filter Strategy',
        'vra': 'VRA (Volatility Ratio) - Breakout Strategy',
        'vqi': 'VQI (Volatility Quality Index) - Trend Quality Strategy',
        'vsi': 'VSI (Volatility Switch Index) - Regime Detection Strategy',
    },
    'volume': {
        'adl': 'ADL (Accumulation/Distribution Line) - SMA Crossover Strategy',
        'ado': 'ADO (Accumulation/Distribution Oscillator) - Zero Line Crossover Strategy',
        'bwm': 'BWM (Buff Weighted Moving Average) - SMA Crossover Strategy',
        'cmf': 'CMF (Chaikin Money Flow) - Zero Line Crossover Strategy',
        'emv': 'EMV (Ease of Movement) - Zero Line Crossover Strategy',
        'foi': 'FOI (Force Index) - Zero Line Crossover Strategy',
        'fve': 'FVE (Finite Volume Elements) - Zero Line Crossover Strategy',
        'kvo': 'KVO (Klinger Volume Oscillator) - Signal Line Crossover Strategy',
        'mfi': 'MFI (Money Flow Index) - Mean Reversion Strategy',
        'nvi': 'NVI (Negative Volume Index) - SMA Crossover Strategy',
        'obv': 'OBV (On-Balance Volume) - SMA Crossover Strategy',
        'pvi': 'PVI (Positive Volume Index) - SMA Crossover Strategy',
        'pvo': 'PVO (Percentage Volume Oscillator) - Signal Line Crossover Strategy',
        'vfi': 'VFI (Volume Flow Indicator) - Zero Line Crossover Strategy',
        'vma': 'VMA (Volume Moving Average) - Volume Cross Strategy',
        'voo': 'VOO (Volume Oscillator) - Zero Line Crossover Strategy',
        'vpt': 'VPT (Volume Price Trend) - SMA Crossover Strategy',
        'vro': 'VRO (Volume Rate of Change) - Zero Line Crossover Strategy',
        'vwa': 'VWA (Volume Weighted Average Price) - Rolling VWAP Cross Strategy',
        'wad': 'WAD (Williams Accumulation/Distribution) - SMA Crossover Strategy',
    },
}


def list_strategies(category: str = None, return_dict: bool = False) -> dict | None:
    """List all available premade backtest strategies with their descriptions.
    
    This function provides a comprehensive catalog of all backtest strategies available
    in the premade_backtest function, organized by category (momentum, trend, volatility, volume).
    
    Args:
        category: Optional filter by category. Options: 'momentum', 'trend', 'volatility', 'volume'.
                 If None, returns all strategies.
        return_dict: If True, returns a dictionary instead of printing. Default is False.
    
    Returns:
        dict or None: If return_dict=True, returns a nested dictionary with structure:
                     {category: {strategy_name: description}}
                     Otherwise, prints the strategies and returns None.
    
    Example:
        >>> list_strategies()  # Print all strategies
        >>> list_strategies(category='momentum')  # Print only momentum strategies
        >>> strategies = list_strategies(return_dict=True)  # Get dictionary of all strategies
    """
    # Filter by category if specified
    if category:
        if category.lower() not in _STRATEGY_CATALOG:
            valid_categories = ', '.join(_STRATEGY_CATALOG.keys())
            raise ValueError(f"Invalid category '{category}'. Valid options: {valid_categories}")
        filtered_catalog = {category.lower(): _STRATEGY_CATALOG[category.lower()]}
    else:
        filtered_catalog = _STRATEGY_CATALOG
    
    # Return dictionary if requested
    if return_dict:
        return filtered_catalog
    
    # Otherwise, print formatted output
    print("\n" + "="*80)
    print("AVAILABLE PREMADE BACKTEST STRATEGIES")
    print("="*80 + "\n")
    
    for cat_name, strategies in filtered_catalog.items():
        print(f"\n{'─'*80}")
        print(f"{cat_name.upper()} STRATEGIES ({len(strategies)} total)")
        print(f"{'─'*80}\n")
        
        for strategy_name, description in sorted(strategies.items()):
            print(f"  • {strategy_name.upper()}")
            print(f"    {description}")
            print()
    
    print("="*80)
    print(f"Total: {sum(len(strategies) for strategies in filtered_catalog.values())} strategies")
    print("="*80 + "\n")
    
    return None


def premade_backtest(data:pd.DataFrame, strategy_name:str, parameters:dict=None):
    
    if parameters is None:
        parameters = {}

    initial_cash = float(parameters.get('initial_cash', 10000.0))
    commission_long = float(parameters.get('commission_long', 0.001))
    commission_short = float(parameters.get('commission_short', 0.001))
    short_borrow_fee_inc_rate = float(parameters.get('short_borrow_fee_inc_rate', 0.0))
    long_borrow_fee_inc_rate = float(parameters.get('long_borrow_fee_inc_rate', 0.0))
    long_entry_pct_cash = float(parameters.get('long_entry_pct_cash', 1))
    short_entry_pct_cash = float(parameters.get('short_entry_pct_cash', 1))
    trading_type = str(parameters.get('trading_type', 'long'))
    day1_position = str(parameters.get('day1_position', 'none'))
    risk_free_rate = float(parameters.get('risk_free_rate', 0.0))

    plotter = BacktestPlotter()
    cross_backtester = CrossTradeBacktester(initial_cash=initial_cash, commission_long=commission_long, 
                                            commission_short=commission_short, short_borrow_fee_inc_rate=short_borrow_fee_inc_rate, 
                                            long_borrow_fee_inc_rate=long_borrow_fee_inc_rate)
    band_backtester = BandTradeBacktester(initial_cash=initial_cash, commission_long=commission_long, 
                                          commission_short=commission_short, short_borrow_fee_inc_rate=short_borrow_fee_inc_rate, 
                                          long_borrow_fee_inc_rate=long_borrow_fee_inc_rate)
    parameters_indicators = dict()

    fig_control = int(parameters.get('fig_control', 0))
    
    # Initialize default values for results and portfolio
    results = None
    portfolio = pd.DataFrame()
    fig = None


    # ==================== MOMENTUM STRATEGIES ====================
    if strategy_name=='awo':
        # AWO (Awesome Oscillator) - Zero Line Crossover Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when AO crosses above zero (bullish momentum), sell when crosses below zero.
        # WHY: AO measures market momentum by comparing recent price action to historical.
        #      Positive AO = fast momentum > slow momentum (bullish), negative = bearish.
        # BEST MARKETS: Trending markets with clear directional moves. Works well on stocks,
        #               forex, and commodities. Less effective in choppy/ranging markets.
        # TIMEFRAME: Daily or weekly charts preferred. Intraday can generate false signals.
        fast_window = int(parameters.get('fast_window', 5))
        slow_window = int(parameters.get('slow_window', 34))
        parameters_indicators["fast_window"] = fast_window
        parameters_indicators["slow_window"] = slow_window
        short_window_indicator=f'AO_{fast_window}_{slow_window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='awo',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='bop':
        # BOP (Balance of Power) - Zero Line Crossover Strategy
        # ------------------------------------------------------
        # LOGIC: Buy when BOP crosses above zero (buyers in control), sell when crosses below.
        # WHY: BOP measures the strength of buyers vs sellers by comparing close-open to high-low.
        #      Positive BOP = buyers driving prices up, negative = sellers in control.
        # BEST MARKETS: Stocks and indices with clear institutional participation.
        #               Works well in trending markets with strong volume.
        # TIMEFRAME: Daily charts. Smoothed version (default) reduces noise.
        window = int(parameters.get('window', 14))
        smooth = parameters.get('smooth', True)
        parameters_indicators["window"] = window
        parameters_indicators["smooth"] = smooth
        
        if smooth:
            short_window_indicator = f'BOP_{window}'
        else:
            short_window_indicator = 'BOP'
        
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='bop',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='cci':
        # CCI (Commodity Channel Index) - Mean Reversion Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when CCI drops below lower threshold (oversold), sell when rises above upper.
        # WHY: CCI measures deviation from statistical mean. Extreme readings suggest price
        #      has moved too far and is likely to revert. Originally designed for commodities.
        # BEST MARKETS: Ranging/sideways markets. Commodities, forex pairs, and stocks in
        #               consolidation phases. Avoid strong trending markets.
        # TIMEFRAME: Works on all timeframes. Higher thresholds (±150-200) for volatile assets.
        window = int(parameters.get('window', 20))
        constant = float(parameters.get('constant', 0.015))
        upper = int(parameters.get('upper', 150))
        lower = int(parameters.get('lower', -150))
        parameters_indicators["window"] = window
        parameters_indicators["constant"] = constant
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'CCI_{window}_{constant}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cci',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'CCI_{window}_{constant}', 'lower', 'upper']

    
    elif strategy_name=='cmo':
        # CMO (Chande Momentum Oscillator) - Mean Reversion Strategy
        # -----------------------------------------------------------
        # LOGIC: Buy when CMO drops below -50 (oversold), sell when rises above +50 (overbought).
        # WHY: CMO compares sum of gains to losses, oscillating between -100 and +100.
        #      Unlike RSI, it's not bounded by smoothing, making it more responsive.
        # BEST MARKETS: Range-bound markets and mean-reverting assets. Stocks in consolidation,
        #               forex pairs, and ETFs. Less effective in strong trends.
        # TIMEFRAME: Daily or 4-hour charts. Adjust thresholds based on asset volatility.
        window = int(parameters.get('window', 14))
        upper = int(parameters.get('upper', 50))
        lower = int(parameters.get('lower', -50))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'CMO_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cmo',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'CMO_{window}', 'lower', 'upper']

    
    elif strategy_name=='cog':
        # COG (Center of Gravity) - Signal Line Crossover Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when COG crosses above its signal line, sell when crosses below.
        # WHY: COG is a zero-lag indicator designed to identify turning points early.
        #      It acts like a leading indicator, spotting reversals before they happen.
        # BEST MARKETS: Cyclical markets and assets with regular oscillations. Works well
        #               on forex, indices, and stocks with predictable cycles.
        # TIMEFRAME: Short-term trading (intraday to daily). Sensitive to noise on very short TFs.
        window = int(parameters.get('window', 10))
        signal_window = int(parameters.get('signal_window', 3))
        parameters_indicators["window"] = window
        short_window_indicator=f'COG_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cog',
        parameters=parameters_indicators,
        figure=False)

        # Create signal line as SMA of COG
        data['COG_Signal'] = data[short_window_indicator].rolling(window=signal_window).mean()
        long_window_indicator = 'COG_Signal'

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='crs':
        # CRS (Connors RSI) - Mean Reversion Strategy
        # --------------------------------------------
        # LOGIC: Buy when CRSI drops below 10 (extreme oversold), sell when rises above 90.
        # WHY: CRSI combines 3 components (RSI, streak RSI, percent rank) for short-term
        #      mean reversion. Designed by Larry Connors for identifying pullback opportunities.
        # BEST MARKETS: Liquid stocks and ETFs in uptrends. Best for buying dips in bull markets.
        #               SPY, QQQ, and large-cap stocks. Avoid in bear markets or downtrends.
        # TIMEFRAME: Daily charts. Designed for short-term trades (2-5 day holding periods).
        rsi_window = int(parameters.get('rsi_window', 3))
        streak_window = int(parameters.get('streak_window', 2))
        rank_window = int(parameters.get('rank_window', 100))
        upper = int(parameters.get('upper', 90))
        lower = int(parameters.get('lower', 10))
        parameters_indicators["rsi_window"] = rsi_window
        parameters_indicators["streak_window"] = streak_window
        parameters_indicators["rank_window"] = rank_window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'CRSI_{rsi_window}_{streak_window}_{rank_window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='crs',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'CRSI_{rsi_window}_{streak_window}_{rank_window}', 'lower', 'upper']

    
    elif strategy_name=='dpo':
        # DPO (Detrended Price Oscillator) - Zero Line Crossover Strategy
        # ----------------------------------------------------------------
        # LOGIC: Buy when DPO crosses above zero (price above displaced MA), sell when crosses below.
        # WHY: DPO removes trend to isolate cycles. Positive DPO = price above its historical average,
        #      negative = below. Zero crossings signal cycle turning points.
        # BEST MARKETS: Cyclical markets and assets with regular oscillations. Stocks, commodities,
        #               and indices with identifiable cycles. Less effective in strong trending markets.
        # TIMEFRAME: Daily charts. 20-period is standard. Useful for identifying cycle peaks/troughs.
        window = int(parameters.get('window', 20))
        parameters_indicators["window"] = window
        short_window_indicator = f'DPO_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='dpo',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='eri':
        # ERI (Elder-Ray Index) - Bear Power Zero Line Crossover Strategy
        # ----------------------------------------------------------------
        # LOGIC: Buy when Bear Power crosses above zero (buyers gaining control), sell when crosses below.
        # WHY: Elder-Ray measures buying/selling pressure. Bear Power above zero means lows are above
        #      the EMA (bullish). Below zero means sellers pushing prices below consensus value.
        # BEST MARKETS: Trending stocks and indices with clear institutional participation.
        #               Works well when combined with EMA trend filter.
        # TIMEFRAME: Daily charts. 13-period EMA is standard (Elder's recommendation).
        window = int(parameters.get('window', 13))
        parameters_indicators["window"] = window
        short_window_indicator = f'ERI_BEAR_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='eri',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'ERI_BULL_{window}', f'ERI_BEAR_{window}', 'zero_line']

    
    elif strategy_name=='fis':
        # FIS (Fisher Transform) - Zero Line Crossover Strategy
        # ------------------------------------------------------
        # LOGIC: Buy when Fisher crosses above zero (bullish), sell when crosses below.
        # WHY: Fisher Transform converts prices to Gaussian distribution, creating sharp
        #      turning points. Zero crossings indicate momentum shifts with clear signals.
        # BEST MARKETS: Trending markets with clear reversals. Forex, stocks, and futures.
        #               Creates sharp peaks/troughs making reversals easier to identify.
        # TIMEFRAME: Daily or 4-hour charts. 9-period is common. Good for swing trading.
        window = int(parameters.get('window', 9))
        parameters_indicators["window"] = window
        short_window_indicator = f'FISH_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='fis',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='imi':
        # IMI (Intraday Momentum Index) - Mean Reversion Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when IMI drops below lower threshold (oversold), sell when above upper.
        # WHY: IMI combines candlestick analysis with RSI logic. Measures buying vs selling
        #      pressure within each bar. High IMI = strong intraday buyers, low = sellers.
        # BEST MARKETS: Stocks and ETFs with significant intraday range. Range-bound markets.
        #               Good for identifying exhaustion in short-term moves.
        # TIMEFRAME: Daily charts. 14-period is standard. Works well for swing trading reversals.
        window = int(parameters.get('window', 14))
        upper = int(parameters.get('upper', 70))
        lower = int(parameters.get('lower', 30))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'IMI_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='imi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'IMI_{window}', 'lower', 'upper']

    
    elif strategy_name=='kst':
        # KST (Know Sure Thing) - Signal Line Crossover Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when KST crosses above its signal line, sell when crosses below.
        # WHY: KST combines 4 ROC timeframes with smoothing, capturing momentum across
        #      multiple cycles. Signal crossovers indicate momentum shifts confirmed by
        #      multiple timeframes.
        # BEST MARKETS: Trending markets across all asset classes. Stocks, forex, commodities.
        #               Excellent for confirming trend changes with multiple timeframe confirmation.
        # TIMEFRAME: Daily or weekly charts. Good for position trading and trend confirmation.
        signal_period = int(parameters.get('signal', 9))
        parameters_indicators["signal"] = signal_period
        short_window_indicator = 'KST'
        long_window_indicator = f'KST_Signal_{signal_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='kst',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='lsi':
        # LSI (Laguerre RSI) - Mean Reversion Strategy
        # ---------------------------------------------
        # LOGIC: Buy when LRSI drops below lower threshold (oversold), sell when above upper.
        # WHY: Laguerre filter creates an RSI with less lag and noise. Reacts faster to price
        #      changes than standard RSI. Gamma controls smoothing (higher = smoother).
        # BEST MARKETS: Short-term trading and scalping. Forex, futures, and liquid stocks.
        #               Popular for quick reversal detection due to low lag.
        # TIMEFRAME: Intraday to daily. Gamma 0.5 is common. Lower gamma = faster response.
        gamma = float(parameters.get('gamma', 0.5))
        upper = int(parameters.get('upper', 80))
        lower = int(parameters.get('lower', 20))
        parameters_indicators["gamma"] = gamma
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        gamma_str = f"{gamma:g}"
        indicator_col = f'LRSI_{gamma_str}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='lsi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'LRSI_{gamma_str}', 'lower', 'upper']

    
    elif strategy_name=='mac':
        # MAC (MACD) - Signal Line Crossover Strategy
        # --------------------------------------------
        # LOGIC: Buy when MACD line crosses above signal line, sell when crosses below.
        # WHY: MACD captures momentum shifts by comparing fast and slow EMAs. Signal line
        #      crossovers indicate changes in trend momentum. Classic trend-following indicator.
        # BEST MARKETS: Trending markets across all asset classes. Stocks, forex, crypto,
        #               commodities. One of the most versatile and widely-used indicators.
        # TIMEFRAME: All timeframes. Daily/weekly for position trading, 4H/1H for swing trading.
        window_fast = int(parameters.get('window_fast', 12))
        window_slow = int(parameters.get('window_slow', 26))
        window_signal = int(parameters.get('window_signal', 26))
        parameters_indicators["window_fast"] = window_fast
        parameters_indicators["window_slow"] = window_slow
        parameters_indicators["window_signal"] = window_signal
        short_window_indicator=f'MACD_{window_fast}_{window_slow}'
        long_window_indicator=f'Signal_{window_signal}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='mac',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='msi':
        # MSI (Momentum Strength Index) - Mean Reversion Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when MSI drops below lower threshold (oversold), sell when above upper.
        # WHY: MSI quantifies momentum strength with optional power scaling to emphasize
        #      strong moves. Similar to RSI but with configurable sensitivity to volatility.
        # BEST MARKETS: Range-bound markets. Power > 1 filters noise in volatile markets.
        #               Stocks, forex, and ETFs with regular mean-reverting behavior.
        # TIMEFRAME: Daily charts. 14-period is standard. Adjust power for volatility.
        window = int(parameters.get('window', 14))
        power = float(parameters.get('power', 1.0))
        upper = int(parameters.get('upper', 70))
        lower = int(parameters.get('lower', 30))
        parameters_indicators["window"] = window
        parameters_indicators["power"] = power
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'MSI_{window}_{power}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='msi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'MSI_{window}_{power}', 'lower', 'upper']

    
    elif strategy_name=='pgo':
        # PGO (Pretty Good Oscillator) - Mean Reversion Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when PGO drops below -3 (oversold), sell when rises above +3 (overbought).
        # WHY: PGO normalizes price deviation from SMA by ATR. Values beyond ±3 indicate
        #      price has moved significantly relative to recent volatility.
        # BEST MARKETS: Range-bound and mean-reverting markets. Stocks, forex, commodities.
        #               Good for identifying overextended moves in volatile assets.
        # TIMEFRAME: Daily charts. 14-period is common. Thresholds adjustable for volatility.
        window = int(parameters.get('window', 14))
        upper = float(parameters.get('upper', 3.0))
        lower = float(parameters.get('lower', -3.0))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'PGO_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='pgo',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'PGO_{window}', 'lower', 'upper']

    
    elif strategy_name=='ppo':
        # PPO (Percentage Price Oscillator) - Signal Line Crossover Strategy
        # -------------------------------------------------------------------
        # LOGIC: Buy when PPO line crosses above signal line, sell when crosses below.
        # WHY: PPO is MACD expressed as a percentage, allowing comparison across different
        #      priced securities. Signal crossovers indicate momentum shifts.
        # BEST MARKETS: Trending markets across all asset classes. Excellent for comparing
        #               momentum across different stocks regardless of price level.
        # TIMEFRAME: All timeframes. Standard settings: 12/26/9 (same as MACD).
        fast_window = int(parameters.get('fast_window', 12))
        slow_window = int(parameters.get('slow_window', 26))
        signal_window = int(parameters.get('signal_window', 9))
        parameters_indicators["fast_window"] = fast_window
        parameters_indicators["slow_window"] = slow_window
        parameters_indicators["signal_window"] = signal_window
        short_window_indicator = f'PPO_{fast_window}_{slow_window}'
        long_window_indicator = f'PPO_SIG_{signal_window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='ppo',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='psy':
        # PSY (Psychological Line) - Mean Reversion Strategy
        # ---------------------------------------------------
        # LOGIC: Buy when PSY drops below 25 (oversold), sell when rises above 75 (overbought).
        # WHY: PSY measures the ratio of up days to total days. Extreme values indicate
        #      overextended sentiment - too many up/down days suggests reversal is likely.
        # BEST MARKETS: Range-bound markets and mean-reverting assets. Stocks, indices,
        #               and ETFs. Good sentiment indicator for contrarian trading.
        # TIMEFRAME: Daily charts. 12-period is common. Good for swing trading.
        window = int(parameters.get('window', 12))
        upper = int(parameters.get('upper', 75))
        lower = int(parameters.get('lower', 25))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'PSY_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='psy',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'PSY_{window}', 'lower', 'upper']

    
    elif strategy_name=='qst':
        # QST (Qstick) - Zero Line Crossover Strategy
        # --------------------------------------------
        # LOGIC: Buy when Qstick crosses above zero (buying pressure), sell when crosses below.
        # WHY: Qstick averages candle bodies (Close-Open). Positive = buyers dominating,
        #      negative = sellers. Zero crossings signal shift in intraday sentiment.
        # BEST MARKETS: Markets with significant candle body variation. Stocks, forex.
        #               Good for confirming trend direction via candlestick analysis.
        # TIMEFRAME: Daily charts. 10-period is common. Works well for swing trading.
        window = int(parameters.get('window', 10))
        parameters_indicators["window"] = window
        short_window_indicator = f'QSTICK_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='qst',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='rmi':
        # RMI (Relative Momentum Index) - Mean Reversion Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when RMI drops below lower threshold (oversold), sell when above upper.
        # WHY: RMI is RSI calculated over a momentum period instead of 1-day changes.
        #      Smoother than RSI, better for identifying cyclical turns.
        # BEST MARKETS: Cyclical markets and assets with regular oscillations. Stocks,
        #               forex, commodities. Good for identifying overbought/oversold in trends.
        # TIMEFRAME: Daily charts. 20-period window with 5-period momentum is common.
        window = int(parameters.get('window', 20))
        momentum_period = int(parameters.get('momentum_period', 5))
        upper = int(parameters.get('upper', 70))
        lower = int(parameters.get('lower', 30))
        parameters_indicators["window"] = window
        parameters_indicators["momentum_period"] = momentum_period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'RMI_{window}_{momentum_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='rmi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'RMI_{window}_{momentum_period}', 'lower', 'upper']

    
    elif strategy_name=='roc':
        # ROC (Rate of Change) - Zero Line Crossover Strategy
        # ----------------------------------------------------
        # LOGIC: Buy when ROC crosses above zero (upward momentum), sell when crosses below.
        # WHY: ROC measures percentage price change over a period. Positive ROC = price rising,
        #      negative = falling. Zero crossings indicate momentum direction change.
        # BEST MARKETS: Trending markets. Stocks, indices, forex, commodities.
        #               Simple and effective for trend following and momentum confirmation.
        # TIMEFRAME: All timeframes. 12-period is common. Longer periods for position trading.
        window = int(parameters.get('window', 12))
        parameters_indicators["window"] = window
        short_window_indicator = f'ROC_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='roc',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='rsi':
        # RSI (Relative Strength Index) - Mean Reversion Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when RSI drops below lower threshold (oversold), sell when above upper.
        # WHY: RSI measures speed and magnitude of price changes. Extreme readings suggest
        #      exhaustion and potential reversal. Most popular momentum oscillator.
        # BEST MARKETS: Range-bound markets, stocks in consolidation, forex pairs.
        #               Use wider thresholds (80/20) in trending markets to avoid early exits.
        # TIMEFRAME: All timeframes. 14-period is standard. Shorter periods = more signals.
        window = int(parameters.get('window', 14))
        upper = int(parameters.get('upper', 80))
        lower = int(parameters.get('lower', 20))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'RSI_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='rsi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'RSI_{window}', 'lower', 'upper']

    
    elif strategy_name=='rvg':
        # RVG (Relative Vigor Index) - Signal Line Crossover Strategy
        # ------------------------------------------------------------
        # LOGIC: Buy when RVG crosses above its signal line, sell when crosses below.
        # WHY: RVG measures the conviction of price moves by comparing close-open to high-low.
        #      Prices tend to close higher than open in uptrends. Signal crossovers confirm momentum.
        # BEST MARKETS: Trending markets with clear directional moves. Stocks, forex, indices.
        #               Works well when combined with trend confirmation indicators.
        # TIMEFRAME: Daily charts. 10-period is standard. Good for swing trading.
        window = int(parameters.get('window', 10))
        parameters_indicators["window"] = window
        short_window_indicator = f'RVG_{window}'
        long_window_indicator = 'RVG_SIG'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='rvg',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
        
    
    elif strategy_name=='sri':
        # SRI (Stochastic RSI) - Mean Reversion Strategy
        # -----------------------------------------------
        # LOGIC: Buy when SRI %D drops below lower threshold (oversold), sell when above upper.
        # WHY: StochRSI applies Stochastic to RSI, creating a more sensitive oscillator.
        #      Reaches extremes more frequently than RSI, good for short-term reversals.
        # BEST MARKETS: Range-bound and volatile markets. Forex, stocks, crypto.
        #               More sensitive than RSI - use with trend filter to avoid false signals.
        # TIMEFRAME: Intraday to daily. Good for scalping and short-term swing trades.
        rsi_window = int(parameters.get('rsi_window', 14))
        stoch_window = int(parameters.get('stoch_window', 14))
        k_window = int(parameters.get('k_window', 3))
        d_window = int(parameters.get('d_window', 3))
        upper = int(parameters.get('upper', 80))
        lower = int(parameters.get('lower', 20))
        parameters_indicators["rsi_window"] = rsi_window
        parameters_indicators["stoch_window"] = stoch_window
        parameters_indicators["k_window"] = k_window
        parameters_indicators["d_window"] = d_window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'SRI_D_{d_window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='sri',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'SRI_K_{rsi_window}_{stoch_window}', f'SRI_D_{d_window}', 'lower', 'upper']

    
    elif strategy_name=='stc':
        # STC (Schaff Trend Cycle) - Mean Reversion Strategy
        # ---------------------------------------------------
        # LOGIC: Buy when STC drops below 25 (oversold), sell when rises above 75 (overbought).
        # WHY: STC combines MACD with Stochastic for faster, more accurate trend detection.
        #      Less lag than MACD, identifies trends earlier with clear overbought/oversold zones.
        # BEST MARKETS: Trending markets across all asset classes. Stocks, forex, crypto.
        #               Excellent for early trend detection and cycle identification.
        # TIMEFRAME: Daily or 4-hour charts. Good for swing trading and position trading.
        window_fast = int(parameters.get('window_fast', 23))
        window_slow = int(parameters.get('window_slow', 50))
        cycle = int(parameters.get('cycle', 10))
        smooth = int(parameters.get('smooth', 3))
        upper = int(parameters.get('upper', 75))
        lower = int(parameters.get('lower', 25))
        parameters_indicators["window_fast"] = window_fast
        parameters_indicators["window_slow"] = window_slow
        parameters_indicators["cycle"] = cycle
        parameters_indicators["smooth"] = smooth
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'STC_{window_fast}_{window_slow}_{cycle}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='stc',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'STC_{window_fast}_{window_slow}_{cycle}', 'lower', 'upper']


    elif strategy_name=='sto':
        # STO (Stochastic Oscillator) - Mean Reversion Strategy
        # ------------------------------------------------------
        # LOGIC: Buy when %D drops below lower threshold (oversold), sell when above upper.
        # WHY: Stochastic compares closing price to price range over a period. Shows where
        #      price closed relative to its recent range. Good for timing entries/exits.
        # BEST MARKETS: Range-bound markets, forex pairs, and stocks in consolidation.
        #               Combine with trend filter for better results in trending markets.
        # TIMEFRAME: All timeframes. Fast stochastic for short-term, slow for swing trading.
        k_period = int(parameters.get('k_period', 14))
        d_period = int(parameters.get('d_period', 14))
        smooth_k = int(parameters.get('smooth_k', 14))
        upper = int(parameters.get('upper', 80))
        lower = int(parameters.get('lower', 20))
        parameters_indicators["k_period"] = k_period
        parameters_indicators["d_period"] = d_period
        parameters_indicators["smooth_k"] = smooth_k
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'STOCH_D_{k_period}_{d_period}_{smooth_k}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='sto',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'STOCH_D_{k_period}_{d_period}_{smooth_k}', 'lower', 'upper']

    
    elif strategy_name=='tsi':
        # TSI (True Strength Index) - Zero Line Crossover Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when TSI crosses above zero (bullish momentum), sell when crosses below.
        # WHY: TSI uses double smoothing to filter noise and highlight true trend strength.
        #      Zero crossings indicate momentum direction change with reduced false signals.
        # BEST MARKETS: Trending markets. Stocks, forex, indices. Smoother than single-smoothed
        #               momentum indicators. Good for confirming trend direction.
        # TIMEFRAME: Daily charts. 25/13 periods are standard. Good for position trading.
        slow = int(parameters.get('slow', 25))
        fast = int(parameters.get('fast', 13))
        parameters_indicators["slow"] = slow
        parameters_indicators["fast"] = fast
        short_window_indicator = f'TSI_{slow}_{fast}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='tsi',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='ttm':
        # TTM (TTM Squeeze) - Momentum Zero Line Crossover Strategy
        # ----------------------------------------------------------
        # LOGIC: Buy when TTM momentum crosses above zero (bullish), sell when crosses below.
        # WHY: TTM Squeeze identifies consolidation (squeeze) followed by breakouts.
        #      Momentum histogram direction after squeeze release indicates trade direction.
        # BEST MARKETS: All markets. Excellent for breakout trading. Stocks, forex, futures.
        #               Combines volatility squeeze with momentum for high-probability setups.
        # TIMEFRAME: Daily or 4-hour charts. 20-period is standard. Good for swing trading.
        length = int(parameters.get('length', 20))
        std_dev = float(parameters.get('std_dev', 2.0))
        atr_length = int(parameters.get('atr_length', 20))
        atr_multiplier = float(parameters.get('atr_multiplier', 1.5))
        smooth = int(parameters.get('smooth', 3))
        parameters_indicators["length"] = length
        parameters_indicators["std_dev"] = std_dev
        parameters_indicators["atr_length"] = atr_length
        parameters_indicators["atr_multiplier"] = atr_multiplier
        parameters_indicators["smooth"] = smooth
        short_window_indicator = f'TTM_MOM_{length}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='ttm',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover strategy
        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='ult':
        # ULT (Ultimate Oscillator) - Mean Reversion Strategy
        # ----------------------------------------------------
        # LOGIC: Buy when ULT drops below 30 (oversold), sell when rises above 70 (overbought).
        # WHY: Ultimate Oscillator combines 3 timeframes (7/14/28) to reduce volatility and
        #      false signals. Weighted average gives more weight to short-term momentum.
        # BEST MARKETS: Range-bound markets. Stocks, forex, indices. Reduces false signals
        #               compared to single-timeframe oscillators. Good for divergence trading.
        # TIMEFRAME: Daily charts. Standard periods: 7/14/28. Good for swing trading.
        short_window = int(parameters.get('short_window', 7))
        medium_window = int(parameters.get('medium_window', 14))
        long_window = int(parameters.get('long_window', 28))
        upper = int(parameters.get('upper', 70))
        lower = int(parameters.get('lower', 30))
        parameters_indicators["short_window"] = short_window
        parameters_indicators["medium_window"] = medium_window
        parameters_indicators["long_window"] = long_window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'ULTOSC_{short_window}_{medium_window}_{long_window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='ult',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'ULTOSC_{short_window}_{medium_window}_{long_window}', 'lower', 'upper']

    
    elif strategy_name=='vor':
        # VOR (Vortex Indicator) - VI+/VI- Crossover Strategy
        # ----------------------------------------------------
        # LOGIC: Buy when VI+ crosses above VI- (bulls in control), sell when VI- crosses above VI+.
        # WHY: Vortex captures positive and negative trend movements using high-low relationships.
        #      Crossovers indicate shifts between bullish and bearish control.
        # BEST MARKETS: Trending markets. Stocks, forex, commodities. Good for identifying
        #               trend reversals and confirming trend direction.
        # TIMEFRAME: Daily charts. 14-period is common. Good for swing and position trading.
        window = int(parameters.get('window', 14))
        parameters_indicators["window"] = window
        short_window_indicator = f'VI_Plus_{window}'
        long_window_indicator = f'VI_Minus_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vor',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='wil':
        # WIL (Williams %R) - Mean Reversion Strategy
        # --------------------------------------------
        # LOGIC: Buy when %R drops below -80 (oversold), sell when rises above -20 (overbought).
        # WHY: Williams %R measures where close is relative to high-low range. Near 0 = overbought
        #      (close near highs), near -100 = oversold (close near lows).
        # BEST MARKETS: Range-bound markets. Stocks, forex, indices. Fast and responsive.
        #               Good for short-term reversals in consolidating markets.
        # TIMEFRAME: All timeframes. 14-period is standard. Popular for day and swing trading.
        window = int(parameters.get('window', 14))
        upper = int(parameters.get('upper', -20))
        lower = int(parameters.get('lower', -80))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'WILLR_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='wil',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'WILLR_{window}', 'lower', 'upper']
    

    # ==================== TREND STRATEGIES ====================
    elif strategy_name == 'ads':
        # ADS (Adaptive Moving Average - Smoothed) - Dual MA Crossover Strategy
        # ----------------------------------------------------------------------
        # LOGIC: Buy when fast ADSMA crosses above slow ADSMA, sell when crosses below.
        # WHY: Adaptive smoothing reduces lag while filtering noise. Crossovers signal
        #      trend changes with better timing than traditional MAs.
        # BEST MARKETS: Trending stocks, forex, and indices. Reduces whipsaws in
        #               moderately volatile markets compared to SMA/EMA.
        # TIMEFRAME: Daily or weekly charts for position trading.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'ADSMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='ads',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'ADSMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='ads',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'adx':
        # ADX (Average Directional Index) - Trend Strength Filter Strategy
        # -----------------------------------------------------------------
        # LOGIC: Only trade when ADX > threshold (strong trend). Use price vs SMA crossover
        #        for entry signals, but filter out trades when trend is weak.
        # WHY: ADX measures trend strength (not direction). High ADX = strong trend worth
        #      following. Avoids choppy markets where crossover strategies fail.
        # BEST MARKETS: Any market, but especially useful for filtering false signals.
        #               Stocks, forex, commodities during trending phases.
        # TIMEFRAME: Daily or weekly. ADX > 25 typically indicates tradeable trend.
        window = int(parameters.get('window', 14))
        adx_threshold = float(parameters.get('adx_threshold', 25))
        ma_window = int(parameters.get('ma_window', 20))  # Moving average for price crossover
        parameters_indicators["window"] = window
        adx_col = f'ADX_{window}'
        price_col = 'Close'

        # Compute ADX indicator
        data, columns, fig = compute_indicator(
        data=data,
        indicator='adx',
        parameters=parameters_indicators,
        figure=False)

        # Compute SMA for price crossover signals
        parameters["window"] = ma_window
        data, columns_sma, fig = compute_indicator(
        data=data,
        indicator='sma',
        parameters=parameters,
        figure=False)
        
        sma_col = f'SMA_{ma_window}'
        
        # ADX Trend-Following Strategy:
        # 1. Use Price vs SMA crossover for entry/exit signals
        # 2. Only trade when ADX > threshold (strong trend exists)
        # 3. +DI/-DI shown in plot to visualize trend direction
        data_filtered = data.copy()
        
        # Create a synthetic indicator that combines price action with ADX filter
        # When ADX is strong: allow price vs SMA crossovers
        # When ADX is weak: neutralize signals by making price = SMA
        
        strong_trend = data_filtered[adx_col] > adx_threshold
        
        # Create conditional price column that only allows trades in strong trends
        data_filtered['Price_ADX_Filtered'] = data_filtered[price_col].copy()
        
        # When trend is weak, set price = SMA to prevent crossovers
        weak_trend_mask = ~strong_trend
        data_filtered.loc[weak_trend_mask, 'Price_ADX_Filtered'] = data_filtered.loc[weak_trend_mask, sma_col]
        
        # Use Price vs SMA crossover with ADX filter
        short_window_indicator = 'Price_ADX_Filtered'
        long_window_indicator = sma_col

        results, portfolio = cross_backtester.run_cross_trade(
        data=data_filtered,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [adx_col]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"ADX Trend Strategy (ADX>{adx_threshold})",
            show_extra_panel=True,
            extra_panel_cols=[price_col, sma_col],
            extra_panel_title=f"Price vs SMA-{ma_window}")

    elif strategy_name == 'alm':
        # ALM (ALMA - Arnaud Legoux Moving Average) - Dual MA Crossover Strategy
        # -----------------------------------------------------------------------
        # LOGIC: Buy when fast ALMA crosses above slow ALMA, sell when crosses below.
        # WHY: ALMA uses Gaussian distribution for weighting, reducing lag while maintaining
        #      smoothness. Offset parameter controls responsiveness vs smoothness tradeoff.
        # BEST MARKETS: Trending markets. Particularly good for stocks and forex where
        #               reduced lag is valuable. Less effective in ranging markets.
        # TIMEFRAME: Daily charts. Can be used on lower TFs with adjusted parameters.
        short_window = int(parameters.get('short_window', 9))
        long_window = int(parameters.get('long_window', 27))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'ALMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='alm',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'ALMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='alm',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'ama':
        # AMA (Kaufman Adaptive Moving Average) - Dual MA Crossover Strategy
        # -------------------------------------------------------------------
        # LOGIC: Buy when fast AMA crosses above slow AMA, sell when crosses below.
        # WHY: AMA adapts its speed based on market efficiency ratio. Fast in trends,
        #      slow in choppy markets. Automatically adjusts to market conditions.
        # BEST MARKETS: Works across all market conditions due to adaptive nature.
        #               Stocks, forex, futures. Reduces whipsaws in ranging markets.
        # TIMEFRAME: Daily or 4-hour charts. One of the best adaptive MAs available.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        fast_period = int(parameters.get('fast_period', 2))
        slow_period = int(parameters.get('slow_period', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'AMA_{short_window}_{fast_period}_{slow_period}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='ama',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'AMA_{long_window}_{fast_period}_{slow_period}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='ama',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'aro':
        # ARO (Aroon) - Up/Down Crossover Strategy
        # -----------------------------------------
        # LOGIC: Buy when Aroon Up crosses above Aroon Down, sell when crosses below.
        # WHY: Aroon measures time since highest high and lowest low. Crossovers indicate
        #      which extreme is more recent, signaling trend direction changes.
        # BEST MARKETS: Trending markets with clear highs and lows. Stocks, commodities,
        #               and forex. Good for identifying new trends early.
        # TIMEFRAME: Daily or weekly. 25-period is common. Longer periods = fewer signals.
        period = int(parameters.get('period', 14))
        parameters_indicators["period"] = period
        short_window_indicator=f'AROON_UP_{period}'
        long_window_indicator=f'AROON_DOWN_{period}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='aro',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'dem':
        # DEM (DEMA - Double Exponential Moving Average) - Dual MA Crossover Strategy
        # ----------------------------------------------------------------------------
        # LOGIC: Buy when fast DEMA crosses above slow DEMA, sell when crosses below.
        # WHY: DEMA reduces lag by applying EMA twice and adjusting. Faster response to
        #      price changes while maintaining smoothness. Good for trend following.
        # BEST MARKETS: Trending stocks, forex, and indices. Faster signals than EMA
        #               but more prone to whipsaws in choppy markets.
        # TIMEFRAME: Daily or 4-hour charts. Popular for swing trading.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'DEMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='dem',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'DEMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='dem',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'eac':
        # EAC (Exponential Adaptive Close) - Alpha-Based Crossover Strategy
        # ------------------------------------------------------------------
        # LOGIC: Buy when fast EAC crosses above slow EAC, sell when crosses below.
        #        Note: Lower alpha = more smoothing (slower), so parameters are swapped.
        # WHY: Alpha-based smoothing allows fine-tuned control over responsiveness.
        #      Different from window-based MAs, directly controls decay rate.
        # BEST MARKETS: Trending markets. Good for traders who want precise control
        #               over indicator sensitivity. Forex and stocks.
        # TIMEFRAME: Daily or weekly charts. Adjust alpha based on volatility.
        short_alpha = float(parameters.get('short_alpha', 0.07))
        long_alpha = float(parameters.get('long_alpha', 0.14))
        price_col='Close'

        # NOTE: For EAC, lower alpha = MORE smoothing (slower)
        # So we need to SWAP: short_alpha creates the LONG (slow) indicator
        # and long_alpha creates the SHORT (fast) indicator
        
        # If short_alpha is 0, use actual price instead of indicator
        if short_alpha == 0:
            slow_indicator = 'Close'
        else:
            slow_indicator = f'EAC_{int(short_alpha*100)}'
            parameters["alpha"] = short_alpha
            data, columns, fig = compute_indicator(
            data=data,
            indicator='eac',
            parameters=parameters,
            figure=False)

        # If long_alpha is 0, use actual price instead of indicator
        if long_alpha == 0:
            fast_indicator = 'Close'
        else:
            fast_indicator = f'EAC_{int(long_alpha*100)}'
            parameters["alpha"] = long_alpha
            data, columns, fig = compute_indicator(
            data=data,
            indicator='eac',
            parameters=parameters,
            figure=False)

        # Swap: fast indicator goes to short_window, slow indicator goes to long_window
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=fast_indicator,
        long_window_indicator=slow_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [fast_indicator, slow_indicator]

    
    elif strategy_name == 'eit':
        # EIT (Ehlers Instantaneous Trendline) - Alpha-Based Crossover Strategy
        # ----------------------------------------------------------------------
        # LOGIC: Buy when fast EIT crosses above slow EIT, sell when crosses below.
        #        Note: Lower alpha = more smoothing (slower), so parameters are swapped.
        # WHY: Ehlers' filter design minimizes lag while reducing noise. Based on
        #      digital signal processing principles for cleaner trend identification.
        # BEST MARKETS: Cyclical markets and assets with regular patterns. Forex,
        #               indices, and commodities. Good for identifying trend changes.
        # TIMEFRAME: Daily charts. Works well on 4-hour for active trading.
        short_alpha = float(parameters.get('short_alpha', 0.07))
        long_alpha = float(parameters.get('long_alpha', 0.14))
        price_col='Close'

        # NOTE: For EIT, lower alpha = MORE smoothing (slower)
        # So we need to SWAP: short_alpha creates the LONG (slow) indicator
        # and long_alpha creates the SHORT (fast) indicator
        
        # If short_alpha is 0, use actual price instead of indicator
        if short_alpha == 0:
            slow_indicator = 'Close'
        else:
            slow_indicator = f'EIT_{short_alpha}'
            parameters["alpha"] = short_alpha
            data, columns, fig = compute_indicator(
            data=data,
            indicator='eit',
            parameters=parameters,
            figure=False)

        # If long_alpha is 0, use actual price instead of indicator
        if long_alpha == 0:
            fast_indicator = 'Close'
        else:
            fast_indicator = f'EIT_{long_alpha}'
            parameters["alpha"] = long_alpha
            data, columns, fig = compute_indicator(
            data=data,
            indicator='eit',
            parameters=parameters,
            figure=False)

        # Swap: fast indicator goes to short_window, slow indicator goes to long_window
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=fast_indicator,
        long_window_indicator=slow_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [fast_indicator, slow_indicator]

    
    elif strategy_name == 'ema':
        # EMA (Exponential Moving Average) - Dual MA Crossover Strategy
        # --------------------------------------------------------------
        # LOGIC: Buy when fast EMA crosses above slow EMA, sell when crosses below.
        # WHY: EMA gives more weight to recent prices, responding faster than SMA.
        #      Classic trend-following approach used by traders worldwide.
        # BEST MARKETS: Trending markets across all asset classes. The most widely
        #               used MA type. Stocks, forex, crypto, commodities.
        # TIMEFRAME: All timeframes. 12/26 for short-term, 50/200 for long-term trends.
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'EMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='ema',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'EMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='ema',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'fma':
        # FMA (Fibonacci Moving Average) - Dual MA Crossover Strategy
        # -------------------------------------------------------------
        # LOGIC: Buy when fast FMA crosses above slow FMA, sell when crosses below.
        #        Optional ADX filter to only trade in strong trends.
        # WHY: Uses Fibonacci-weighted smoothing for natural market rhythm alignment.
        #      ADX filter prevents trading in weak/choppy market conditions.
        # BEST MARKETS: Trending markets. The Fibonacci weighting may resonate with
        #               markets that exhibit natural retracement patterns.
        # TIMEFRAME: Daily or weekly. ADX filter recommended for noisy markets.
        short_window = int(parameters.get('short_window', 8))
        long_window = int(parameters.get('long_window', 24))
        use_adx_filter = bool(parameters.get('use_adx_filter', False))
        adx_window = int(parameters.get('adx_window', 14))
        adx_threshold = float(parameters.get('adx_threshold', 25))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'FMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='fma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'FMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='fma',
        parameters=parameters,
        figure=False)

        # Optional ADX filter
        if use_adx_filter:
            parameters_indicators["window"] = adx_window
            adx_col = f'ADX_{adx_window}'
            data, columns_adx, fig = compute_indicator(
            data=data,
            indicator='adx',
            parameters=parameters_indicators,
            figure=False)
            
            # Apply ADX filter: neutralize signals when ADX < threshold
            data_filtered = data.copy()
            strong_trend = data_filtered[adx_col] > adx_threshold
            
            # Create filtered version of short indicator
            if short_window == 0:
                # If using Close price, create filtered version
                data_filtered['FMA_Short_Filtered'] = data_filtered[short_window_indicator].copy()
                weak_trend_mask = ~strong_trend
                data_filtered.loc[weak_trend_mask, 'FMA_Short_Filtered'] = data_filtered.loc[weak_trend_mask, long_window_indicator]
                short_window_indicator_filtered = 'FMA_Short_Filtered'
            else:
                # If using FMA indicator, create filtered version
                data_filtered[f'{short_window_indicator}_Filtered'] = data_filtered[short_window_indicator].copy()
                weak_trend_mask = ~strong_trend
                data_filtered.loc[weak_trend_mask, f'{short_window_indicator}_Filtered'] = data_filtered.loc[weak_trend_mask, long_window_indicator]
                short_window_indicator_filtered = f'{short_window_indicator}_Filtered'
            
            results, portfolio = cross_backtester.run_cross_trade(
            data=data_filtered,
            short_window_indicator=short_window_indicator_filtered,
            long_window_indicator=long_window_indicator,
            price_col=price_col,
            long_entry_pct_cash=long_entry_pct_cash,
            short_entry_pct_cash=short_entry_pct_cash,
            trading_type=trading_type,
            day1_position=day1_position,
            risk_free_rate=risk_free_rate)
            
            indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
            
            # Custom plotting with ADX in fourth panel
            if fig_control==1:
                short_label = 'Close' if short_window == 0 else f'FMA-{short_window}'
                
                # Add threshold line to data for visualization
                data_filtered[f'ADX_Threshold_{adx_threshold}'] = adx_threshold
                
                fig = plotter.plot_results(
                data_df=data_filtered,
                history_df=portfolio,
                price_col=price_col,
                indicator_cols=indicator_cols_to_plot,
                title=f"FMA Strategy ({short_label} vs FMA-{long_window}) with ADX Filter (>{adx_threshold})",
                show_extra_panel=True,
                extra_panel_cols=[adx_col, f'ADX_Threshold_{adx_threshold}'],
                extra_panel_title=f"ADX Trend Strength (Threshold: {adx_threshold})")
        else:
            # No ADX filter
            results, portfolio = cross_backtester.run_cross_trade(
            data=data,
            short_window_indicator=short_window_indicator,
            long_window_indicator=long_window_indicator,
            price_col=price_col,
            long_entry_pct_cash=long_entry_pct_cash,
            short_entry_pct_cash=short_entry_pct_cash,
            trading_type=trading_type,
            day1_position=day1_position,
            risk_free_rate=risk_free_rate)
            
            indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'gma':
        # GMA (Guppy Multiple Moving Average) - Multi-EMA Crossover Strategy
        # -------------------------------------------------------------------
        # LOGIC: Buy when fastest short EMA crosses above slowest long EMA, sell when below.
        # WHY: Uses multiple EMAs to visualize trend strength and trader sentiment.
        #      Short EMAs = short-term traders, Long EMAs = long-term investors.
        # BEST MARKETS: Trending markets where you want to see trend conviction.
        #               Stocks, forex, indices. Compression of EMAs signals consolidation.
        # TIMEFRAME: Daily or weekly. Visual indicator - good for discretionary trading.
        # Guppy Multiple Moving Average - uses multiple EMAs
        short_windows = parameters.get('short_windows', (3, 5, 8, 10, 12, 15))
        long_windows = parameters.get('long_windows', (30, 35, 40, 45, 50, 60))
        price_col='Close'
        
        # Compute GMA indicator (returns multiple EMA columns)
        parameters_indicators["short_windows"] = short_windows
        parameters_indicators["long_windows"] = long_windows
        data, columns, fig = compute_indicator(
        data=data,
        indicator='gma',
        parameters=parameters_indicators,
        figure=False)
        
        # Use the fastest short EMA and slowest long EMA for crossover signals
        short_window_indicator = f'GMA_short_{min(short_windows)}'
        long_window_indicator = f'GMA_long_{max(long_windows)}'
        
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)
        
        # Plot all GMA lines
        indicator_cols_to_plot = [col for col in columns if col.startswith('GMA_')]

    
    elif strategy_name == 'hma':
        # HMA (Hull Moving Average) - Dual MA Crossover Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when fast HMA crosses above slow HMA, sell when crosses below.
        # WHY: HMA uses weighted MAs and square root of period to minimize lag while
        #      maintaining smoothness. One of the fastest-responding MAs available.
        # BEST MARKETS: Fast-moving markets where lag is costly. Day trading stocks,
        #               forex, and futures. May generate more signals than other MAs.
        # TIMEFRAME: All timeframes. Particularly popular for intraday trading.
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'HMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='hma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'HMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='hma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'htt':
        # HTT (Hilbert Transform Trendline) - Dual Crossover Strategy
        # ------------------------------------------------------------
        # LOGIC: Buy when fast HTT crosses above slow HTT, sell when crosses below.
        # WHY: Based on Ehlers' Hilbert Transform, designed to extract trend from
        #      cyclical components. Provides smooth trendline with minimal lag.
        # BEST MARKETS: Cyclical markets with regular oscillations. Forex pairs,
        #               commodities, and indices. Good for identifying dominant cycles.
        # TIMEFRAME: Daily charts. Based on signal processing theory.
        short_window = int(parameters.get('short_window', 8))
        long_window = int(parameters.get('long_window', 16))
        price_col='Close'
        
        # Compute short HTT indicator
        parameters_indicators["window"] = short_window
        short_window_indicator = f'HTT_{short_window}'
        data, columns, fig = compute_indicator(
        data=data,
        indicator='htt',
        parameters=parameters_indicators,
        figure=False)
        
        # Compute long HTT indicator
        parameters_indicators["window"] = long_window
        long_window_indicator = f'HTT_{long_window}'
        data, columns, fig = compute_indicator(
        data=data,
        indicator='htt',
        parameters=parameters_indicators,
        figure=False)
        
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)
        
        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='ich':
        # ICH (Ichimoku Cloud) - Tenkan/Kijun Crossover Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when Tenkan-sen crosses above Kijun-sen, sell when crosses below.
        # WHY: Ichimoku is a complete trading system. Tenkan = fast line (9-period),
        #      Kijun = slow line (26-period). Crossovers signal trend changes.
        # BEST MARKETS: Trending markets. Originally designed for Japanese stocks.
        #               Works well on forex, indices, and liquid stocks.
        # TIMEFRAME: Daily or weekly. Traditional settings (9/26/52) work best.
        tenkan_period = int(parameters.get('tenkan_period', 9))
        kijun_period = int(parameters.get('kijun_period', 26))
        senkou_b_period = int(parameters.get('senkou_b_period', 52))
        displacement = int(parameters.get('displacement', 26))
        parameters_indicators["tenkan_period"] = tenkan_period
        parameters_indicators["kijun_period"] = kijun_period
        parameters_indicators["senkou_b_period"] = senkou_b_period
        parameters_indicators["displacement"] = displacement
        short_window_indicator=f'tenkan_sen_{tenkan_period}'
        long_window_indicator=f'kijun_sen_{kijun_period}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='ich',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'jma':
        # JMA (Jurik Moving Average) - Dual MA Crossover Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when fast JMA crosses above slow JMA, sell when crosses below.
        # WHY: JMA is designed to be smooth with minimal lag. Uses adaptive volatility
        #      filtering to reduce noise while maintaining responsiveness.
        # BEST MARKETS: All markets. Particularly good for volatile assets where
        #               noise reduction is important. Stocks, forex, crypto.
        # TIMEFRAME: All timeframes. Premium indicator known for quality signals.
        short_length = int(parameters.get('short_length', 14))
        long_length = int(parameters.get('long_length', 42))
        price_col='Close'

        # If short_length is 0, use actual price instead of indicator
        if short_length == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'JMA_{short_length}'
            parameters["length"] = short_length
            data, columns, fig = compute_indicator(
            data=data,
            indicator='jma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'JMA_{long_length}'
        parameters["length"] = long_length
        data, columns, fig = compute_indicator(
        data=data,
        indicator='jma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'kma':
        # KMA (Kaufman Moving Average) - Dual MA Crossover Strategy
        # ----------------------------------------------------------
        # LOGIC: Buy when fast KMA crosses above slow KMA, sell when crosses below.
        # WHY: Similar to AMA, adapts to market efficiency. Faster in trends,
        #      slower in choppy markets. Reduces whipsaws automatically.
        # BEST MARKETS: Works across all market conditions. Stocks, forex, futures.
        #               Good for traders who want adaptive behavior without tuning.
        # TIMEFRAME: Daily or 4-hour charts. Robust across different conditions.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        fast_period = int(parameters.get('fast_period', 2))
        slow_period = int(parameters.get('slow_period', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'KMA_{short_window}_{fast_period}_{slow_period}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='kma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'KMA_{long_window}_{fast_period}_{slow_period}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='kma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'lsm':
        # LSM (Least Squares Moving Average) - Dual MA Crossover Strategy
        # ----------------------------------------------------------------
        # LOGIC: Buy when fast LSMA crosses above slow LSMA, sell when crosses below.
        # WHY: LSMA uses linear regression to project trend. Provides endpoint of
        #      regression line, which can lead price action in trending markets.
        # BEST MARKETS: Trending markets with clear directional moves. Stocks,
        #               forex, commodities. May lead price at turning points.
        # TIMEFRAME: Daily or 4-hour. Good for identifying trend direction.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'LSMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='lsm',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'LSMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='lsm',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'mgd':
        # MGD (McGinley Dynamic) - Dual MA Crossover Strategy
        # ----------------------------------------------------
        # LOGIC: Buy when fast MGD crosses above slow MGD, sell when crosses below.
        # WHY: MGD automatically adjusts speed based on market conditions. Designed
        #      to track price more closely than EMA while avoiding whipsaws.
        # BEST MARKETS: All markets. Particularly good for volatile conditions.
        #               Stocks, forex, indices. Self-adjusting nature is valuable.
        # TIMEFRAME: All timeframes. Robust indicator that adapts automatically.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'MGD_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='mgd',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'MGD_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='mgd',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='psa':
        # PSA (Parabolic SAR) - Price vs SAR Crossover Strategy
        # ------------------------------------------------------
        # LOGIC: Buy when price crosses above PSAR (bullish), sell when below (bearish).
        # WHY: PSAR provides trailing stop levels that accelerate with trend. When price
        #      crosses SAR, it signals trend reversal. Good for trend following with stops.
        # BEST MARKETS: Strongly trending markets. Stocks, forex, commodities in clear
        #               trends. Generates many signals in ranging markets (use filter).
        # TIMEFRAME: Daily or 4-hour. Adjust AF parameters for different volatilities.
        af_initial = float(parameters.get('af_initial', 0.03))
        af_step = float(parameters.get('af_step', 0.03))
        af_max = float(parameters.get('af_max', 0.3))
        parameters_indicators["af_initial"] = af_initial
        parameters_indicators["af_step"] = af_step
        parameters_indicators["af_max"] = af_max
        short_window_indicator="Close"
        long_window_indicator=f"PSAR_Bullish_{af_initial}_{af_step}_{af_max}"
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='psa',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'sma':
        # SMA (Simple Moving Average) - Dual MA Crossover Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when fast SMA crosses above slow SMA, sell when crosses below.
        # WHY: SMA is the most basic MA, giving equal weight to all prices in window.
        #      Classic trend-following approach. Golden Cross (50/200) is famous example.
        # BEST MARKETS: Trending markets across all asset classes. The foundational
        #               MA type. Stocks, forex, indices, commodities.
        # TIMEFRAME: All timeframes. 50/200 for long-term, 10/30 for short-term.
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'SMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='sma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'SMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='sma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'soa':
        # SOA (Second Order Adaptive) - Dual MA Crossover Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when fast SOA crosses above slow SOA, sell when crosses below.
        # WHY: Second-order adaptive filtering provides smoother output while
        #      maintaining responsiveness. Reduces noise in volatile markets.
        # BEST MARKETS: Volatile markets where noise reduction is important.
        #               Stocks, forex, crypto. Good for swing trading.
        # TIMEFRAME: Daily or 4-hour charts. Adjust windows based on volatility.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'SOA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='soa',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'SOA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='soa',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'str':
        # STR (SuperTrend) - Price vs SuperTrend Crossover Strategy
        # ----------------------------------------------------------
        # LOGIC: Buy when price crosses above SuperTrend, sell when crosses below.
        # WHY: SuperTrend combines ATR with price action to create dynamic support/
        #      resistance levels. Flips direction on breakouts, providing clear signals.
        # BEST MARKETS: Trending markets. Very popular in Indian markets. Works well
        #               on stocks, forex, and commodities with clear trends.
        # TIMEFRAME: Daily or 4-hour. Multiplier controls sensitivity to volatility.
        period = int(parameters.get('period', 7))
        multiplier = float(parameters.get('multiplier', 3.0))
        parameters_indicators["period"] = period
        parameters_indicators["multiplier"] = multiplier
        short_window_indicator="Close"
        long_window_indicator=f'Supertrend_Bullish_{period}_{multiplier}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='str',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'swm':
        # SWM (Sine-Weighted Moving Average) - Dual MA Crossover Strategy
        # ----------------------------------------------------------------
        # LOGIC: Buy when fast SWMA crosses above slow SWMA, sell when crosses below.
        # WHY: Uses sine wave weighting to emphasize middle of window. Provides
        #      smooth output that may align with natural market cycles.
        # BEST MARKETS: Cyclical markets with regular patterns. Forex pairs,
        #               commodities, and indices. Good for swing trading.
        # TIMEFRAME: Daily charts. Interesting alternative to traditional MAs.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'SWMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='swm',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'SWMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='swm',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'tem':
        # TEM (TEMA - Triple Exponential Moving Average) - Dual MA Crossover Strategy
        # ----------------------------------------------------------------------------
        # LOGIC: Buy when fast TEMA crosses above slow TEMA, sell when crosses below.
        # WHY: TEMA applies EMA three times with adjustments to minimize lag further
        #      than DEMA. Very responsive to price changes while staying smooth.
        # BEST MARKETS: Fast-moving trending markets. Stocks, forex, futures.
        #               May generate more signals than slower MAs.
        # TIMEFRAME: All timeframes. Popular for active trading strategies.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'TEMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='tem',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'TEMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='tem',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'tma':
        # TMA (Triangular Moving Average) - Dual MA Crossover Strategy
        # -------------------------------------------------------------
        # LOGIC: Buy when fast TMA crosses above slow TMA, sell when crosses below.
        # WHY: TMA is double-smoothed SMA, giving more weight to middle of window.
        #      Very smooth output, good for identifying underlying trend direction.
        # BEST MARKETS: Noisy markets where smoothing is valuable. Stocks, forex.
        #               Slower to react but fewer false signals.
        # TIMEFRAME: Daily or weekly. Good for position trading.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'TMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='tma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'TMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='tma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name=='tri':
        # TRI (TRIX) - TRIX vs Signal Line Crossover Strategy
        # ----------------------------------------------------
        # LOGIC: Buy when TRIX crosses above signal line, sell when crosses below.
        # WHY: TRIX is triple-smoothed EMA rate of change. Filters out short-term
        #      fluctuations, showing only significant trend changes.
        # BEST MARKETS: Trending markets where you want to filter noise. Stocks,
        #               indices, forex. Good for identifying major trend changes.
        # TIMEFRAME: Daily or weekly. Longer windows = fewer but stronger signals.
        window = int(parameters.get('window', 7))
        parameters_indicators["window"] = window
        short_window_indicator=f'TRIX_{window}'
        long_window_indicator=f'TRIX_SIGNAL_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='tri',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'vid':
        # VID (Variable Index Dynamic Average) - Dual MA Crossover Strategy
        # ------------------------------------------------------------------
        # LOGIC: Buy when fast VIDYA crosses above slow VIDYA, sell when crosses below.
        # WHY: VIDYA adapts speed based on CMO (volatility measure). Fast in trends,
        #      slow in consolidation. Automatically adjusts to market conditions.
        # BEST MARKETS: All market conditions due to adaptive nature. Stocks, forex,
        #               futures. Reduces whipsaws in ranging markets.
        # TIMEFRAME: Daily or 4-hour. One of the best adaptive indicators.
        short_window = int(parameters.get('short_window', 14))
        long_window = int(parameters.get('long_window', 42))
        cmo_window = int(parameters.get('cmo_window', 9))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'VID_{short_window}_{cmo_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='vid',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'VID_{long_window}_{cmo_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='vid',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'wma':
        # WMA (Weighted Moving Average) - Dual MA Crossover Strategy
        # -----------------------------------------------------------
        # LOGIC: Buy when fast WMA crosses above slow WMA, sell when crosses below.
        # WHY: WMA gives linearly increasing weight to recent prices. More responsive
        #      than SMA but smoother than EMA. Good balance of lag and smoothness.
        # BEST MARKETS: Trending markets across all asset classes. Stocks, forex,
        #               commodities. Classic indicator with proven track record.
        # TIMEFRAME: All timeframes. 10/30 for short-term, 50/200 for long-term.
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'WMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='wma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'WMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='wma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    
    elif strategy_name == 'zma':
        # ZMA (Zero-Lag Moving Average) - Dual MA Crossover Strategy
        # -----------------------------------------------------------
        # LOGIC: Buy when fast ZMA crosses above slow ZMA, sell when crosses below.
        # WHY: ZMA attempts to eliminate lag by adding momentum component. Responds
        #      faster to price changes than traditional MAs.
        # BEST MARKETS: Fast-moving markets where lag is costly. Day trading,
        #               forex, futures. May overshoot in volatile conditions.
        # TIMEFRAME: All timeframes. Popular for short-term trading.
        short_window = int(parameters.get('short_window', 10))
        long_window = int(parameters.get('long_window', 30))
        price_col='Close'

        # If short_window is 0, use actual price instead of indicator
        if short_window == 0:
            short_window_indicator = 'Close'
        else:
            short_window_indicator = f'ZMA_{short_window}'
            parameters["window"] = short_window
            data, columns, fig = compute_indicator(
            data=data,
            indicator='zma',
            parameters=parameters,
            figure=False)

        long_window_indicator=f'ZMA_{long_window}'
        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='zma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

    

    # ==================== VOLATILITY STRATEGIES ====================
    elif strategy_name=='acb':
        # ACB (Acceleration Bands) - Band Breakout Strategy
        # --------------------------------------------------
        # LOGIC: Buy when price breaks above upper band (bullish breakout), sell when below lower band.
        # WHY: Acceleration Bands use percentage-based bands around price. Price breaking above
        #      upper band indicates strong uptrend momentum, below lower band indicates downtrend.
        # BEST MARKETS: Trending markets. Stocks, forex, futures. Good for breakout trading
        #               and trend following. Avoid in choppy, range-bound markets.
        # TIMEFRAME: Daily or 4-hour charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        factor = float(parameters.get('factor', 0.001))
        parameters_indicators["period"] = period
        parameters_indicators["factor"] = factor
        indicator_col = 'Close'
        price_col = 'Close'
        factor_pct = factor * 100

        data, columns, fig = compute_indicator(
        data=data,
        indicator='acb',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'ACB_Upper_{period}_{factor_pct:.2f}',
        lower_band_col=f'ACB_Lower_{period}_{factor_pct:.2f}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['Close', f'ACB_Upper_{period}_{factor_pct:.2f}', f'ACB_Middle_{period}', f'ACB_Lower_{period}_{factor_pct:.2f}']

    
    elif strategy_name=='atp':
        # ATP (Average True Range Percent) - Volatility Threshold Strategy
        # -----------------------------------------------------------------
        # LOGIC: Buy when ATRP drops below lower threshold (low volatility squeeze),
        #        sell when rises above upper threshold (high volatility).
        # WHY: ATRP normalizes ATR as percentage of price. Low ATRP indicates consolidation
        #      (potential breakout setup), high ATRP indicates overextension.
        # BEST MARKETS: All markets. Good for identifying volatility regimes and timing entries.
        #               Use low ATRP for breakout setups, high ATRP for mean reversion.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        window = int(parameters.get('window', 14))
        upper = float(parameters.get('upper', 5.0))
        lower = float(parameters.get('lower', 2.0))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'ATRP_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='atp',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'ATRP_{window}', 'lower', 'upper']

    
    elif strategy_name=='atr':
        # ATR (Average True Range) - Volatility Threshold Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when ATR drops below lower percentile (low volatility squeeze),
        #        sell when rises above upper percentile (high volatility).
        # WHY: ATR measures market volatility. Low ATR indicates consolidation and potential
        #      breakout setup. High ATR indicates strong moves or overextension.
        # BEST MARKETS: All markets. Good for volatility-based position sizing and timing.
        #               Combine with trend indicators for directional trades.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        # NOTE: Uses rolling percentile bands since ATR is in price units and varies by stock.
        window = int(parameters.get('window', 14))
        upper_pct = float(parameters.get('upper_pct', 80))  # Upper percentile threshold
        lower_pct = float(parameters.get('lower_pct', 20))  # Lower percentile threshold
        lookback = int(parameters.get('lookback', 100))  # Lookback for percentile calculation
        parameters_indicators["window"] = window
        indicator_col = f'ATR_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='atr',
        parameters=parameters_indicators,
        figure=False)

        # Calculate rolling percentile bands for ATR
        data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(upper_pct / 100)
        data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(lower_pct / 100)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'ATR_{window}', 'lower', 'upper']

    
    elif strategy_name=='bbw':
        # BBW (Bollinger Band Width) - Volatility Squeeze Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when BBW drops below lower threshold (squeeze/consolidation),
        #        sell when rises above upper threshold (volatility expansion).
        # WHY: BBW measures the width of Bollinger Bands. Low BBW indicates "The Squeeze" -
        #      a period of low volatility that often precedes significant breakouts.
        # BEST MARKETS: All markets. Excellent for identifying pre-breakout setups.
        #               Combine with price action for breakout direction.
        # TIMEFRAME: Daily charts. 20-period with 2 std dev is standard. Good for swing trading.
        window = int(parameters.get('window', 20))
        num_std = float(parameters.get('num_std', 2.0))
        upper = float(parameters.get('upper', 10.0))
        lower = float(parameters.get('lower', 4.0))
        parameters_indicators["window"] = window
        parameters_indicators["num_std"] = num_std
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'BBW_{window}_{num_std}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='bbw',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'BBW_{window}_{num_std}', 'lower', 'upper']

    
    elif strategy_name=='bol':
        # BOL (Bollinger Bands) - Mean Reversion Strategy
        # ------------------------------------------------
        # LOGIC: Buy when price drops below middle band (oversold), sell when above upper band.
        # WHY: Bollinger Bands measure volatility using standard deviations. Price tends to
        #      revert to the mean (middle band). Touches of outer bands suggest extremes.
        # BEST MARKETS: Range-bound markets and mean-reverting assets. Stocks, forex,
        #               indices. In trends, use as trailing stop rather than reversal signal.
        # TIMEFRAME: All timeframes. 20-period with 2 std dev is standard.
        window = int(parameters.get('window', 20))
        num_std = float(parameters.get('num_std', 2))
        parameters_indicators["window"] = window
        parameters_indicators["num_std"] = num_std
        indicator_col='Close'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='bol',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'BB_Upper_{window}_{num_std}',
        lower_band_col=f'BB_Middle_{window}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['Close', f'BB_Upper_{window}_{num_std}', f'BB_Middle_{window}']

    
    elif strategy_name=='cha':
        # CHA (Chaikin Volatility) - Volatility Threshold Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when Chaikin Volatility drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: Chaikin Volatility measures rate of change of the high-low range. Rising values
        #      indicate increasing volatility (often at tops/bottoms), falling values indicate
        #      decreasing volatility (consolidation).
        # BEST MARKETS: All markets. Good for identifying volatility expansion/contraction.
        #               Peaks often correlate with market turning points.
        # TIMEFRAME: Daily charts. 10-period is standard. Good for swing trading.
        ema_window = int(parameters.get('ema_window', 10))
        roc_window = int(parameters.get('roc_window', 10))
        upper = float(parameters.get('upper', 20.0))
        lower = float(parameters.get('lower', -20.0))
        parameters_indicators["ema_window"] = ema_window
        parameters_indicators["roc_window"] = roc_window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'CHAIK_{ema_window}_{roc_window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cha',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'CHAIK_{ema_window}_{roc_window}', 'lower', 'upper']

    
    elif strategy_name=='cho':
        # CHO (Choppiness Index) - Market Regime Strategy
        # ------------------------------------------------
        # LOGIC: Buy when CHOP drops below lower threshold (trending market),
        #        sell when rises above upper threshold (choppy market).
        # WHY: Choppiness Index measures whether market is trending or consolidating.
        #      Low values (<38.2) indicate trending, high values (>61.8) indicate choppy.
        # BEST MARKETS: All markets. Use to filter trend-following strategies.
        #               Avoid trend trades when CHOP is high.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for regime identification.
        period = int(parameters.get('period', 14))
        upper = float(parameters.get('upper', 61.8))
        lower = float(parameters.get('lower', 38.2))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'CHOP_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cho',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'CHOP_{period}', 'lower', 'upper']

    
    elif strategy_name=='don':
        # DON (Donchian Channels) - Breakout Strategy
        # --------------------------------------------
        # LOGIC: Buy when price breaks above upper band (bullish breakout),
        #        sell when price breaks below middle band (bearish breakout).
        # WHY: Donchian Channels plot highest high and lowest low over a period.
        #      Breakouts above/below these levels indicate strong momentum.
        #      Basis of the famous "Turtle Trading" system.
        # BEST MARKETS: Trending markets. Stocks, forex, commodities, futures.
        #               Excellent for breakout and trend-following strategies.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing/position trading.
        # NOTE: Bands are swapped because band_backtester buys when indicator < lower_band.
        #       By swapping, we get: buy when Close < Upper (i.e., Close breaks above Upper),
        #       sell when Close > Middle (i.e., Close breaks below Middle).
        window = int(parameters.get('window', 20))
        parameters_indicators["window"] = window
        indicator_col = 'Close'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='don',
        parameters=parameters_indicators,
        figure=False)

        # Shift bands by 1 to compare Close against PREVIOUS period's High/Low
        # Otherwise Close <= High <= Upper_Band (current), so Close > Upper is impossible
        data[f'DONCH_Upper_{window}'] = data[f'DONCH_Upper_{window}'].shift(1)
        data[f'DONCH_Middle_{window}'] = data[f'DONCH_Middle_{window}'].shift(1)
        data[f'DONCH_Lower_{window}'] = data[f'DONCH_Lower_{window}'].shift(1)

        # Use band_backtester with strategy_type=2 (Breakout)
        # Logic: Buy when Indicator > Upper, Sell when Indicator < Lower
        # We map: Indicator=Close, Upper=DONCH_Upper, Lower=DONCH_Middle
        # Result: Buy when Close > Upper (Breakout), Sell when Close < Middle (Trend Change)
        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'DONCH_Upper_{window}',
        lower_band_col=f'DONCH_Middle_{window}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate,
        strategy_type=2)

        indicator_cols_to_plot = ['Close', f'DONCH_Upper_{window}', f'DONCH_Middle_{window}', f'DONCH_Lower_{window}']

    
    elif strategy_name=='dvi':
        # DVI (Dynamic Volatility Index) - Mean Reversion Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when DVI drops below lower threshold (oversold),
        #        sell when rises above upper threshold (overbought).
        # WHY: DVI combines magnitude and stretch components to identify overbought/oversold
        #      conditions based on volatility-adjusted price movements.
        # BEST MARKETS: Range-bound markets. Stocks, ETFs. Good for mean reversion trading.
        #               Counter-trend entries at extremes.
        # TIMEFRAME: Daily charts. Good for short-term swing trading.
        magnitude_period = int(parameters.get('magnitude_period', 5))
        stretch_period = int(parameters.get('stretch_period', 100))
        smooth_period = int(parameters.get('smooth_period', 3))
        upper = float(parameters.get('upper', 70))
        lower = float(parameters.get('lower', 30))
        parameters_indicators["magnitude_period"] = magnitude_period
        parameters_indicators["stretch_period"] = stretch_period
        parameters_indicators["smooth_period"] = smooth_period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'DVI_{magnitude_period}_{stretch_period}_{smooth_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='dvi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'DVI_{magnitude_period}_{stretch_period}_{smooth_period}', 'lower', 'upper']

    
    elif strategy_name=='efr':
        # EFR (Efficiency Ratio) - Trend Strength Strategy
        # -------------------------------------------------
        # LOGIC: Buy when ER rises above upper threshold (strong trend),
        #        sell when drops below lower threshold (choppy market).
        # WHY: Efficiency Ratio measures how efficiently price moves. High ER (near 1)
        #      indicates trending, low ER (near 0) indicates choppy/sideways.
        # BEST MARKETS: All markets. Use to filter trend-following strategies.
        #               Trade trends when ER is high, avoid when low.
        # TIMEFRAME: Daily charts. 10-period is standard. Good for regime identification.
        period = int(parameters.get('period', 10))
        upper = float(parameters.get('upper', 0.7))
        lower = float(parameters.get('lower', 0.3))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'ER_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='efr',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'ER_{period}', 'lower', 'upper']

    
    elif strategy_name=='fdi':
        # FDI (Fractal Dimension Index) - Market Structure Strategy
        # ----------------------------------------------------------
        # LOGIC: Buy when FDI drops below lower threshold (trending/persistent),
        #        sell when rises above upper threshold (mean-reverting/jagged).
        # WHY: FDI measures market complexity. Near 1.0 = trending, near 1.5 = random,
        #      near 2.0 = mean-reverting. Helps identify market structure.
        # BEST MARKETS: All markets. Use for strategy selection based on market structure.
        #               Trend-follow when FDI < 1.5, mean-revert when FDI > 1.5.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for regime identification.
        period = int(parameters.get('period', 20))
        upper = float(parameters.get('upper', 1.6))
        lower = float(parameters.get('lower', 1.4))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'FDI_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='fdi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'FDI_{period}', 'lower', 'upper']

    
    elif strategy_name=='grv':
        # GRV (Garman-Klass Volatility) - Volatility Threshold Strategy
        # --------------------------------------------------------------
        # LOGIC: Buy when GK Volatility drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: Garman-Klass uses OHLC data for more efficient volatility estimation.
        #      Low volatility often precedes breakouts, high volatility may indicate
        #      overextension.
        # BEST MARKETS: All markets. Superior volatility estimation for options pricing,
        #               risk management, and position sizing.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        upper = float(parameters.get('upper', 30.0))
        lower = float(parameters.get('lower', 15.0))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'GK_VOL_{period}_Ann'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='grv',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'GK_VOL_{period}_Ann', 'lower', 'upper']

    
    elif strategy_name=='hav':
        # HAV (Heikin-Ashi Volatility) - Volatility Threshold Strategy
        # -------------------------------------------------------------
        # LOGIC: Buy when HAV drops below lower percentile (low volatility/consolidation),
        #        sell when rises above upper percentile (high volatility).
        # WHY: HAV applies Heikin-Ashi smoothing to filter noise before measuring volatility.
        #      Provides cleaner volatility signals than standard ATR.
        # BEST MARKETS: All markets. Good for trend identification and breakout detection.
        #               Smoother signals reduce false breakouts.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        # NOTE: Uses rolling percentile bands since HAV is in price units and varies by stock.
        period = int(parameters.get('period', 14))
        method = parameters.get('method', 'atr')
        upper_pct = float(parameters.get('upper_pct', 80))  # Upper percentile threshold
        lower_pct = float(parameters.get('lower_pct', 20))  # Lower percentile threshold
        lookback = int(parameters.get('lookback', 100))  # Lookback for percentile calculation
        parameters_indicators["period"] = period
        parameters_indicators["method"] = method
        indicator_col = f'HAV_{period}_{method.upper()}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='hav',
        parameters=parameters_indicators,
        figure=False)

        # Calculate rolling percentile bands for HAV
        data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=period).quantile(upper_pct / 100)
        data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=period).quantile(lower_pct / 100)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'HAV_{period}_{method.upper()}', 'lower', 'upper']

    
    elif strategy_name=='hiv':
        # HIV (Historical Volatility) - Volatility Threshold Strategy
        # ------------------------------------------------------------
        # LOGIC: Buy when HV drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: Historical Volatility measures realized volatility using log returns.
        #      Low HV indicates consolidation (potential breakout), high HV indicates
        #      active trending or overextension.
        # BEST MARKETS: All markets. Essential for options pricing, risk management,
        #               and position sizing. Compare with implied volatility.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        upper = float(parameters.get('upper', 30.0))
        lower = float(parameters.get('lower', 15.0))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'HV_{period}_Ann'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='hiv',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'HV_{period}_Ann', 'lower', 'upper']

    
    elif strategy_name=='kel':
        # KEL (Keltner Channel) - Mean Reversion Strategy
        # ------------------------------------------------
        # LOGIC: Buy when price drops below middle band, sell when above upper band.
        # WHY: Keltner uses ATR instead of standard deviation, making it less sensitive
        #      to price spikes. Smoother bands than Bollinger, good for trend following.
        # BEST MARKETS: Trending markets with moderate volatility. Stocks, forex,
        #               commodities. Often used with Bollinger for squeeze detection.
        # TIMEFRAME: Daily or 4-hour. ATR-based bands adapt to volatility.
        ema_window = int(parameters.get('ema_window', 20))
        atr_window = int(parameters.get('atr_window', 10))
        atr_multiplier = float(parameters.get('atr_multiplier', 2.0))
        parameters_indicators["ema_window"] = ema_window
        parameters_indicators["atr_window"] = atr_window
        parameters_indicators["atr_multiplier"] = atr_multiplier
        indicator_col='Close'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='kel',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}',
        lower_band_col=f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['Close', f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}']

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Keltner Channel Strategy")

    
    elif strategy_name=='mad':
        # MAD (Median Absolute Deviation) - Volatility Threshold Strategy
        # ----------------------------------------------------------------
        # LOGIC: Buy when MAD drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: MAD is a robust volatility measure less sensitive to outliers than
        #      standard deviation. Better for non-normal distributions.
        # BEST MARKETS: All markets, especially those with fat-tailed distributions.
        #               Good for risk management and position sizing.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        upper = float(parameters.get('upper', 2.0))
        lower = float(parameters.get('lower', 0.5))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'MAD_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='mad',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'MAD_{period}', 'lower', 'upper']

    
    elif strategy_name=='mai':
        # MAI (Mass Index) - Reversal Bulge Strategy
        # ------------------------------------------
        # LOGIC: Buy when MI rises above upper threshold then drops below it (reversal bulge),
        #        sell when MI drops below lower threshold.
        # WHY: Mass Index identifies trend reversals by measuring range expansion/contraction.
        #      The "reversal bulge" (above 27, then below 26.5) signals potential reversal.
        # BEST MARKETS: All markets. Good for identifying trend exhaustion and reversals.
        #               Does not indicate direction, only potential reversal.
        # TIMEFRAME: Daily charts. 9-period EMA, 25-period sum is standard.
        ema_period = int(parameters.get('ema_period', 9))
        sum_period = int(parameters.get('sum_period', 25))
        upper = float(parameters.get('upper', 27.0))
        lower = float(parameters.get('lower', 26.5))
        parameters_indicators["ema_period"] = ema_period
        parameters_indicators["sum_period"] = sum_period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'MI_{ema_period}_{sum_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='mai',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'MI_{ema_period}_{sum_period}', 'lower', 'upper']

    
    elif strategy_name=='nat':
        # NAT (Normalized ATR) - Volatility Threshold Strategy
        # -----------------------------------------------------
        # LOGIC: Buy when NATR drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: NATR normalizes ATR as percentage of price, allowing cross-asset comparison.
        #      Low NATR indicates consolidation, high NATR indicates active movement.
        # BEST MARKETS: All markets. Good for cross-asset volatility comparison,
        #               position sizing, and stop-loss placement.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        window = int(parameters.get('window', 14))
        upper = float(parameters.get('upper', 5.0))
        lower = float(parameters.get('lower', 2.0))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'NATR_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='nat',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'NATR_{window}', 'lower', 'upper']

    
    elif strategy_name=='pav':
        # PAV (Parkinson Volatility) - Volatility Threshold Strategy
        # -----------------------------------------------------------
        # LOGIC: Buy when Parkinson Volatility drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: Parkinson uses high-low range for more efficient volatility estimation.
        #      Better than close-to-close when no overnight gaps.
        # BEST MARKETS: All markets. Superior volatility estimation for risk management.
        #               Good for options pricing and position sizing.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        upper = float(parameters.get('upper', 30.0))
        lower = float(parameters.get('lower', 15.0))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'PARK_VOL_{period}_Ann'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='pav',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'PARK_VOL_{period}_Ann', 'lower', 'upper']

    
    elif strategy_name=='pcw':
        # PCW (Price Channel Width) - Volatility Threshold Strategy
        # ----------------------------------------------------------
        # LOGIC: Buy when PCW drops below lower threshold (narrow channel/consolidation),
        #        sell when rises above upper threshold (wide channel/trending).
        # WHY: PCW measures Donchian-style channel width as percentage of price.
        #      Low PCW indicates consolidation (potential breakout setup).
        # BEST MARKETS: All markets. Good for breakout identification and volatility
        #               comparison across assets.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        upper = float(parameters.get('upper', 15.0))
        lower = float(parameters.get('lower', 5.0))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'PCW_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='pcw',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'PCW_{period}', 'lower', 'upper']

    
    elif strategy_name=='pro':
        # PRO (Projection Oscillator) - Zero Line Crossover Strategy
        # -----------------------------------------------------------
        # LOGIC: Buy when PO crosses above zero (upward trend), sell when crosses below.
        # WHY: Projection Oscillator measures normalized slope of price movement.
        #      Positive = uptrend, negative = downtrend. Zero crossovers signal trend changes.
        # BEST MARKETS: Trending markets. Stocks, forex, indices. Good for trend
        #               strength measurement and direction identification.
        # TIMEFRAME: Daily charts. 10-period with 3-period smoothing is standard.
        period = int(parameters.get('period', 10))
        smooth_period = int(parameters.get('smooth_period', 3))
        parameters_indicators["period"] = period
        parameters_indicators["smooth_period"] = smooth_period
        short_window_indicator = f'PO_{period}_{smooth_period}'
        long_window_indicator = 'zero_line'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='pro',
        parameters=parameters_indicators,
        figure=False)

        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='rsv':
        # RSV (Rogers-Satchell Volatility) - Volatility Threshold Strategy
        # -----------------------------------------------------------------
        # LOGIC: Buy when RS Volatility drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: Rogers-Satchell accounts for drift in price movements, better for trending
        #      markets than Garman-Klass. Uses all OHLC components.
        # BEST MARKETS: Trending markets. Superior volatility estimation for options
        #               pricing and risk management in trending conditions.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        upper = float(parameters.get('upper', 30.0))
        lower = float(parameters.get('lower', 15.0))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'RS_VOL_{period}_Ann'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='rsv',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'RS_VOL_{period}_Ann', 'lower', 'upper']

    
    elif strategy_name=='rvi':
        # RVI (Relative Volatility Index) - Mean Reversion Strategy
        # ----------------------------------------------------------
        # LOGIC: Buy when RVI drops below lower threshold (oversold volatility),
        #        sell when rises above upper threshold (overbought volatility).
        # WHY: RVI applies RSI formula to standard deviation. RVI > 50 means volatility
        #      is associated with rising prices (bullish), < 50 with falling (bearish).
        # BEST MARKETS: All markets. Good for trend confirmation and divergence detection.
        #               Use for volatility direction analysis.
        # TIMEFRAME: Daily charts. 10-period std, 14-period smoothing is standard.
        window = int(parameters.get('window', 10))
        rvi_period = int(parameters.get('rvi_period', 14))
        upper = float(parameters.get('upper', 70))
        lower = float(parameters.get('lower', 30))
        parameters_indicators["window"] = window
        parameters_indicators["rvi_period"] = rvi_period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'RVI_{window}_{rvi_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='rvi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'RVI_{window}_{rvi_period}', 'lower', 'upper']

    
    elif strategy_name=='std':
        # STD (Standard Deviation) - Volatility Threshold Strategy
        # ---------------------------------------------------------
        # LOGIC: Buy when STD drops below lower threshold (low volatility),
        #        sell when rises above upper threshold (high volatility).
        # WHY: Standard deviation is the classic volatility measure. Low STD indicates
        #      consolidation (potential breakout), high STD indicates active movement.
        # BEST MARKETS: All markets. Fundamental volatility measure for risk assessment,
        #               position sizing, and Bollinger Bands component.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        window = int(parameters.get('window', 20))
        upper = float(parameters.get('upper', 5.0))
        lower = float(parameters.get('lower', 2.0))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'STD_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='std',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'STD_{window}', 'lower', 'upper']

    
    elif strategy_name=='svi':
        # SVI (Stochastic Volatility Index) - Mean Reversion Strategy
        # ------------------------------------------------------------
        # LOGIC: Buy when SVI drops below lower threshold (low volatility regime),
        #        sell when rises above upper threshold (high volatility regime).
        # WHY: SVI applies stochastic formula to ATR, creating normalized 0-100 oscillator.
        #      Low SVI indicates low volatility (potential breakout), high SVI indicates
        #      high volatility regime.
        # BEST MARKETS: All markets. Good for volatility regime identification and
        #               breakout prediction from low volatility.
        # TIMEFRAME: Daily charts. 14-period ATR, 14-period stochastic is standard.
        atr_period = int(parameters.get('atr_period', 14))
        stoch_period = int(parameters.get('stoch_period', 14))
        smooth_k = int(parameters.get('smooth_k', 3))
        smooth_d = int(parameters.get('smooth_d', 3))
        upper = float(parameters.get('upper', 80))
        lower = float(parameters.get('lower', 20))
        parameters_indicators["atr_period"] = atr_period
        parameters_indicators["stoch_period"] = stoch_period
        parameters_indicators["smooth_k"] = smooth_k
        parameters_indicators["smooth_d"] = smooth_d
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'SVI_K_{atr_period}_{stoch_period}_{smooth_k}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='svi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'SVI_K_{atr_period}_{stoch_period}_{smooth_k}', f'SVI_D_{atr_period}_{stoch_period}_{smooth_d}', 'lower', 'upper']

    
    elif strategy_name=='tsv':
        # TSI_VOL (TSI Volatility) - Zero Line Crossover Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when TSI Volatility crosses above zero (rising volatility momentum),
        #        sell when crosses below zero (falling volatility momentum).
        # WHY: TSI Volatility applies double-smoothed momentum to ATR. Positive = rising
        #      volatility, negative = falling volatility. Zero crossovers signal changes.
        # BEST MARKETS: All markets. Good for volatility trend identification and
        #               divergence detection between price and volatility momentum.
        # TIMEFRAME: Daily charts. 14-period ATR, 25/13 smoothing is standard.
        atr_period = int(parameters.get('atr_period', 14))
        long_period = int(parameters.get('long_period', 25))
        short_period = int(parameters.get('short_period', 13))
        parameters_indicators["atr_period"] = atr_period
        parameters_indicators["long_period"] = long_period
        parameters_indicators["short_period"] = short_period
        short_window_indicator = f'TSI_VOL_{atr_period}_{long_period}_{short_period}'
        long_window_indicator = 'zero_line'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='tsv',
        parameters=parameters_indicators,
        figure=False)

        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='uli':
        # ULI (Ulcer Index) - Downside Risk Strategy
        # ------------------------------------------
        # LOGIC: Buy when UI drops below lower threshold (low drawdown risk),
        #        sell when rises above upper threshold (high drawdown risk).
        # WHY: Ulcer Index measures downside risk by focusing on depth and duration of
        #      drawdowns. Unlike STD, it doesn't penalize upside volatility.
        # BEST MARKETS: All markets. Essential for downside risk measurement, portfolio
        #               optimization, and drawdown monitoring.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        period = int(parameters.get('period', 14))
        upper = float(parameters.get('upper', 5.0))
        lower = float(parameters.get('lower', 1.0))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'UI_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='uli',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'UI_{period}', 'lower', 'upper']

    
    elif strategy_name=='vhf':
        # VHF (Vertical Horizontal Filter) - Trend Strength Strategy
        # -----------------------------------------------------------
        # LOGIC: Buy when VHF rises above upper threshold (strong trend),
        #        sell when drops below lower threshold (congestion).
        # WHY: VHF determines if prices are trending or in congestion by comparing
        #      price range to sum of price changes. High VHF = trending, low = choppy.
        # BEST MARKETS: All markets. Use for trend identification and indicator selection.
        #               Use moving averages when VHF high, oscillators when VHF low.
        # TIMEFRAME: Daily charts. 28-period is standard. Good for regime identification.
        period = int(parameters.get('period', 28))
        upper = float(parameters.get('upper', 0.40))
        lower = float(parameters.get('lower', 0.25))
        parameters_indicators["period"] = period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'VHF_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vhf',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'VHF_{period}', 'lower', 'upper']

    
    elif strategy_name=='vra':
        # VOR_VOL (Volatility Ratio) - Regime Change Strategy
        # ----------------------------------------------------
        # LOGIC: Buy when VR rises above upper threshold (volatility expansion),
        #        sell when drops below lower threshold (volatility contraction).
        # WHY: Volatility Ratio compares short-term to long-term volatility. VR > 1
        #      indicates expanding volatility (potential breakout), < 1 indicates contraction.
        # BEST MARKETS: All markets. Good for volatility regime detection and breakout
        #               confirmation. Strategy switching between breakout and mean reversion.
        # TIMEFRAME: Daily charts. 5/20 periods is standard. Good for swing trading.
        short_period = int(parameters.get('short_period', 5))
        long_period = int(parameters.get('long_period', 20))
        upper = float(parameters.get('upper', 1.5))
        lower = float(parameters.get('lower', 0.8))
        parameters_indicators["short_period"] = short_period
        parameters_indicators["long_period"] = long_period
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'VR_{short_period}_{long_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vra',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'VR_{short_period}_{long_period}', 'lower', 'upper']

    
    elif strategy_name=='vqi':
        # VQI (Volatility Quality Index) - Zero Line Crossover Strategy
        # --------------------------------------------------------------
        # LOGIC: Buy when VQI crosses above zero (quality uptrend), sell when crosses below.
        # WHY: VQI measures quality of price movements by analyzing price changes, volume,
        #      and volatility. Positive = quality uptrend, negative = quality downtrend.
        # BEST MARKETS: Markets with volume data. Good for trend quality assessment and
        #               filtering false moves. Divergence detection.
        # TIMEFRAME: Daily charts. 9-period with 9-period smoothing is standard.
        period = int(parameters.get('period', 9))
        smooth_period = int(parameters.get('smooth_period', 9))
        parameters_indicators["period"] = period
        parameters_indicators["smooth_period"] = smooth_period
        short_window_indicator = f'VQI_{period}_{smooth_period}'
        long_window_indicator = 'zero_line'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vqi',
        parameters=parameters_indicators,
        figure=False)

        data['zero_line'] = 0

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, 'zero_line']

    
    elif strategy_name=='vsi':
        # VSI (Volatility Switch Index) - Binary Regime Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when VSI switches to 0 (low volatility regime),
        #        sell when switches to 1 (high volatility regime).
        # WHY: VSI is a binary indicator that identifies volatility regime changes.
        #      VSI = 1 means elevated volatility, VSI = 0 means baseline volatility.
        # BEST MARKETS: All markets. Good for binary regime identification and strategy
        #               switching. Reduce size when VSI = 1.
        # TIMEFRAME: Daily charts. 10/50 periods with 1.2 threshold is standard.
        short_period = int(parameters.get('short_period', 10))
        long_period = int(parameters.get('long_period', 50))
        threshold = float(parameters.get('threshold', 1.2))
        upper = float(parameters.get('upper', 0.5))
        lower = float(parameters.get('lower', 0.5))
        parameters_indicators["short_period"] = short_period
        parameters_indicators["long_period"] = long_period
        parameters_indicators["threshold"] = threshold
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col = f'VSI_{short_period}_{long_period}_{threshold}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vsi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'VSI_{short_period}_{long_period}_{threshold}', 'lower', 'upper']



    # ==================== VOLUME STRATEGIES ====================

    elif strategy_name=='adl':
        # ADL (Accumulation/Distribution Line) - Trend Confirmation Strategy
        # -------------------------------------------------------------------
        # LOGIC: Buy when ADL crosses above its SMA (accumulation),
        #        sell when ADL crosses below its SMA (distribution).
        # WHY: ADL measures cumulative money flow. Rising ADL indicates buying pressure,
        #      falling ADL indicates selling pressure. Divergence with price signals reversals.
        # BEST MARKETS: Stocks, ETFs. Good for confirming price trends with volume.
        # TIMEFRAME: Daily charts. Good for swing trading and trend confirmation.
        sma_period = int(parameters.get('sma_period', 20))
        parameters_indicators["sma_period"] = sma_period
        indicator_col = 'ADLINE'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='adl',
        parameters=parameters_indicators,
        figure=False)

        # Calculate SMA of ADL for crossover signals
        data[f'ADLINE_SMA_{sma_period}'] = data['ADLINE'].rolling(window=sma_period).mean()

        # Use cross trade: buy when ADL crosses above SMA, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='ADLINE',
        long_window_indicator=f'ADLINE_SMA_{sma_period}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['ADLINE', f'ADLINE_SMA_{sma_period}']

    
    elif strategy_name=='ado':
        # ADO (Accumulation/Distribution Oscillator) - Zero Line Cross Strategy
        # ----------------------------------------------------------------------
        # LOGIC: Buy when ADO crosses above zero (accumulation momentum),
        #        sell when ADO crosses below zero (distribution momentum).
        # WHY: ADO measures the rate of change of the A/D Line. Positive values indicate
        #      increasing accumulation, negative values indicate increasing distribution.
        # BEST MARKETS: Stocks, ETFs. Good for momentum-based volume analysis.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        period = int(parameters.get('period', 14))
        parameters_indicators["period"] = period
        indicator_col = f'ADO_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='ado',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when ADO crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'ADO_{period}', 'zero']

    
    elif strategy_name=='bwm':
        # BWM (Bill Williams Market Facilitation Index) - Percentile Strategy
        # --------------------------------------------------------------------
        # LOGIC: Buy when BWMFI drops below lower percentile (low facilitation),
        #        sell when rises above upper percentile (high facilitation).
        # WHY: BWMFI measures price movement efficiency per unit of volume.
        #      Low values indicate consolidation, high values indicate strong moves.
        # BEST MARKETS: All markets. Good for identifying breakout potential.
        # TIMEFRAME: Daily charts. Good for swing trading.
        # NOTE: Uses rolling percentile bands since BWMFI values vary by asset.
        upper_pct = float(parameters.get('upper_pct', 80))
        lower_pct = float(parameters.get('lower_pct', 20))
        lookback = int(parameters.get('lookback', 100))
        indicator_col = 'BWMFI'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='bwm',
        parameters=parameters_indicators,
        figure=False)

        # Calculate rolling percentile bands
        data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=20).quantile(upper_pct / 100)
        data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=20).quantile(lower_pct / 100)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['BWMFI', 'lower', 'upper']

    
    elif strategy_name=='cmf':
        # CMF (Chaikin Money Flow) - Zero Line Cross Strategy
        # ----------------------------------------------------
        # LOGIC: Buy when CMF crosses above zero (buying pressure),
        #        sell when CMF crosses below zero (selling pressure).
        # WHY: CMF measures money flow over a period. Positive CMF indicates accumulation,
        #      negative CMF indicates distribution. Good for trend confirmation.
        # BEST MARKETS: Stocks, ETFs. Good for confirming breakouts and trends.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        period = int(parameters.get('period', 20))
        parameters_indicators["period"] = period
        indicator_col = f'CMF_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cmf',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when CMF crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'CMF_{period}', 'zero']

    
    elif strategy_name=='emv':
        # EMV (Ease of Movement) - Zero Line Cross Strategy
        # --------------------------------------------------
        # LOGIC: Buy when EMV crosses above zero (price rising easily),
        #        sell when EMV crosses below zero (price falling easily).
        # WHY: EMV relates price change to volume. Positive EMV means price moves up
        #      with ease, negative EMV means price moves down with ease.
        # BEST MARKETS: Stocks, ETFs. Good for volume-weighted momentum analysis.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        period = int(parameters.get('period', 14))
        parameters_indicators["period"] = period
        indicator_col = f'EMV_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='emv',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when EMV crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'EMV_{period}', 'zero']

    
    elif strategy_name=='foi':
        # FOI (Force Index) - Zero Line Cross Strategy
        # ---------------------------------------------
        # LOGIC: Buy when Force Index crosses above zero (buying pressure),
        #        sell when Force Index crosses below zero (selling pressure).
        # WHY: Force Index combines price change and volume to measure the power
        #      behind price moves. Positive values indicate bulls in control.
        # BEST MARKETS: Stocks, ETFs. Good for trend confirmation and divergence.
        # TIMEFRAME: Daily charts. 13-period EMA is standard. Good for swing trading.
        period = int(parameters.get('period', 13))
        parameters_indicators["period"] = period
        indicator_col = f'FI_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='foi',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when FI crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'FI_{period}', 'zero']

    
    elif strategy_name=='fve':
        # FVE (Finite Volume Elements) - Zero Line Cross Strategy
        # --------------------------------------------------------
        # LOGIC: Buy when FVE crosses above zero (bullish money flow),
        #        sell when FVE crosses below zero (bearish money flow).
        # WHY: FVE separates volume into bullish/bearish components based on price action.
        #      Positive FVE indicates net buying, negative indicates net selling.
        # BEST MARKETS: Stocks, ETFs. Good for money flow analysis with volatility filter.
        # TIMEFRAME: Daily charts. 22-period is standard. Good for swing trading.
        period = int(parameters.get('period', 22))
        factor = float(parameters.get('factor', 0.3))
        parameters_indicators["period"] = period
        parameters_indicators["factor"] = factor
        indicator_col = f'FVE_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='fve',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when FVE crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'FVE_{period}', 'zero']

    
    elif strategy_name=='kvo':
        # KVO (Klinger Volume Oscillator) - Signal Line Cross Strategy
        # -------------------------------------------------------------
        # LOGIC: Buy when KVO crosses above Signal line (bullish momentum),
        #        sell when KVO crosses below Signal line (bearish momentum).
        # WHY: KVO is a long-term money flow indicator comparing fast and slow volume EMAs.
        #      Signal line crossovers indicate changes in money flow momentum.
        # BEST MARKETS: Stocks, ETFs. Good for long-term trend identification.
        # TIMEFRAME: Daily charts. 34/55/13 periods are standard. Good for position trading.
        fast_period = int(parameters.get('fast_period', 34))
        slow_period = int(parameters.get('slow_period', 55))
        signal_period = int(parameters.get('signal_period', 13))
        parameters_indicators["fast_period"] = fast_period
        parameters_indicators["slow_period"] = slow_period
        parameters_indicators["signal_period"] = signal_period
        indicator_col = f'KVO_{fast_period}_{slow_period}'
        signal_col = f'KVO_SIGNAL_{signal_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='kvo',
        parameters=parameters_indicators,
        figure=False)

        # Use cross trade: buy when KVO crosses above Signal, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator=signal_col,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'KVO_{fast_period}_{slow_period}', f'KVO_SIGNAL_{signal_period}']

    
    elif strategy_name=='mfi':
        # MFI (Money Flow Index) - Overbought/Oversold Strategy
        # ------------------------------------------------------
        # LOGIC: Buy when MFI drops below lower threshold (oversold),
        #        sell when MFI rises above upper threshold (overbought).
        # WHY: MFI is volume-weighted RSI. Values below 20 indicate oversold conditions,
        #      above 80 indicate overbought. Good for mean reversion trading.
        # BEST MARKETS: Stocks, ETFs. Good for range-bound markets and reversals.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        period = int(parameters.get('period', 14))
        upper = float(parameters.get('upper', 80))
        lower = float(parameters.get('lower', 20))
        parameters_indicators["period"] = period
        indicator_col = f'MFI_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='mfi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'MFI_{period}', 'lower', 'upper']

    
    elif strategy_name=='nvi':
        # NVI (Negative Volume Index) - SMA Cross Strategy
        # -------------------------------------------------
        # LOGIC: Buy when NVI crosses above its SMA (smart money accumulating),
        #        sell when NVI crosses below its SMA (smart money distributing).
        # WHY: NVI tracks price changes on low volume days, believed to reflect
        #      "smart money" activity. Rising NVI indicates institutional accumulation.
        # BEST MARKETS: Stocks, ETFs. Good for long-term trend identification.
        # TIMEFRAME: Daily charts. 255-period SMA is traditional. Good for position trading.
        sma_period = int(parameters.get('sma_period', 255))
        parameters_indicators["sma_period"] = sma_period
        indicator_col = 'NVI'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='nvi',
        parameters=parameters_indicators,
        figure=False)

        # Calculate SMA of NVI for crossover signals
        data[f'NVI_SMA_{sma_period}'] = data['NVI'].rolling(window=sma_period).mean()

        # Use cross trade: buy when NVI crosses above SMA, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='NVI',
        long_window_indicator=f'NVI_SMA_{sma_period}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['NVI', f'NVI_SMA_{sma_period}']

    
    elif strategy_name=='obv':
        # OBV (On-Balance Volume) - SMA Cross Strategy
        # ---------------------------------------------
        # LOGIC: Buy when OBV crosses above its SMA (accumulation),
        #        sell when OBV crosses below its SMA (distribution).
        # WHY: OBV measures buying/selling pressure by adding volume on up days
        #      and subtracting on down days. Rising OBV confirms uptrend.
        # BEST MARKETS: Stocks, ETFs. Good for trend confirmation and divergence.
        # TIMEFRAME: Daily charts. Good for swing trading and trend following.
        sma_period = int(parameters.get('sma_period', 20))
        parameters_indicators["sma_period"] = sma_period
        indicator_col = 'OBV'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='obv',
        parameters=parameters_indicators,
        figure=False)

        # Calculate SMA of OBV for crossover signals
        data[f'OBV_SMA_{sma_period}'] = data['OBV'].rolling(window=sma_period).mean()

        # Use cross trade: buy when OBV crosses above SMA, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='OBV',
        long_window_indicator=f'OBV_SMA_{sma_period}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['OBV', f'OBV_SMA_{sma_period}']

    
    elif strategy_name=='pvi':
        # PVI (Positive Volume Index) - SMA Cross Strategy
        # -------------------------------------------------
        # LOGIC: Buy when PVI crosses above its SMA (crowd buying),
        #        sell when PVI crosses below its SMA (crowd selling).
        # WHY: PVI tracks price changes on high volume days, reflecting
        #      "crowd" or uninformed investor activity. Used with NVI for confirmation.
        # BEST MARKETS: Stocks, ETFs. Good for sentiment analysis.
        # TIMEFRAME: Daily charts. 255-period SMA is traditional. Good for position trading.
        sma_period = int(parameters.get('sma_period', 255))
        parameters_indicators["sma_period"] = sma_period
        indicator_col = 'PVI'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='pvi',
        parameters=parameters_indicators,
        figure=False)

        # Calculate SMA of PVI for crossover signals
        data[f'PVI_SMA_{sma_period}'] = data['PVI'].rolling(window=sma_period).mean()

        # Use cross trade: buy when PVI crosses above SMA, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='PVI',
        long_window_indicator=f'PVI_SMA_{sma_period}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['PVI', f'PVI_SMA_{sma_period}']

    
    elif strategy_name=='pvo':
        # PVO (Percentage Volume Oscillator) - Signal Line Cross Strategy
        # ----------------------------------------------------------------
        # LOGIC: Buy when PVO crosses above Signal line (volume momentum increasing),
        #        sell when PVO crosses below Signal line (volume momentum decreasing).
        # WHY: PVO is like MACD but for volume. It shows the relationship between
        #      fast and slow volume EMAs. Signal crossovers indicate volume trend changes.
        # BEST MARKETS: Stocks, ETFs. Good for confirming breakouts and trends.
        # TIMEFRAME: Daily charts. 12/26/9 periods are standard. Good for swing trading.
        fast_period = int(parameters.get('fast_period', 12))
        slow_period = int(parameters.get('slow_period', 26))
        signal_period = int(parameters.get('signal_period', 9))
        parameters_indicators["fast_period"] = fast_period
        parameters_indicators["slow_period"] = slow_period
        parameters_indicators["signal_period"] = signal_period
        indicator_col = f'PVO_{fast_period}_{slow_period}'
        signal_col = f'PVO_SIGNAL_{signal_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='pvo',
        parameters=parameters_indicators,
        figure=False)

        # Use cross trade: buy when PVO crosses above Signal, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator=signal_col,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'PVO_{fast_period}_{slow_period}', f'PVO_SIGNAL_{signal_period}']

    
    elif strategy_name=='vfi':
        # VFI (Volume Flow Indicator) - Zero Line Cross Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when VFI crosses above zero (bullish money flow),
        #        sell when VFI crosses below zero (bearish money flow).
        # WHY: VFI is a long-term trend-following indicator based on OBV but with
        #      noise reduction. Positive VFI indicates accumulation.
        # BEST MARKETS: Stocks, ETFs. Good for long-term trend identification.
        # TIMEFRAME: Daily charts. 130-period is standard. Good for position trading.
        period = int(parameters.get('period', 130))
        coef = float(parameters.get('coef', 0.2))
        vcoef = float(parameters.get('vcoef', 2.5))
        smoothing_period = int(parameters.get('smoothing_period', 3))
        parameters_indicators["period"] = period
        parameters_indicators["coef"] = coef
        parameters_indicators["vcoef"] = vcoef
        parameters_indicators["smoothing_period"] = smoothing_period
        indicator_col = f'VFI_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vfi',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when VFI crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'VFI_{period}', 'zero']

    
    elif strategy_name=='vma':
        # VMA (Volume Moving Average) - Price Cross Strategy
        # ---------------------------------------------------
        # LOGIC: Buy when Close crosses above VMA (bullish, price above fair value),
        #        sell when Close crosses below VMA (bearish, price below fair value).
        # WHY: VMA is a volume-weighted moving average that gives more weight to
        #      prices with higher volume. Acts as dynamic support/resistance.
        # BEST MARKETS: Stocks, ETFs. Good for trend following with volume confirmation.
        # TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
        window = int(parameters.get('window', 20))
        parameters_indicators["window"] = window
        indicator_col = f'VMA_{window}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vma',
        parameters=parameters_indicators,
        figure=False)

        # Use cross trade: buy when Close crosses above VMA, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='Close',
        long_window_indicator=indicator_col,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['Close', f'VMA_{window}']

    
    elif strategy_name=='voo':
        # VOO (Volume Oscillator) - Zero Line Cross Strategy
        # ---------------------------------------------------
        # LOGIC: Buy when VO crosses above zero (volume increasing),
        #        sell when VO crosses below zero (volume decreasing).
        # WHY: VO shows the difference between fast and slow volume SMAs.
        #      Positive VO indicates increasing market participation.
        # BEST MARKETS: Stocks, ETFs. Good for confirming breakouts.
        # TIMEFRAME: Daily charts. 5/10 periods are standard. Good for swing trading.
        fast_period = int(parameters.get('fast_period', 5))
        slow_period = int(parameters.get('slow_period', 10))
        parameters_indicators["fast_period"] = fast_period
        parameters_indicators["slow_period"] = slow_period
        indicator_col = f'VO_{fast_period}_{slow_period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='voo',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when VO crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'VO_{fast_period}_{slow_period}', 'zero']

    
    elif strategy_name=='vpt':
        # VPT (Volume Price Trend) - SMA Cross Strategy
        # ----------------------------------------------
        # LOGIC: Buy when VPT crosses above its SMA (accumulation),
        #        sell when VPT crosses below its SMA (distribution).
        # WHY: VPT relates volume to percentage price change. Rising VPT indicates
        #      buying pressure, falling VPT indicates selling pressure.
        # BEST MARKETS: Stocks, ETFs. Good for trend confirmation and divergence.
        # TIMEFRAME: Daily charts. Good for swing trading.
        sma_period = int(parameters.get('sma_period', 20))
        parameters_indicators["sma_period"] = sma_period
        indicator_col = 'VPT'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vpt',
        parameters=parameters_indicators,
        figure=False)

        # Calculate SMA of VPT for crossover signals
        data[f'VPT_SMA_{sma_period}'] = data['VPT'].rolling(window=sma_period).mean()

        # Use cross trade: buy when VPT crosses above SMA, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='VPT',
        long_window_indicator=f'VPT_SMA_{sma_period}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['VPT', f'VPT_SMA_{sma_period}']

    
    elif strategy_name=='vro':
        # VRO (Volume Rate of Change) - Zero Line Cross Strategy
        # -------------------------------------------------------
        # LOGIC: Buy when VROC crosses above zero (volume increasing),
        #        sell when VROC crosses below zero (volume decreasing).
        # WHY: VROC measures the rate of change in volume. Positive VROC indicates
        #      increasing trading activity, often accompanying breakouts.
        # BEST MARKETS: Stocks, ETFs. Good for breakout confirmation.
        # TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
        period = int(parameters.get('period', 14))
        parameters_indicators["period"] = period
        indicator_col = f'VROC_{period}'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='vro',
        parameters=parameters_indicators,
        figure=False)

        # Create zero line for crossover
        data['zero'] = 0

        # Use cross trade: buy when VROC crosses above zero, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'VROC_{period}', 'zero']

    
    elif strategy_name=='vwa':
        # VWA (Volume Weighted Average Price) - Rolling VWAP Cross Strategy
        # ------------------------------------------------------------------
        # LOGIC: Buy when Close crosses above rolling VWAP (bullish, price above fair value),
        #        sell when Close crosses below rolling VWAP (bearish, price below fair value).
        # WHY: Rolling VWAP gives the average price weighted by volume over a lookback period.
        #      Price above VWAP indicates buyers are in control, below indicates sellers.
        # NOTE: Standard cumulative VWAP becomes too smooth for swing trading. We use a
        #       rolling window VWAP to generate meaningful crossover signals.
        # BEST MARKETS: Stocks, ETFs. Good for intraday and swing trading.
        # TIMEFRAME: Daily charts. 20-period rolling window is standard for swing trading.
        window = int(parameters.get('window', 20))
        parameters_indicators["window"] = window
        price_col = 'Close'

        # Calculate rolling VWAP manually since the indicator uses cumulative
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        tpv = typical_price * data['Volume']
        rolling_tpv = tpv.rolling(window=window).sum()
        rolling_volume = data['Volume'].rolling(window=window).sum()
        data[f'VWAP_{window}'] = rolling_tpv / rolling_volume

        # Use cross trade: buy when Close crosses above rolling VWAP, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='Close',
        long_window_indicator=f'VWAP_{window}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['Close', f'VWAP_{window}']

    
    elif strategy_name=='wad':
        # WAD (Williams Accumulation/Distribution) - SMA Cross Strategy
        # --------------------------------------------------------------
        # LOGIC: Buy when WAD crosses above its SMA (accumulation),
        #        sell when WAD crosses below its SMA (distribution).
        # WHY: WAD uses True Range to measure accumulation/distribution pressure.
        #      Rising WAD indicates buying pressure, falling indicates selling.
        # BEST MARKETS: Stocks, ETFs. Good for trend confirmation and divergence.
        # TIMEFRAME: Daily charts. Good for swing trading.
        sma_period = int(parameters.get('sma_period', 20))
        parameters_indicators["sma_period"] = sma_period
        indicator_col = 'WAD'
        price_col = 'Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='wad',
        parameters=parameters_indicators,
        figure=False)

        # Calculate SMA of WAD for crossover signals
        data[f'WAD_SMA_{sma_period}'] = data['WAD'].rolling(window=sma_period).mean()

        # Use cross trade: buy when WAD crosses above SMA, sell when crosses below
        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator='WAD',
        long_window_indicator=f'WAD_SMA_{sma_period}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['WAD', f'WAD_SMA_{sma_period}']


    # Generic plotting for strategies that don't have custom plotting
    if fig_control==1 and fig is None and 'indicator_cols_to_plot' in locals():
        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col if 'price_col' in locals() else 'Close',
        indicator_cols=indicator_cols_to_plot, 
        title=f"{strategy_name.upper()} Strategy")

    return results, portfolio, fig
