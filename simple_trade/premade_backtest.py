import pandas as pd

from .indicator_handler import compute_indicator
from .cross_trade import CrossTradeBacktester
from .band_trade import BandTradeBacktester
from .plot_test import BacktestPlotter


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
    if strategy_name=='cci':
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

    
    elif strategy_name=='mac':
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

    
    elif strategy_name=='rsi':
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

    
    elif strategy_name=='sto':
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

    

    # ==================== TREND STRATEGIES ====================
    elif strategy_name == 'ads':
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
    elif strategy_name=='bol':
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

    
    elif strategy_name=='kel':
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

    # Generic plotting for strategies that don't have custom plotting
    if fig_control==1 and fig is None and 'indicator_cols_to_plot' in locals():
        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col if 'price_col' in locals() else 'Close',
        indicator_cols=indicator_cols_to_plot, 
        title=f"{strategy_name.upper()} Strategy")

    return results, portfolio, fig
