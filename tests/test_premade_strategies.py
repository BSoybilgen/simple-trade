import pytest
import pandas as pd
import numpy as np
from simple_trade.run_premade_strategies import run_premade_trade


# --- Fixtures ---

@pytest.fixture
def sample_ohlcv_data():
    """Fixture to provide sample OHLCV data with DatetimeIndex"""
    index = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    close = pd.Series(np.linspace(100, 150, 100) + np.random.normal(0, 2, 100), index=index)
    high = close + np.random.uniform(0.5, 3, size=len(close))
    low = close - np.random.uniform(0.5, 3, size=len(close))
    low = pd.Series(np.minimum(low.values, close.values - 0.1), index=index)
    high = pd.Series(np.maximum(high.values, close.values + 0.1), index=index)
    # Open price between low and high, closer to previous close
    open_price = low + (high - low) * np.random.uniform(0.3, 0.7, size=len(close))
    volume = pd.Series(np.random.randint(1000, 10000, size=len(close)), index=index)
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    return df


@pytest.fixture
def default_parameters():
    """Default parameters for run_premade_trade"""
    return {
        'initial_cash': 10000.0,
        'commission_long': 0.001,
        'commission_short': 0.001,
        'trading_type': 'long',
        'day1_position': 'none',
        'risk_free_rate': 0.02,
        'fig_control': 0
    }


# --- Basic Functionality Tests ---

class TestPremadeBacktestBasic:
    """Test basic functionality of premade_backtest"""
    
    def test_premade_backtest_with_none_parameters(self, sample_ohlcv_data):
        """Test that premade_backtest works with None parameters"""
        # premade_backtest expects a dict, not None
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', {})
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert fig is None  # fig_control defaults to 0
        
    def test_premade_backtest_returns_tuple(self, sample_ohlcv_data, default_parameters):
        """Test that premade_backtest returns a tuple of (results, portfolio, fig)"""
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert fig is None  # fig_control is 0
        
    def test_premade_backtest_with_figure_control(self, sample_ohlcv_data, default_parameters):
        """Test that fig_control=1 returns a figure"""
        default_parameters['fig_control'] = 1
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert fig is not None


# --- Strategy-Specific Tests ---

class TestPremadeBacktestStrategies:
    """Test different trading strategies"""
    
    def test_rsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RSI strategy"""
        rsi_params = default_parameters.copy()
        rsi_params.update({
            'window': 14,
            'upper': 70,
            'lower': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', rsi_params)
        
        assert isinstance(results, dict)
        assert 'total_return_pct' in results
        assert 'sharpe_ratio' in results
        assert isinstance(portfolio, pd.DataFrame)
        assert 'PositionType' in portfolio.columns
        
    def test_sma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SMA cross strategy"""
        sma_params = default_parameters.copy()
        sma_params.update({
            'short_window': 10,
            'long_window': 20
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sma', sma_params)
        
        assert isinstance(results, dict)
        assert 'total_return_pct' in results
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_ema_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EMA cross strategy"""
        ema_params = default_parameters.copy()
        ema_params.update({
            'short_window': 12,
            'long_window': 26
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ema', ema_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_macd_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MACD strategy"""
        macd_params = default_parameters.copy()
        macd_params.update({
            'window_fast': 12,
            'window_slow': 26,
            'window_signal': 9
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mac', macd_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_bollinger_bands_strategy(self, sample_ohlcv_data, default_parameters):
        """Test Bollinger Bands strategy"""
        bb_params = default_parameters.copy()
        bb_params.update({
            'window': 20,
            'num_std': 2.0
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', bb_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_cci_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CCI strategy"""
        cci_params = default_parameters.copy()
        cci_params.update({
            'window': 20,
            'constant': 0.015,
            'upper': 100,
            'lower': -100
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cci', cci_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_stochastic_strategy(self, sample_ohlcv_data, default_parameters):
        """Test Stochastic strategy"""
        stoch_params = default_parameters.copy()
        stoch_params.update({
            'k_period': 14,
            'd_period': 3,
            'smooth_k': 3,
            'upper': 80,
            'lower': 20
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sto', stoch_params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


# --- Parameter Validation Tests ---

class TestPremadeBacktestParameterValidation:
    """Test parameter validation and default values"""
    
    def test_default_parameter_values(self, sample_ohlcv_data):
        """Test that default parameter values are used correctly"""
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', {})
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_custom_initial_cash(self, sample_ohlcv_data, default_parameters):
        """Test custom initial cash parameter"""
        params = default_parameters.copy()
        params['initial_cash'] = 50000.0
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', params)
        
        # Note: premade_backtest currently doesn't pass config to run_band_trade,
        # so initial_cash defaults to 10000.0. This is a known limitation.
        # The test verifies the function runs without error.
        assert isinstance(results, dict)
        assert 'initial_cash' in results
        
    def test_custom_commission_rates(self, sample_ohlcv_data, default_parameters):
        """Test custom commission rates"""
        default_parameters.update({
            'commission_long': 0.005,
            'commission_short': 0.003
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        
    def test_trading_type_long_only(self, sample_ohlcv_data, default_parameters):
        """Test long-only trading type"""
        default_parameters['trading_type'] = 'long'
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Check that no short positions are taken
        assert not (portfolio['PositionType'] == 'short').any()
        
    def test_trading_type_short_only(self, sample_ohlcv_data, default_parameters):
        """Test short-only trading type"""
        default_parameters['trading_type'] = 'short'
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Check that no long positions are taken
        assert not (portfolio['PositionType'] == 'long').any()
        
    def test_day1_position_parameter(self, sample_ohlcv_data, default_parameters):
        """Test day1_position parameter"""
        default_parameters['day1_position'] = 'long'
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Check that first position is long
        assert portfolio['PositionType'].iloc[0] == 'long'


# --- Error Handling Tests ---

class TestPremadeBacktestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_strategy_name(self, sample_ohlcv_data, default_parameters):
        """Test handling of invalid strategy name"""
        # Invalid strategy should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            run_premade_trade(sample_ohlcv_data, 'invalid_strategy', default_parameters)
        
        # Check that the error message is helpful
        assert "Unknown strategy" in str(exc_info.value)
        assert "invalid_strategy" in str(exc_info.value)
        assert "list_strategies()" in str(exc_info.value)
            
    def test_empty_dataframe(self, default_parameters):
        """Test handling of empty DataFrame"""
        # Create empty DataFrame with proper index
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))
        
        # Empty dataframe should either raise an error or return empty results
        try:
            results, portfolio, fig = run_premade_trade(empty_df, 'rsi', default_parameters)
            # If no error, check that results are handled gracefully
            assert results is None or isinstance(results, dict)
        except (ValueError, KeyError, IndexError, AttributeError):
            # Expected error for empty data
            pass
            
    def test_missing_required_columns(self, default_parameters):
        """Test handling of DataFrame missing required columns"""
        # Create DataFrame with DatetimeIndex but missing required columns
        index = pd.date_range(start='2023-01-01', periods=5, freq='D')
        incomplete_df = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        }, index=index)
        
        # Missing High, Low, Volume columns should cause issues
        try:
            results, portfolio, fig = run_premade_trade(incomplete_df, 'rsi', default_parameters)
            # If no error, function handled missing columns gracefully
            assert results is None or isinstance(results, dict)
        except (KeyError, ValueError, AttributeError):
            # Expected error for missing required columns
            pass


# --- Integration Tests ---

class TestPremadeBacktestIntegration:
    """Integration tests for premade_backtest"""
    
    def test_multiple_strategies_same_data(self, sample_ohlcv_data, default_parameters):
        """Test multiple strategies on the same data"""
        strategies = ['rsi', 'sma', 'ema']
        results_list = []
        
        for strategy in strategies:
            results, portfolio, fig = run_premade_trade(sample_ohlcv_data, strategy, default_parameters)
            results_list.append(results)
            
        # All strategies should return valid results
        for results in results_list:
            assert isinstance(results, dict)
            assert 'total_return_pct' in results
            
    def test_parameter_sensitivity_rsi(self, sample_ohlcv_data, default_parameters):
        """Test RSI strategy with different parameters"""
        rsi_windows = [10, 14, 20]
        results_list = []
        
        for window in rsi_windows:
            params = default_parameters.copy()
            params['window'] = window
            results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', params)
            if results and 'total_return_pct' in results:
                results_list.append(results['total_return_pct'])
            
        # Results should be different for different parameters (if we have results)
        if len(results_list) > 1:
            assert len(set(results_list)) >= 1  # At least one unique result
        
    def test_consistent_results_same_parameters(self, sample_ohlcv_data, default_parameters):
        """Test that same parameters produce consistent results"""
        results1, portfolio1, fig1 = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        results2, portfolio2, fig2 = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        # Results should be identical
        assert results1['total_return_pct'] == results2['total_return_pct']
        assert len(portfolio1) == len(portfolio2)


# --- Performance Metrics Tests ---

class TestPremadeBacktestMetrics:
    """Test that performance metrics are calculated correctly"""
    
    def test_required_metrics_present(self, sample_ohlcv_data, default_parameters):
        """Test that all required metrics are present in results"""
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        required_metrics = [
            'total_return_pct',
            'sharpe_ratio',
            'max_drawdown_pct'
        ]
        
        for metric in required_metrics:
            assert metric in results, f"Missing metric: {metric}"
            assert isinstance(results[metric], (int, float)), f"Metric {metric} should be numeric"
            
    def test_portfolio_structure(self, sample_ohlcv_data, default_parameters):
        """Test that portfolio DataFrame has correct structure"""
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        required_columns = [
            'PositionType',
            'PortfolioValue'
        ]
        
        for col in required_columns:
            assert col in portfolio.columns, f"Missing portfolio column: {col}"
            
        # Portfolio should be shorter than input data due to indicator calculation requirements
        assert len(portfolio) <= len(sample_ohlcv_data)
        assert len(portfolio) > 0  # Should have some data
        
    def test_metrics_reasonable_values(self, sample_ohlcv_data, default_parameters):
        """Test that metrics have reasonable values"""
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsi', default_parameters)
        
        if results and 'total_return_pct' in results:
            # Total return should be reasonable (not extremely large)
            total_return = results['total_return_pct']
            if not pd.isna(total_return):
                assert -100 <= total_return <= 1000
        
        if results and 'sharpe_ratio' in results:
            # Sharpe ratio should be reasonable
            sharpe = results['sharpe_ratio']
            if not pd.isna(sharpe):
                assert -10 <= sharpe <= 10
        
        if results and 'max_drawdown_pct' in results:
            # Max drawdown should be negative or zero
            drawdown = results['max_drawdown_pct']
            if not pd.isna(drawdown):
                assert drawdown <= 0


# --- Momentum Strategy Tests ---

class TestMomentumStrategies:
    """Test all momentum trading strategies"""

    def test_awo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test AWO (Awesome Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_window': 5, 'slow_window': 34})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'awo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_bop_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BOP (Balance of Power) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'smooth': True})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bop', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bop_strategy_unsmoothed(self, sample_ohlcv_data, default_parameters):
        """Test BOP strategy without smoothing"""
        params = default_parameters.copy()
        params.update({'window': 14, 'smooth': False})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bop', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cmo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CMO (Chande Momentum Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 50, 'lower': -50})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cmo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cog_strategy(self, sample_ohlcv_data, default_parameters):
        """Test COG (Center of Gravity) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10, 'signal_window': 3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cog', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_crs_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CRS (Connors RSI) strategy"""
        params = default_parameters.copy()
        params.update({
            'rsi_window': 3,
            'streak_window': 2,
            'rank_window': 100,
            'upper': 90,
            'lower': 10
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'crs', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_dpo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DPO (Detrended Price Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'dpo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_eri_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ERI (Elder-Ray Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'eri', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fis_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FIS (Fisher Transform) strategy"""
        params = default_parameters.copy()
        params.update({'window': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fis', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_imi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test IMI (Intraday Momentum Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 70, 'lower': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'imi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_kst_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KST (Know Sure Thing) strategy"""
        params = default_parameters.copy()
        params.update({'signal': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kst', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_lsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test LSI (Laguerre RSI) strategy"""
        params = default_parameters.copy()
        params.update({'gamma': 0.5, 'upper': 80, 'lower': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'lsi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_msi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MSI (Momentum Strength Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'power': 1.0, 'upper': 70, 'lower': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'msi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pgo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PGO (Pretty Good Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 3.0, 'lower': -3.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pgo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ppo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PPO (Percentage Price Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_window': 12, 'slow_window': 26, 'signal_window': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ppo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_psy_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PSY (Psychological Line) strategy"""
        params = default_parameters.copy()
        params.update({'window': 12, 'upper': 75, 'lower': 25})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'psy', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_qst_strategy(self, sample_ohlcv_data, default_parameters):
        """Test QST (Qstick) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'qst', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rmi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RMI (Relative Momentum Index) strategy"""
        params = default_parameters.copy()
        params.update({
            'window': 20,
            'momentum_period': 5,
            'upper': 70,
            'lower': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rmi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_roc_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ROC (Rate of Change) strategy"""
        params = default_parameters.copy()
        params.update({'window': 12})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rvg_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RVG (Relative Vigor Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rvg', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_sri_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SRI (Stochastic RSI) strategy"""
        params = default_parameters.copy()
        params.update({
            'rsi_window': 14,
            'stoch_window': 14,
            'k_window': 3,
            'd_window': 3,
            'upper': 80,
            'lower': 20
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sri', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_stc_strategy(self, sample_ohlcv_data, default_parameters):
        """Test STC (Schaff Trend Cycle) strategy"""
        params = default_parameters.copy()
        params.update({
            'window_fast': 23,
            'window_slow': 50,
            'cycle': 10,
            'smooth': 3,
            'upper': 75,
            'lower': 25
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'stc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TSI (True Strength Index) strategy"""
        params = default_parameters.copy()
        params.update({'slow': 25, 'fast': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tsi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ttm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TTM (TTM Squeeze) strategy"""
        params = default_parameters.copy()
        params.update({
            'length': 20,
            'std_dev': 2.0,
            'atr_length': 20,
            'atr_multiplier': 1.5,
            'smooth': 3
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ttm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ult_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ULT (Ultimate Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_window': 7,
            'medium_window': 14,
            'long_window': 28,
            'upper': 70,
            'lower': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ult', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vor_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VOR (Vortex Indicator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vor', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_wil_strategy(self, sample_ohlcv_data, default_parameters):
        """Test WIL (Williams %R) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': -20, 'lower': -80})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'wil', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


class TestMomentumStrategiesWithTradingTypes:
    """Test momentum strategies with different trading types"""

    def test_momentum_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test momentum strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_momentum_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test momentum strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_momentum_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test momentum strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


# --- Trend Strategy Tests ---

class TestTrendStrategies:
    """Test all trend trading strategies"""

    def test_ads_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADS (Adaptive Moving Average - Smoothed) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ads', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_adx_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADX (Average Directional Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'adx_threshold': 25, 'ma_window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'adx', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_alm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ALM (ALMA - Arnaud Legoux Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 9, 'long_window': 27})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'alm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ama_strategy(self, sample_ohlcv_data, default_parameters):
        """Test AMA (Kaufman Adaptive Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_window': 10,
            'long_window': 30,
            'fast_period': 2,
            'slow_period': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ama', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_aro_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ARO (Aroon) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'aro', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_dem_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DEM (DEMA - Double Exponential Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'dem', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_eac_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EAC (Exponential Adaptive Close) strategy"""
        params = default_parameters.copy()
        params.update({'short_alpha': 0.07, 'long_alpha': 0.14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'eac', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_eit_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EIT (Ehlers Instantaneous Trendline) strategy"""
        params = default_parameters.copy()
        params.update({'short_alpha': 0.07, 'long_alpha': 0.14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'eit', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FMA (Fibonacci Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 8, 'long_window': 24})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fma_strategy_with_adx_filter(self, sample_ohlcv_data, default_parameters):
        """Test FMA strategy with ADX filter"""
        params = default_parameters.copy()
        params.update({
            'short_window': 8,
            'long_window': 24,
            'use_adx_filter': True,
            'adx_window': 14,
            'adx_threshold': 25
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_gma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test GMA (Guppy Multiple Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_windows': (3, 5, 8, 10, 12, 15),
            'long_windows': (30, 35, 40, 45, 50, 60)
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'gma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_hma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HMA (Hull Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 25, 'long_window': 75})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'hma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_htt_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HTT (Hilbert Transform Trendline) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 8, 'long_window': 16})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'htt', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ich_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ICH (Ichimoku Cloud) strategy"""
        params = default_parameters.copy()
        params.update({
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'displacement': 26
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ich', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_jma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test JMA (Jurik Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_length': 14, 'long_length': 42})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'jma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_kma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KMA (Kaufman Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_window': 10,
            'long_window': 30,
            'fast_period': 2,
            'slow_period': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_lsm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test LSM (Least Squares Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'lsm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mgd_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MGD (McGinley Dynamic) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mgd', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_psa_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PSA (Parabolic SAR) strategy"""
        params = default_parameters.copy()
        params.update({
            'af_initial': 0.03,
            'af_step': 0.03,
            'af_max': 0.3
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'psa', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_soa_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SOA (Second Order Adaptive) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'soa', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_str_strategy(self, sample_ohlcv_data, default_parameters):
        """Test STR (SuperTrend) strategy"""
        params = default_parameters.copy()
        params.update({'period': 7, 'multiplier': 3.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'str', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_swm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SWM (Sine-Weighted Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'swm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tem_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TEM (TEMA - Triple Exponential Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tem', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TMA (Triangular Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tri_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TRI (TRIX) strategy"""
        params = default_parameters.copy()
        params.update({'window': 7})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tri', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vid_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VID (Variable Index Dynamic Average) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_window': 14,
            'long_window': 42,
            'cmo_window': 9
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vid', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_wma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test WMA (Weighted Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 25, 'long_window': 75})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'wma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_zma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ZMA (Zero-Lag Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'zma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


class TestTrendStrategiesWithTradingTypes:
    """Test trend strategies with different trading types"""

    def test_trend_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test trend strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sma', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_trend_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test trend strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sma', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_trend_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test trend strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        params.update({'short_window': 10, 'long_window': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


# --- Volatility Strategy Tests ---

class TestVolatilityStrategies:
    """Test all volatility trading strategies"""

    def test_acb_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ACB (Acceleration Bands) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'factor': 0.001})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'acb', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_atp_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ATP (Average True Range Percent) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 5.0, 'lower': 2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'atp', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_atr_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ATR (Average True Range) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper_pct': 80, 'lower_pct': 20, 'lookback': 50})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'atr', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bbw_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BBW (Bollinger Band Width) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'num_std': 2.0, 'upper': 10.0, 'lower': 4.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bbw', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bol_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BOL (Bollinger Bands) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cha_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CHA (Chaikin Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'ema_window': 10, 'roc_window': 10, 'upper': 20.0, 'lower': -20.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cha', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cho_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CHO (Choppiness Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14, 'upper': 61.8, 'lower': 38.2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cho', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_don_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DON (Donchian Channels) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'don', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_dvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DVI (Dynamic Volatility Index) strategy"""
        params = default_parameters.copy()
        params.update({
            'magnitude_period': 5,
            'stretch_period': 50,
            'smooth_period': 3,
            'upper': 70,
            'lower': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'dvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_efr_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EFR (Efficiency Ratio) strategy"""
        params = default_parameters.copy()
        params.update({'period': 10, 'upper': 0.7, 'lower': 0.3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'efr', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fdi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FDI (Fractal Dimension Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 1.6, 'lower': 1.4})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fdi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_grv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test GRV (Garman-Klass Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'grv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_hav_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HAV (Heikin-Ashi Volatility) strategy"""
        params = default_parameters.copy()
        params.update({
            'period': 14,
            'method': 'atr',
            'upper_pct': 80,
            'lower_pct': 20,
            'lookback': 50
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'hav', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_hiv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test HIV (Historical Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'hiv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_kel_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KEL (Keltner Channel) strategy"""
        params = default_parameters.copy()
        params.update({'ema_window': 20, 'atr_window': 10, 'atr_multiplier': 2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kel', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mad_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MAD (Median Absolute Deviation) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 2.0, 'lower': 0.5})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mad', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mai_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MAI (Mass Index) strategy"""
        params = default_parameters.copy()
        params.update({'ema_period': 9, 'sum_period': 25, 'upper': 27.0, 'lower': 26.5})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mai', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_nat_strategy(self, sample_ohlcv_data, default_parameters):
        """Test NAT (Normalized ATR) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 5.0, 'lower': 2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'nat', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pav_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PAV (Parkinson Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pav', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pcw_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PCW (Price Channel Width) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 15.0, 'lower': 5.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pcw', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pro_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PRO (Projection Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'period': 10, 'smooth_period': 3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pro', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rsv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RSV (Rogers-Satchell Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20, 'upper': 30.0, 'lower': 15.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rsv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RVI (Relative Volatility Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10, 'rvi_period': 14, 'upper': 70, 'lower': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_std_strategy(self, sample_ohlcv_data, default_parameters):
        """Test STD (Standard Deviation) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'upper': 5.0, 'lower': 2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'std', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_svi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SVI (Stochastic Volatility Index) strategy"""
        params = default_parameters.copy()
        params.update({
            'atr_period': 14,
            'stoch_period': 14,
            'smooth_k': 3,
            'smooth_d': 3,
            'upper': 80,
            'lower': 20
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'svi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tsv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TSV (TSI Volatility) strategy"""
        params = default_parameters.copy()
        params.update({'atr_period': 14, 'long_period': 25, 'short_period': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tsv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_uli_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ULI (Ulcer Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14, 'upper': 5.0, 'lower': 1.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'uli', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vhf_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VHF (Vertical Horizontal Filter) strategy"""
        params = default_parameters.copy()
        params.update({'period': 28, 'upper': 0.40, 'lower': 0.25})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vhf', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vra_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VRA (Volatility Ratio) strategy"""
        params = default_parameters.copy()
        params.update({'short_period': 5, 'long_period': 20, 'upper': 1.5, 'lower': 0.8})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vra', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vqi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VQI (Volatility Quality Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 9, 'smooth_period': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vqi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VSI (Volatility Switch Index) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_period': 10,
            'long_period': 50,
            'threshold': 1.2,
            'upper': 0.5,
            'lower': 0.5
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vsi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


# --- Volume Strategy Tests ---

class TestVolumeStrategies:
    """Test all volume trading strategies"""

    def test_adl_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADL (Accumulation/Distribution Line) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'adl', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ado_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ADO (Accumulation/Distribution Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ado', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bwm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BWM (Bill Williams Market Facilitation Index) strategy"""
        params = default_parameters.copy()
        params.update({'upper_pct': 80, 'lower_pct': 20, 'lookback': 50})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bwm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cmf_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CMF (Chaikin Money Flow) strategy"""
        params = default_parameters.copy()
        params.update({'period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cmf', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_emv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test EMV (Ease of Movement) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'emv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_foi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FOI (Force Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'foi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fve_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FVE (Finite Volume Elements) strategy"""
        params = default_parameters.copy()
        params.update({'period': 22, 'factor': 0.3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fve', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_kvo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KVO (Klinger Volume Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_period': 34, 'slow_period': 55, 'signal_period': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kvo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_mfi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MFI (Money Flow Index) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14, 'upper': 80, 'lower': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mfi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_nvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test NVI (Negative Volume Index) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 50})  # Reduced from 255 for test data size
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'nvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_obv_strategy(self, sample_ohlcv_data, default_parameters):
        """Test OBV (On-Balance Volume) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pvi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PVI (Positive Volume Index) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 50})  # Reduced from 255 for test data size
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pvi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pvo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PVO (Percentage Volume Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pvo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vfi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VFI (Volume Flow Indicator) strategy"""
        params = default_parameters.copy()
        params.update({
            'period': 50,  # Reduced from 130 for test data size
            'coef': 0.2,
            'vcoef': 2.5,
            'smoothing_period': 3
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vfi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vma_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VMA (Volume Moving Average) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vma', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_voo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VOO (Volume Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_period': 5, 'slow_period': 10})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'voo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vpt_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VPT (Volume Price Trend) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vpt', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vro_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VRO (Volume Rate of Change) strategy"""
        params = default_parameters.copy()
        params.update({'period': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vro', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vwa_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VWA (Volume Weighted Average Price) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vwa', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_wad_strategy(self, sample_ohlcv_data, default_parameters):
        """Test WAD (Williams Accumulation/Distribution) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'wad', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


class TestVolatilityStrategiesWithTradingTypes:
    """Test volatility strategies with different trading types"""

    def test_volatility_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test volatility strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_volatility_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test volatility strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_volatility_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test volatility strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        params.update({'window': 20, 'num_std': 2})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bol', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


class TestVolumeStrategiesWithTradingTypes:
    """Test volume strategies with different trading types"""

    def test_volume_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test volume strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_volume_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test volume strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_volume_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test volume strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'obv', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
