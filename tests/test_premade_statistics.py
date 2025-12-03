"""
Tests for statistics premade strategies.
"""
import pandas as pd
from simple_trade.run_premade_strategies import run_premade_trade


class TestStatisticsStrategies:
    """Test all statistics trading strategies"""

    def test_std_strategy(self, sample_ohlcv_data, default_parameters):
        """Test STD (Standard Deviation) strategy"""
        params = default_parameters.copy()
        params.update({
            'window': 20, 'upper_pct': 80, 'lower_pct': 20, 'lookback': 100
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'std', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_kur_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KUR (Kurtosis) strategy"""
        params = default_parameters.copy()
        params.update({
            'window': 20, 'upper_pct': 80, 'lower_pct': 20, 'lookback': 100
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kur', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_mab_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MAB (Mean Absolute Deviation) strategy"""
        params = default_parameters.copy()
        params.update({
            'window': 20, 'upper_pct': 80, 'lower_pct': 20, 'lookback': 100
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'mab', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_med_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MED (Rolling Median) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'med', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_qua_strategy(self, sample_ohlcv_data, default_parameters):
        """Test QUA (Rolling Quantile) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'upper_quantile': 0.75, 'lower_quantile': 0.25})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'qua', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_skw_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SKW (Skewness) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'upper_threshold': 0.5, 'lower_threshold': -0.5})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'skw', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_var_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VAR (Variance) strategy"""
        params = default_parameters.copy()
        params.update({
            'window': 20, 'upper_pct': 80, 'lower_pct': 20, 'lookback': 100
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'var', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_zsc_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ZSC (Z-Score) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'upper_threshold': 2.0, 'lower_threshold': -2.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'zsc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results


class TestStatisticsStrategiesWithTradingTypes:
    """Test statistics strategies with different trading types"""

    def test_statistics_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test statistics strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'zsc', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_statistics_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test statistics strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'zsc', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_statistics_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test statistics strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'zsc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
